import { generateKernel } from './kernel_template';
import { computeWeightLayout, type ArchConfig, type WeightLayout } from './arch_config';
import { CheckpointPool, type Checkpoint } from './checkpoint_pool';

export const CONFIG = {
  numBoards: 256,
  maxSteps: 600,
  ppoEpochs: 3,
  lr: 3e-4,
  beta1: 0.9,
  beta2: 0.999,
  epsAdam: 1e-8,
  epsilonClip: 0.2,
  c1Value: 0.5,
  c2Entropy: 0.01,
  gamma: 0.99,
  gaeLambda: 0.95,
  gradClip: 1.0,
  weightDecay: 0.01,
  seed: 42,
  wallDensity: 0.15,
  territoryBonus: 0.5,
  checkpointInterval: 20,
  evalFraction: 0.5
};

const K = 24;
const N = K * K;
const WPB = Math.ceil(N / 16);
const BOARD_STATE_STRIDE = 160 + 600 * 160;
const PARAMS_SIZE = 24 * 4;

export class GPUTrainer {
  device!: GPUDevice;
  pool = new CheckpointPool();

  archConfig: ArchConfig;
  layout: WeightLayout;

  paramsBuf!: GPUBuffer;
  boardsBuf!: GPUBuffer;
  embedBuf!: GPUBuffer;
  denseBuf!: GPUBuffer;
  readbackBuf!: GPUBuffer;

  initPipeline!: GPUComputePipeline;
  rolloutPipeline!: GPUComputePipeline;
  gaePipeline!: GPUComputePipeline;
  ppoPipeline!: GPUComputePipeline;

  bindGroup!: GPUBindGroup;
  gaeBindGroup!: GPUBindGroup;
  ppoBindGroup!: GPUBindGroup;
  initBindGroup!: GPUBindGroup;

  adamStep = 0;
  rollout = 0;

  onStats?: (stats: any) => void;
  onBoard?: (board: Uint32Array) => void;

  constructor(config: ArchConfig) {
    this.archConfig = config;
    this.layout = computeWeightLayout(config.D);
  }

  async init() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter");

    const hasF16 = adapter.features.has("shader-f16");
    if (!hasF16) {
      throw new Error("Your browser does not support the 'shader-f16' WebGPU feature. Please enable it in your browser flags (e.g. chrome://flags/#enable-webgpu-developer-features).");
    }
    
    this.device = await adapter.requestDevice({
      requiredFeatures: ["shader-f16"],
    });

    // Initialize checkpoint pool from IDB before checking for saved checkpoints
    await this.pool.init();

    const wgsl = generateKernel(this.archConfig.D);
    const shaderModule = this.device.createShaderModule({ code: wgsl, label: `plague_ppo_D${this.archConfig.D}` });

    console.log("Worker: Compiling initPipeline...");
    this.initPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "init_boards" },
    });

    console.log("Worker: Compiling rolloutPipeline...");
    this.rolloutPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "rollout_step" },
    });

    console.log("Worker: Compiling gaePipeline...");
    this.gaePipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "gae_scan" },
    });

    console.log("Worker: Compiling ppoPipeline (this might take a few seconds)...");
    this.ppoPipeline = await this.device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "ppo_step" },
    });
    console.log("Worker: All pipelines compiled successfully!");

    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    const READ = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    const B = CONFIG.numBoards;

    this.paramsBuf = this.device.createBuffer({ size: PARAMS_SIZE, usage: UNIFORM });
    this.boardsBuf = this.device.createBuffer({ size: B * BOARD_STATE_STRIDE, usage: STORAGE });
    
    // 6 slots: w, m, v for player, and w, m, v for opponent
    this.embedBuf = this.device.createBuffer({ size: this.layout.embedWeightCount * 6 * 2, usage: STORAGE });
    this.denseBuf = this.device.createBuffer({ size: this.layout.denseWeightCount * 6 * 4, usage: STORAGE });
    
    this.readbackBuf = this.device.createBuffer({ size: B * Math.max(BOARD_STATE_STRIDE, 4), usage: READ });

    // Initialize weights
    const denseWeights = new Float32Array(this.layout.denseWeightCount * 4);
    let embedWeights = typeof Float16Array !== "undefined" ? new Float16Array(this.layout.embedWeightCount * 4) : null;

    if (this.pool.checkpoints.length > 0) {
      const latest = this.pool.checkpoints[this.pool.checkpoints.length - 1];
      console.log(`Worker: Resuming from IDB checkpoint ${latest.id} at rollout ${latest.generation}`);
      this.rollout = latest.generation;
      
      // The saved dense weights are only length this.layout.denseWeightCount (just the w component)
      // but the buffer needs space for m, v, and opp. We just copy the weights into the front.
      denseWeights.set(latest.dense, 0);
      if (embedWeights && latest.embed) {
        embedWeights.set(latest.embed, 0);
      }
    } else {
      let rng = 12345;
      const nextRng = () => { rng = (rng * 747796405 + 2891336453) >>> 0; return rng / 4294967296; };
      for (let i = 0; i < this.layout.denseWeightCount; i++) {
        const u = nextRng(), v = nextRng();
        denseWeights[i] = Math.sqrt(-2 * Math.log(u + 1e-12)) * Math.cos(2 * Math.PI * v) * 0.1;
      }
      
      if (embedWeights) {
        for (let i = 0; i < this.layout.embedWeightCount; i++) {
          embedWeights[i] = (nextRng() - 0.5) * 0.02;
        }
      }
    }

    this.device.queue.writeBuffer(this.denseBuf, 0, denseWeights);
    if (embedWeights) {
      this.device.queue.writeBuffer(this.embedBuf, 0, embedWeights);
    }

    this.initBindGroup = this.device.createBindGroup({
      layout: this.initPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.boardsBuf } },
      ],
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.rolloutPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.boardsBuf } },
        { binding: 2, resource: { buffer: this.embedBuf } },
        { binding: 3, resource: { buffer: this.denseBuf } },
      ],
    });
    this.gaeBindGroup = this.device.createBindGroup({
      layout: this.gaePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.boardsBuf } },
      ],
    });
    this.ppoBindGroup = this.device.createBindGroup({
      layout: this.ppoPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: this.boardsBuf } },
        { binding: 2, resource: { buffer: this.embedBuf } },
        { binding: 3, resource: { buffer: this.denseBuf } },
      ],
    });
  }

  writeParams(step: number, useOpponent: boolean, eloScale: number) {
    const data = new ArrayBuffer(PARAMS_SIZE);
    const u32 = new Uint32Array(data);
    const f32 = new Float32Array(data);
    u32[0] = CONFIG.numBoards;
    u32[1] = step;
    u32[2] = CONFIG.seed + this.rollout;
    u32[3] = CONFIG.maxSteps;
    u32[4] = this.adamStep;
    u32[5] = useOpponent ? 1 : 0;
    f32[8] = CONFIG.lr;
    f32[9] = CONFIG.beta1;
    f32[10] = CONFIG.beta2;
    f32[11] = CONFIG.epsAdam;
    f32[12] = CONFIG.epsilonClip;
    f32[13] = CONFIG.c1Value;
    f32[14] = CONFIG.c2Entropy;
    f32[15] = CONFIG.gamma;
    f32[16] = CONFIG.gaeLambda;
    f32[17] = CONFIG.gradClip;
    f32[18] = CONFIG.weightDecay;
    f32[19] = eloScale;
    f32[20] = CONFIG.territoryBonus;
    f32[21] = CONFIG.wallDensity;
    this.device.queue.writeBuffer(this.paramsBuf, 0, data);
  }

  initBoards() {
    // We already wrote params in runStep before calling initBoards, 
    // but let's ensure params are up to date for wallDensity and seed.
    // The GPU kernel init_boards clears the board, generates walls, and sets player to 1.
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.initPipeline);
    pass.setBindGroup(0, this.initBindGroup);
    pass.dispatchWorkgroups(CONFIG.numBoards);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async runStep() {
    const start = performance.now();
    let useOpponent = false;
    let opponent: Checkpoint | null = null;
    let evalWinRate = -1;

    if (this.rollout > 0 && this.rollout % CONFIG.checkpointInterval === 0) {
      // Save checkpoint
      await this.saveCheckpoint();
    }

    if (Math.random() < CONFIG.evalFraction) {
      opponent = this.pool.sampleOpponent();
      if (opponent) {
        useOpponent = true;
        this.device.queue.writeBuffer(this.denseBuf, this.layout.denseWeightCount * 3 * 4, opponent.dense.buffer);
        this.device.queue.writeBuffer(this.embedBuf, this.layout.embedWeightCount * 3 * 2, opponent.embed.buffer);
      }
    }

    let eloScale = 1.0;
    if (useOpponent && opponent) {
      // Scale reward based on Elo difference
      const expected = 1 / (1 + Math.pow(10, (opponent.elo - this.pool.currentElo) / 400));
      eloScale = 1.0 / (expected + 0.1); // just a simple scale heuristic
    }

    this.writeParams(0, useOpponent, eloScale);
    this.initBoards();

    // Rollout
    for (let step = 0; step < CONFIG.maxSteps; step++) {
      this.writeParams(step, useOpponent, eloScale);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.rolloutPipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(CONFIG.numBoards);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      
      if (step % 20 === 0) {
        if (await this.allGamesDone()) break;
      }
    }
    
    // Update Elo if eval
    if (useOpponent && opponent) {
       const winRate = await this.getWinRate();
       evalWinRate = winRate;
       const isDraw = winRate === 0.5;
       const isWin = winRate > 0.5;
       this.pool.updateElo(opponent.elo, isWin, isDraw);
    }

    // Capture board for rendering
    await this.captureBoard();

    // GAE
    this.writeParams(0, useOpponent, eloScale);
    {
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.gaePipeline);
      pass.setBindGroup(0, this.gaeBindGroup);
      pass.dispatchWorkgroups(Math.ceil(CONFIG.numBoards / 64));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    }

    // PPO + Adam (Internal Loop)
    for (let epoch = 0; epoch < CONFIG.ppoEpochs; epoch++) {
      this.writeParams(0, useOpponent, eloScale);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.ppoPipeline);
      pass.setBindGroup(0, this.ppoBindGroup);
      pass.dispatchWorkgroups(CONFIG.numBoards);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      
    }

    await this.device.queue.onSubmittedWorkDone();
    const stepCounts = await this.readStepCounts();
    const totalSteps = stepCounts.reduce((sum, count) => sum + count, 0);
    // Use actual step counts from GPU instead of hardcoded approximation
    this.adamStep += Math.round(totalSteps / CONFIG.numBoards);
    const avgSteps = totalSteps / CONFIG.numBoards;
    const finalLoss = await this.readLoss();
    const finalEntropy = await this.readEntropy();
    const time = performance.now() - start;
    const gamesPerSecTrained = (CONFIG.numBoards / time) * 1000;
    const stepsPerSecTrained = time > 0 ? (totalSteps / time) * 1000 : 0;
    
    this.rollout++;
    
    if (this.onStats) {
      this.onStats({
         rollout: this.rollout,
         games: this.rollout * CONFIG.numBoards,
         loss: finalLoss,
         elo: this.pool.currentElo,
         timeMs: time,
         trainedGamesPerSec: gamesPerSecTrained,
         trainedStepsPerSec: stepsPerSecTrained,
         avgStepsPerGame: avgSteps,
         winRate: evalWinRate,
         entropy: finalEntropy
      });
    }
  }

  async allGamesDone() {
    const encoder = this.device.createCommandEncoder();
    for (let i = 0; i < CONFIG.numBoards; i++) {
      encoder.copyBufferToBuffer(this.boardsBuf, i * BOARD_STATE_STRIDE + 144, this.readbackBuf, i * 4, 4);
    }
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const flags = new Uint32Array(this.readbackBuf.getMappedRange().slice(0, CONFIG.numBoards * 4));
    const allDone = flags.every(f => f !== 0);
    this.readbackBuf.unmap();
    return allDone;
  }

  async getWinRate() {
    const encoder = this.device.createCommandEncoder();
    for (let i = 0; i < CONFIG.numBoards; i++) {
      encoder.copyBufferToBuffer(this.boardsBuf, i * BOARD_STATE_STRIDE, this.readbackBuf, i * WPB * 4, WPB * 4);
    }
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const packed = new Uint32Array(this.readbackBuf.getMappedRange().slice(0, CONFIG.numBoards * WPB * 4));
    
    let p1Wins = 0;
    let p2Wins = 0;
    
    for (let i = 0; i < CONFIG.numBoards; i++) {
      let p1 = 0;
      let p2 = 0;
      for (let w = 0; w < WPB; w++) {
        let word = packed[i * WPB + w];
        for (let c = 0; c < 16; c++) {
          const state = (word >> (c * 2)) & 3;
          if (state === 1) p1++;
          else if (state === 2) p2++;
        }
      }
      if (p1 > p2) p1Wins++;
      else if (p2 > p1) p2Wins++;
    }
    
    this.readbackBuf.unmap();
    const total = p1Wins + p2Wins;
    return total > 0 ? p1Wins / total : 0.5;
  }

  async captureBoard() {
    if (!this.onBoard) return;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.boardsBuf, 0, this.readbackBuf, 0, WPB * 4);
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const packed = new Uint32Array(this.readbackBuf.getMappedRange().slice(0, WPB * 4));
    const copy = new Uint32Array(packed);
    this.readbackBuf.unmap();
    if (this.rollout % 10 === 0) console.log("captureBoard packed:", copy[0], copy[1]);
    this.onBoard(copy);
  }

  async readLoss() {
    const encoder = this.device.createCommandEncoder();
    for (let i = 0; i < CONFIG.numBoards; i++) {
      encoder.copyBufferToBuffer(this.boardsBuf, i * BOARD_STATE_STRIDE + 160 + 8, this.readbackBuf, i * 4, 4);
    }
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    
    let sum = 0;
    if (typeof Float16Array !== "undefined") {
      const view = new Float16Array(this.readbackBuf.getMappedRange().slice(0, CONFIG.numBoards * 4));
      for (let i = 0; i < CONFIG.numBoards; i++) { sum += view[i * 2]; }
    }
    this.readbackBuf.unmap();
    return sum / CONFIG.numBoards;
  }

  async readEntropy() {
    // Mirror readLoss exactly: entropy is written to transitions[0]._pad (f16).
    // Transition header offset = 160; _pad sits at byte 14 within the transition.
    const encoder = this.device.createCommandEncoder();
    for (let i = 0; i < CONFIG.numBoards; i++) {
      encoder.copyBufferToBuffer(this.boardsBuf, i * BOARD_STATE_STRIDE + 160 + 14, this.readbackBuf, i * 4, 4);
    }
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ);

    let sum = 0;
    if (typeof Float16Array !== "undefined") {
      const view = new Float16Array(this.readbackBuf.getMappedRange().slice(0, CONFIG.numBoards * 4));
      for (let i = 0; i < CONFIG.numBoards; i++) { sum += view[i * 2]; }
    }
    this.readbackBuf.unmap();
    return sum / CONFIG.numBoards;
  }

  async readStepCounts() {
    const encoder = this.device.createCommandEncoder();
    for (let i = 0; i < CONFIG.numBoards; i++) {
      encoder.copyBufferToBuffer(this.boardsBuf, i * BOARD_STATE_STRIDE + 152, this.readbackBuf, i * 4, 4);
    }
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const steps = new Uint32Array(this.readbackBuf.getMappedRange().slice(0, CONFIG.numBoards * 4));
    const copy = new Uint32Array(steps);
    this.readbackBuf.unmap();
    return copy;
  }

  /** Read back current weights from GPU — needed for cross-architecture eval. */
  async readWeights(): Promise<{ dense: Float32Array; embed: Float16Array }> {
    const denseBytes = this.layout.denseWeightCount * 4;
    const embedBytes = this.layout.embedWeightCount * 2;
    const totalBytes = denseBytes + embedBytes;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.denseBuf, 0, this.readbackBuf, 0, denseBytes);
    encoder.copyBufferToBuffer(this.embedBuf, 0, this.readbackBuf, denseBytes, embedBytes);
    this.device.queue.submit([encoder.finish()]);
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const mapped = this.readbackBuf.getMappedRange(0, totalBytes);
    const dense = new Float32Array(new Float32Array(mapped, 0, this.layout.denseWeightCount));
    const embed = new Float16Array(new Float16Array(mapped, denseBytes, this.layout.embedWeightCount));
    this.readbackBuf.unmap();
    return { dense, embed };
  }

  async saveCheckpoint() {
    const denseBytes = this.layout.denseWeightCount * 4;
    const embedBytes = this.layout.embedWeightCount * 2;
    const checkpointBytes = denseBytes + embedBytes;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.denseBuf, 0, this.readbackBuf, 0, denseBytes);
    encoder.copyBufferToBuffer(this.embedBuf, 0, this.readbackBuf, denseBytes, embedBytes);
    this.device.queue.submit([encoder.finish()]);
    
    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    const mapped = this.readbackBuf.getMappedRange(0, checkpointBytes);
    const denseCopy = new Float32Array(new Float32Array(mapped, 0, this.layout.denseWeightCount));
    let embedCopy = new Float16Array(this.layout.embedWeightCount);
    if (typeof Float16Array !== "undefined") {
      embedCopy = new Float16Array(new Float16Array(mapped, denseBytes, this.layout.embedWeightCount));
    }
    
    this.readbackBuf.unmap();
    await this.pool.saveCheckpoint(embedCopy, denseCopy, this.rollout);
  }
}
