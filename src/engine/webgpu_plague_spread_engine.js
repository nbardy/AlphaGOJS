/**
 * WebGPU batched plague spread — core engine (WGSL source passed in so Node benchmarks
 * can read plague_env.wgsl from disk; browser uses webgpu_plague_spread.js which bundles it).
 * Only the `spread_pass` entry point is used; the module may contain other env passes.
 */

import { copyGpuBufferToUint32 } from './webgpu_copy_read_u32.js';

/** Unpacked u32 cell codes (matches WGSL). */
export const CELL_EMPTY = 0;
export const CELL_P1 = 1;
export const CELL_P2 = 2;
export const CELL_WALL = 3;

export function encodePlagueWallsBoardToPacked(int8Board) {
  var n = int8Board.length;
  var out = new Uint32Array(n);
  for (var i = 0; i < n; i++) {
    var v = int8Board[i];
    if (v === 0) out[i] = CELL_EMPTY;
    else if (v === 1) out[i] = CELL_P1;
    else if (v === -1) out[i] = CELL_P2;
    else if (v === 2) out[i] = CELL_WALL;
    else out[i] = CELL_EMPTY;
  }
  return out;
}

export function decodePackedToPlagueWallsInt8(packed) {
  var n = packed.length;
  var out = new Int8Array(n);
  for (var i = 0; i < n; i++) {
    var u = packed[i];
    if (u === CELL_EMPTY) out[i] = 0;
    else if (u === CELL_P1) out[i] = 1;
    else if (u === CELL_P2) out[i] = -1;
    else if (u === CELL_WALL) out[i] = 2;
    else out[i] = 0;
  }
  return out;
}

export class WebGPUPlagueSpreadEngine {
  /**
   * @param {GPUDevice} device
   * @param {{ rows: number, cols: number, numGames: number }} config
   * @param {string} wgslSource full WGSL source (e.g. contents of plague_env.wgsl)
   */
  constructor(device, config, wgslSource) {
    if (!wgslSource || typeof wgslSource !== 'string') {
      throw new Error('WebGPUPlagueSpreadEngine: wgslSource string required');
    }
    this.device = device;
    this.rows = config.rows | 0;
    this.cols = config.cols | 0;
    this.numGames = config.numGames | 0;
    this.boardSize = this.rows * this.cols;
    this._tick = 0 >>> 0;

    var bytes = this.boardSize * this.numGames * 4;
    this.bufA = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    this.bufB = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    var shaderModule = device.createShaderModule({ code: wgslSource });
    var bindLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
      compute: { module: shaderModule, entryPoint: 'spread_pass' }
    });

    this._bindReadA = device.createBindGroup({
      layout: bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bufA } },
        { binding: 2, resource: { buffer: this.bufB } }
      ]
    });
    this._bindReadB = device.createBindGroup({
      layout: bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bufB } },
        { binding: 2, resource: { buffer: this.bufA } }
      ]
    });

    this._readIsA = true;
    this._workgroups = Math.ceil((this.boardSize * this.numGames) / 64);
    this._writeUniform();
  }

  _writeUniform() {
    var u = new Uint32Array([
      this.rows >>> 0,
      this.cols >>> 0,
      this._tick >>> 0,
      this.numGames >>> 0
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, u);
  }

  /**
   * Benchmarks / tests: next uploadPacked + spread sequence uses tick 0 and read side A
   * (deterministic starting point without reallocating buffers).
   */
  resetSimulationState() {
    this._tick = 0 >>> 0;
    this._readIsA = true;
    this._writeUniform();
  }

  uploadPacked(packed) {
    var expected = this.boardSize * this.numGames;
    if (packed.length !== expected) {
      throw new Error('uploadPacked: expected ' + expected + ' cells, got ' + packed.length);
    }
    var readBuf = this._readIsA ? this.bufA : this.bufB;
    this.device.queue.writeBuffer(readBuf, 0, packed);
  }

  uploadGameSlot(slot, packedOneGame) {
    if (packedOneGame.length !== this.boardSize) {
      throw new Error('uploadGameSlot: expected boardSize ' + this.boardSize);
    }
    var readBuf = this._readIsA ? this.bufA : this.bufB;
    var offset = slot * this.boardSize * 4;
    this.device.queue.writeBuffer(readBuf, offset, packedOneGame);
  }

  spread() {
    this._writeUniform();
    var enc = this.device.createCommandEncoder();
    var pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this._readIsA ? this._bindReadA : this._bindReadB);
    pass.dispatchWorkgroups(this._workgroups);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this._readIsA = !this._readIsA;
    this._tick = (this._tick + 1) >>> 0;
  }

  async spreadAndSync() {
    this.spread();
    await this.device.queue.onSubmittedWorkDone();
  }

  async downloadPacked() {
    var readBuf = this._readIsA ? this.bufA : this.bufB;
    var count = this.boardSize * this.numGames;
    return copyGpuBufferToUint32(this.device, readBuf, 0, count);
  }

  dispose() {
    try { this.bufA.destroy(); } catch (e) {}
    try { this.bufB.destroy(); } catch (e) {}
    try { this.uniformBuffer.destroy(); } catch (e) {}
    this.bufA = null;
    this.bufB = null;
    this.uniformBuffer = null;
  }
}

export async function requestPlagueSpreadDevice() {
  var gpu = typeof navigator !== 'undefined' && navigator.gpu ? navigator.gpu : null;
  if (!gpu) {
    throw new Error('WebGPU not available (navigator.gpu missing)');
  }
  var adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  var device = await adapter.requestDevice();
  return { adapter: adapter, device: device };
}
