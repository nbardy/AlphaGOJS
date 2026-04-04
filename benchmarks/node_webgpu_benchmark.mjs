#!/usr/bin/env node
import { performance } from 'node:perf_hooks';
import {
  emitBenchmarkReport,
  prepareBenchmarkOutput
} from './benchmark_output.mjs';

function now() {
  return performance.now();
}

function fmt(n, d = 2) {
  return Number(n).toFixed(d);
}

function clampInt(v, fallback, min, max) {
  const n = Number.parseInt(v, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function percentile(sorted, p) {
  if (!sorted.length) return NaN;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function summarize(samples) {
  if (!samples.length) {
    return { count: 0, min: NaN, max: NaN, mean: NaN, median: NaN, p95: NaN };
  }
  const sorted = samples.slice().sort((a, b) => a - b);
  const sum = samples.reduce((acc, v) => acc + v, 0);
  return {
    count: samples.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / samples.length,
    median: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95)
  };
}

async function runSampled(warmup, runs, stepFn) {
  for (let i = 0; i < warmup; i++) {
    await stepFn();
  }
  const samples = [];
  for (let i = 0; i < runs; i++) {
    samples.push(await stepFn());
  }
  return { warmup, runs, samples, stats: summarize(samples) };
}

async function runDetailedSampled(warmup, runs, stepFn, keys) {
  for (let i = 0; i < warmup; i++) {
    await stepFn();
  }
  const byStage = {};
  keys.forEach((k) => { byStage[k] = []; });

  for (let i = 0; i < runs; i++) {
    const out = await stepFn();
    keys.forEach((k) => byStage[k].push(out[k]));
  }

  const summary = {};
  keys.forEach((k) => { summary[k] = summarize(byStage[k]); });
  return { warmup, runs, samples: byStage, summary };
}

function makeData(length, seed, scale) {
  const out = new Float32Array(length);
  let x = seed >>> 0;
  for (let i = 0; i < length; i++) {
    x = (1664525 * x + 1013904223) >>> 0;
    const u = x / 4294967296;
    out[i] = (u * 2 - 1) * scale;
  }
  return out;
}

function createStorageBufferFromData(device, data) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true
  });
  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

function createUniformU32(device, a, b, c, d) {
  const data = new Uint32Array([a >>> 0, b >>> 0, c >>> 0, (d || 0) >>> 0]);
  const buffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function createUniformAttn(device, t, d, scale) {
  const ab = new ArrayBuffer(16);
  const dv = new DataView(ab);
  dv.setUint32(0, t >>> 0, true);
  dv.setUint32(4, d >>> 0, true);
  dv.setFloat32(8, scale, true);
  dv.setUint32(12, 0, true);
  const buffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buffer, 0, ab);
  return buffer;
}

const MATMUL_SHADER = `
  struct Dims {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
  };

  @group(0) @binding(0) var<storage, read> a: array<f32>;
  @group(0) @binding(1) var<storage, read> b: array<f32>;
  @group(0) @binding(2) var<storage, read_write> c: array<f32>;
  @group(0) @binding(3) var<uniform> dims: Dims;

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= dims.m || col >= dims.n) {
      return;
    }
    var sum = 0.0;
    var i: u32 = 0u;
    loop {
      if (i >= dims.k) {
        break;
      }
      sum = sum + a[row * dims.k + i] * b[i * dims.n + col];
      i = i + 1u;
    }
    c[row * dims.n + col] = sum;
  }
`;

const ATTN_SCORES_SHADER = `
  struct Dims {
    t: u32,
    d: u32,
    scale: f32,
    _pad: u32,
  };

  @group(0) @binding(0) var<storage, read> q: array<f32>;
  @group(0) @binding(1) var<storage, read> k: array<f32>;
  @group(0) @binding(2) var<storage, read_write> scores: array<f32>;
  @group(0) @binding(3) var<uniform> dims: Dims;

  @compute @workgroup_size(8, 8, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= dims.t || col >= dims.t) {
      return;
    }

    var sum = 0.0;
    var i: u32 = 0u;
    loop {
      if (i >= dims.d) {
        break;
      }
      let qv = q[row * dims.d + i];
      let kv = k[col * dims.d + i];
      sum = sum + qv * kv;
      i = i + 1u;
    }

    scores[row * dims.t + col] = sum * dims.scale;
  }
`;

const SOFTMAX_SHADER = `
  struct Dims {
    t: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
  };

  @group(0) @binding(0) var<storage, read_write> scores: array<f32>;
  @group(0) @binding(1) var<uniform> dims: Dims;

  @compute @workgroup_size(64, 1, 1)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= dims.t) {
      return;
    }

    var maxv = -1e30;
    var c: u32 = 0u;
    loop {
      if (c >= dims.t) {
        break;
      }
      let idx = row * dims.t + c;
      maxv = max(maxv, scores[idx]);
      c = c + 1u;
    }

    var sum = 0.0;
    c = 0u;
    loop {
      if (c >= dims.t) {
        break;
      }
      let idx = row * dims.t + c;
      let e = exp(scores[idx] - maxv);
      scores[idx] = e;
      sum = sum + e;
      c = c + 1u;
    }

    let inv = 1.0 / max(sum, 1e-20);
    c = 0u;
    loop {
      if (c >= dims.t) {
        break;
      }
      let idx = row * dims.t + c;
      scores[idx] = scores[idx] * inv;
      c = c + 1u;
    }
  }
`;

function createWriteBufferBenchmark(device, bytesPerRound, rounds) {
  const payload = new Uint8Array(bytesPerRound);
  for (let i = 0; i < payload.length; i += 4096) payload[i] = i & 255;
  const gpuBuffer = device.createBuffer({
    size: bytesPerRound,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });

  return {
    bytesPerRound,
    rounds,
    totalBytes: bytesPerRound * rounds,
    async run() {
      const t0 = now();
      for (let i = 0; i < rounds; i++) {
        device.queue.writeBuffer(gpuBuffer, 0, payload);
      }
      await device.queue.onSubmittedWorkDone();
      return now() - t0;
    },
    dispose() {
      gpuBuffer.destroy();
    }
  };
}

function createDispatchBenchmark(device, elementCount, dispatches) {
  const shader = device.createShaderModule({
    code: `
      @group(0) @binding(0) var<storage, read_write> data: array<u32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (gid.x < ${elementCount}u) {
          data[gid.x] = data[gid.x] + 1u;
        }
      }
    `
  });

  const bytes = elementCount * 4;
  const dataBuf = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });

  const bindLayout = device.createBindGroupLayout({
    entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
    compute: { module: shader, entryPoint: 'main' }
  });
  const bind = device.createBindGroup({
    layout: bindLayout,
    entries: [{ binding: 0, resource: { buffer: dataBuf } }]
  });

  const groups = Math.ceil(elementCount / 64);

  return {
    elementCount,
    dispatches,
    bytes,
    async run() {
      const t0 = now();
      for (let i = 0; i < dispatches; i++) {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        device.queue.submit([enc.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      return now() - t0;
    },
    dispose() {
      dataBuf.destroy();
    }
  };
}

function createReadbackBenchmark(device, bytes) {
  const payload = new Uint8Array(bytes);
  for (let i = 0; i < payload.length; i += 4096) payload[i] = (i >> 3) & 255;

  const src = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  const read = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  return {
    bytes,
    async run() {
      const t0 = now();
      device.queue.writeBuffer(src, 0, payload);
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, read, 0, bytes);
      device.queue.submit([enc.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const firstByte = new Uint8Array(read.getMappedRange())[0];
      read.unmap();
      const t1 = now();
      return { ms: t1 - t0, firstByte };
    },
    dispose() {
      src.destroy();
      read.destroy();
    }
  };
}

function createTransformerBenchmark(device, seqLen, hiddenDim) {
  const t = seqLen;
  const d = hiddenDim;
  const scale = 1 / Math.sqrt(d);

  const input = makeData(t * d, 1337, 0.03);
  const wq = makeData(d * d, 1001, 0.02);
  const wk = makeData(d * d, 1002, 0.02);
  const wv = makeData(d * d, 1003, 0.02);
  const wo = makeData(d * d, 1004, 0.02);

  const inputBuf = createStorageBufferFromData(device, input);
  const wqBuf = createStorageBufferFromData(device, wq);
  const wkBuf = createStorageBufferFromData(device, wk);
  const wvBuf = createStorageBufferFromData(device, wv);
  const woBuf = createStorageBufferFromData(device, wo);

  const qBuf = device.createBuffer({ size: t * d * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const kBuf = device.createBuffer({ size: t * d * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const vBuf = device.createBuffer({ size: t * d * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const scoresBuf = device.createBuffer({ size: t * t * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const ctxBuf = device.createBuffer({ size: t * d * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const outBuf = device.createBuffer({ size: t * d * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const readBuf = device.createBuffer({ size: t * d * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const projDims = createUniformU32(device, t, d, d, 0);
  const scoreDims = createUniformAttn(device, t, d, scale);
  const softmaxDims = createUniformU32(device, t, 0, 0, 0);
  const ctxDims = createUniformU32(device, t, t, d, 0);

  const matmulBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
    ]
  });
  const scoreBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
    ]
  });
  const softmaxBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
    ]
  });

  const matmulPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [matmulBGL] }),
    compute: { module: device.createShaderModule({ code: MATMUL_SHADER }), entryPoint: 'main' }
  });
  const scorePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [scoreBGL] }),
    compute: { module: device.createShaderModule({ code: ATTN_SCORES_SHADER }), entryPoint: 'main' }
  });
  const softmaxPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [softmaxBGL] }),
    compute: { module: device.createShaderModule({ code: SOFTMAX_SHADER }), entryPoint: 'main' }
  });

  const qProjBind = device.createBindGroup({
    layout: matmulBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: wqBuf } },
      { binding: 2, resource: { buffer: qBuf } },
      { binding: 3, resource: { buffer: projDims } }
    ]
  });
  const kProjBind = device.createBindGroup({
    layout: matmulBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: wkBuf } },
      { binding: 2, resource: { buffer: kBuf } },
      { binding: 3, resource: { buffer: projDims } }
    ]
  });
  const vProjBind = device.createBindGroup({
    layout: matmulBGL,
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: wvBuf } },
      { binding: 2, resource: { buffer: vBuf } },
      { binding: 3, resource: { buffer: projDims } }
    ]
  });
  const scoreBind = device.createBindGroup({
    layout: scoreBGL,
    entries: [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kBuf } },
      { binding: 2, resource: { buffer: scoresBuf } },
      { binding: 3, resource: { buffer: scoreDims } }
    ]
  });
  const softmaxBind = device.createBindGroup({
    layout: softmaxBGL,
    entries: [
      { binding: 0, resource: { buffer: scoresBuf } },
      { binding: 1, resource: { buffer: softmaxDims } }
    ]
  });
  const ctxBind = device.createBindGroup({
    layout: matmulBGL,
    entries: [
      { binding: 0, resource: { buffer: scoresBuf } },
      { binding: 1, resource: { buffer: vBuf } },
      { binding: 2, resource: { buffer: ctxBuf } },
      { binding: 3, resource: { buffer: ctxDims } }
    ]
  });
  const outBind = device.createBindGroup({
    layout: matmulBGL,
    entries: [
      { binding: 0, resource: { buffer: ctxBuf } },
      { binding: 1, resource: { buffer: woBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: projDims } }
    ]
  });

  const groupsProjX = Math.ceil(d / 8);
  const groupsProjY = Math.ceil(t / 8);
  const groupsScoreX = Math.ceil(t / 8);
  const groupsScoreY = Math.ceil(t / 8);
  const groupsSoftmax = Math.ceil(t / 64);

  function encodeQKV(encoder) {
    let pass = encoder.beginComputePass();
    pass.setPipeline(matmulPipeline);
    pass.setBindGroup(0, qProjBind);
    pass.dispatchWorkgroups(groupsProjX, groupsProjY);
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(matmulPipeline);
    pass.setBindGroup(0, kProjBind);
    pass.dispatchWorkgroups(groupsProjX, groupsProjY);
    pass.end();

    pass = encoder.beginComputePass();
    pass.setPipeline(matmulPipeline);
    pass.setBindGroup(0, vProjBind);
    pass.dispatchWorkgroups(groupsProjX, groupsProjY);
    pass.end();
  }

  function encodeScores(encoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(scorePipeline);
    pass.setBindGroup(0, scoreBind);
    pass.dispatchWorkgroups(groupsScoreX, groupsScoreY);
    pass.end();
  }

  function encodeSoftmax(encoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(softmaxPipeline);
    pass.setBindGroup(0, softmaxBind);
    pass.dispatchWorkgroups(groupsSoftmax);
    pass.end();
  }

  function encodeContext(encoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(matmulPipeline);
    pass.setBindGroup(0, ctxBind);
    pass.dispatchWorkgroups(groupsProjX, groupsProjY);
    pass.end();
  }

  function encodeOutProj(encoder) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(matmulPipeline);
    pass.setBindGroup(0, outBind);
    pass.dispatchWorkgroups(groupsProjX, groupsProjY);
    pass.end();
  }

  function encodeWholeStep(encoder) {
    encodeQKV(encoder);
    encodeScores(encoder);
    encodeSoftmax(encoder);
    encodeContext(encoder);
    encodeOutProj(encoder);
  }

  return {
    seqLen: t,
    hiddenDim: d,
    async runComputeOnly() {
      const t0 = now();
      const encoder = device.createCommandEncoder();
      encodeWholeStep(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return now() - t0;
    },
    async runEndToEnd() {
      const t0 = now();
      device.queue.writeBuffer(inputBuf, 0, input);
      const encoder = device.createCommandEncoder();
      encodeWholeStep(encoder);
      encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, t * d * 4);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readBuf.mapAsync(GPUMapMode.READ);
      const first = new Float32Array(readBuf.getMappedRange())[0];
      readBuf.unmap();
      return { ms: now() - t0, firstValue: first };
    },
    async runDetailed() {
      const breakdown = {
        uploadMs: 0,
        qkvMs: 0,
        attnScoresMs: 0,
        softmaxMs: 0,
        contextMs: 0,
        outProjMs: 0,
        readbackMs: 0,
        totalMs: 0
      };
      const start = now();

      let t0 = now();
      device.queue.writeBuffer(inputBuf, 0, input);
      await device.queue.onSubmittedWorkDone();
      breakdown.uploadMs = now() - t0;

      t0 = now();
      let encoder = device.createCommandEncoder();
      encodeQKV(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      breakdown.qkvMs = now() - t0;

      t0 = now();
      encoder = device.createCommandEncoder();
      encodeScores(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      breakdown.attnScoresMs = now() - t0;

      t0 = now();
      encoder = device.createCommandEncoder();
      encodeSoftmax(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      breakdown.softmaxMs = now() - t0;

      t0 = now();
      encoder = device.createCommandEncoder();
      encodeContext(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      breakdown.contextMs = now() - t0;

      t0 = now();
      encoder = device.createCommandEncoder();
      encodeOutProj(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      breakdown.outProjMs = now() - t0;

      t0 = now();
      encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, t * d * 4);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readBuf.mapAsync(GPUMapMode.READ);
      const first = new Float32Array(readBuf.getMappedRange())[0];
      readBuf.unmap();
      breakdown.readbackMs = now() - t0;
      breakdown.totalMs = now() - start;
      breakdown.firstValue = first;
      return breakdown;
    },
    dispose() {
      [
        inputBuf, wqBuf, wkBuf, wvBuf, woBuf,
        qBuf, kBuf, vBuf, scoresBuf, ctxBuf, outBuf,
        readBuf, projDims, scoreDims, softmaxDims, ctxDims
      ].forEach((b) => {
        try { b.destroy(); } catch (e) {}
      });
    }
  };
}

async function ensureWebGPUGlobals() {
  if (globalThis?.navigator?.gpu || globalThis?.gpu) {
    return 'native';
  }

  // Fallback: Dawn-backed Node WebGPU package.
  // Source: https://github.com/dawn-gpu/node-webgpu
  let mod = null;
  try {
    mod = await import('webgpu');
  } catch (e) {
    return null;
  }
  if (!mod || typeof mod.create !== 'function') {
    return null;
  }

  if (mod.globals && typeof mod.globals === 'object') {
    Object.assign(globalThis, mod.globals);
  }
  if (!globalThis.navigator) globalThis.navigator = {};
  if (!globalThis.navigator.gpu) {
    globalThis.navigator.gpu = mod.create([]);
  }
  return 'node-webgpu';
}

function getGPU() {
  if (globalThis?.navigator?.gpu) return globalThis.navigator.gpu;
  if (globalThis?.gpu) return globalThis.gpu;
  return null;
}

async function initWebGPU() {
  const provider = await ensureWebGPUGlobals();
  const gpu = getGPU();
  if (!gpu) {
    throw new Error(
      'WebGPU API not found. Install `webgpu` (Dawn Node backend) or use a runtime with native WebGPU.'
    );
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter available');
  const device = await adapter.requestDevice();
  return { adapter, device, provider: provider || 'unknown' };
}

function parseArgs(argv) {
  const out = {
    warmup: 3,
    runs: 10,
    seqLen: 128,
    hiddenDim: 128,
    writeBytesPerRound: 16 * 1024 * 1024,
    writeRounds: 32,
    dispatchElements: 1 << 20,
    dispatches: 200,
    readbackBytes: 4 * 1024 * 1024
  };

  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const [key, value = ''] = arg.slice(2).split('=', 2);
    if (key === 'warmup') out.warmup = clampInt(value, out.warmup, 0, 20);
    else if (key === 'runs') out.runs = clampInt(value, out.runs, 1, 100);
    else if (key === 'seq') out.seqLen = clampInt(value, out.seqLen, 16, 512);
    else if (key === 'hidden') out.hiddenDim = clampInt(value, out.hiddenDim, 16, 512);
  }
  return out;
}

function gbps(totalBytes, ms) {
  return (totalBytes / (1024 ** 3)) / (ms / 1000);
}

function toFeatureList(features) {
  if (!features) return [];
  if (typeof features.values === 'function') return Array.from(features.values());
  if (typeof features[Symbol.iterator] === 'function') return Array.from(features);
  if (typeof features.keys === 'function') return Array.from(features.keys());
  if (typeof features === 'object') {
    return Object.keys(features).filter((k) => !!features[k]);
  }
  return [];
}

async function main() {
  const argv = process.argv.slice(2);
  const output = prepareBenchmarkOutput('node_webgpu_benchmark', argv);
  let writeBench = null;
  let dispatchBench = null;
  let readbackBench = null;
  let transformerBench = null;

  try {
    const cfg = parseArgs(argv);
    const { adapter, device, provider } = await initWebGPU();

    writeBench = createWriteBufferBenchmark(device, cfg.writeBytesPerRound, cfg.writeRounds);
    dispatchBench = createDispatchBenchmark(device, cfg.dispatchElements, cfg.dispatches);
    readbackBench = createReadbackBenchmark(device, cfg.readbackBytes);
    transformerBench = createTransformerBenchmark(device, cfg.seqLen, cfg.hiddenDim);

    const writeRes = await runSampled(cfg.warmup, cfg.runs, () => writeBench.run());
    const dispatchRes = await runSampled(cfg.warmup, cfg.runs, () => dispatchBench.run());
    const readbackRes = await runSampled(cfg.warmup, cfg.runs, async () => (await readbackBench.run()).ms);
    const txCompute = await runSampled(cfg.warmup, cfg.runs, () => transformerBench.runComputeOnly());
    const txE2E = await runSampled(cfg.warmup, cfg.runs, async () => (await transformerBench.runEndToEnd()).ms);

    const stageKeys = ['uploadMs', 'qkvMs', 'attnScoresMs', 'softmaxMs', 'contextMs', 'outProjMs', 'readbackMs', 'totalMs'];
    const detailedRuns = Math.max(3, Math.min(cfg.runs, 8));
    const txDetailed = await runDetailedSampled(cfg.warmup, detailedRuns, () => transformerBench.runDetailed(), stageKeys);

    const derived = {
      uploadGbps: {
        median: gbps(writeBench.totalBytes, writeRes.stats.median),
        p95_conservative: gbps(writeBench.totalBytes, writeRes.stats.p95)
      },
      dispatch: {
        medianDispatchesPerSec: dispatchBench.dispatches / (dispatchRes.stats.median / 1000),
        medianGigaElemsPerSec: ((dispatchBench.elementCount * dispatchBench.dispatches) / 1e9) / (dispatchRes.stats.median / 1000)
      },
      transformer: {
        computeOnlyStepsPerSec: 1000 / txCompute.stats.median,
        e2eStepsPerSec: 1000 / txE2E.stats.median,
        computeOnlyTokensPerSec: (cfg.seqLen * 1000) / txCompute.stats.median,
        e2eTokensPerSec: (cfg.seqLen * 1000) / txE2E.stats.median
      }
    };

    const results = {
      env: {
        node: process.version,
        provider,
        platform: process.platform,
        arch: process.arch,
        adapterFeatures: toFeatureList(adapter.features),
        limits: {
          maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
          maxStorageBufferBindingSize: Number(device.limits.maxStorageBufferBindingSize),
          maxBufferSize: Number(device.limits.maxBufferSize)
        },
        config: cfg,
        timestamp: new Date().toISOString()
      },
      benchmarks: {
        writeBuffer: writeRes,
        computeDispatch: dispatchRes,
        readback: readbackRes,
        transformerComputeOnly: txCompute,
        transformerEndToEnd: txE2E,
        transformerDetailedStages: txDetailed
      },
      derived
    };

    emitBenchmarkReport(output, results, [
      'Node WebGPU benchmark complete',
      'warmup=' + cfg.warmup + ' runs=' + cfg.runs + ' seq=' + cfg.seqLen + ' hidden=' + cfg.hiddenDim,
      'writeBuffer median=' + fmt(writeRes.stats.median, 2) + 'ms p95=' + fmt(writeRes.stats.p95, 2) + 'ms medianGBps=' + fmt(derived.uploadGbps.median, 3),
      'computeDispatch median=' + fmt(dispatchRes.stats.median, 2) + 'ms p95=' + fmt(dispatchRes.stats.p95, 2) + 'ms medianDispatchesPerSec=' + fmt(derived.dispatch.medianDispatchesPerSec, 1),
      'readback median=' + fmt(readbackRes.stats.median, 2) + 'ms p95=' + fmt(readbackRes.stats.p95, 2) + 'ms',
      'transformer computeOnly median=' + fmt(txCompute.stats.median, 2) + 'ms e2e median=' + fmt(txE2E.stats.median, 2) + 'ms e2eTokensPerSec=' + fmt(derived.transformer.e2eTokensPerSec, 1),
      'transformer stage medians: upload=' + fmt(txDetailed.summary.uploadMs.median, 2)
        + ' qkv=' + fmt(txDetailed.summary.qkvMs.median, 2)
        + ' scores=' + fmt(txDetailed.summary.attnScoresMs.median, 2)
        + ' softmax=' + fmt(txDetailed.summary.softmaxMs.median, 2)
        + ' context=' + fmt(txDetailed.summary.contextMs.median, 2)
        + ' out=' + fmt(txDetailed.summary.outProjMs.median, 2)
        + ' readback=' + fmt(txDetailed.summary.readbackMs.median, 2)
        + ' total=' + fmt(txDetailed.summary.totalMs.median, 2)
    ]);
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    console.error('Node WebGPU benchmark failed:', msg);
    process.exitCode = 1;
  } finally {
    if (writeBench) writeBench.dispose();
    if (dispatchBench) dispatchBench.dispose();
    if (readbackBench) readbackBench.dispose();
    if (transformerBench) transformerBench.dispose();
  }
}

main();
