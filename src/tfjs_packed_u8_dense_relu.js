/**
 * TF.js–compatible packed uint8 dense layer forward: y = ReLU((x·w) * scale + bias)
 * using WGSL packed_4x8_integer_dot_product on WebGPU when the device exposes it.
 *
 * Usage:
 *   import '@tensorflow/tfjs-backend-webgpu';
 *   import { registerPackedU8DenseReluKernels, packedU8DenseRelu } from './tfjs_packed_u8_dense_relu.js';
 *   registerPackedU8DenseReluKernels();
 *   await tf.setBackend('webgpu'); // or 'cpu'
 *
 * Fast shader path: the WebGPU device must include GPUFeatureName
 *   "packed-4x8-integer-dot-product" (add it in requestDevice.requiredFeatures).
 *   Stock tfjs-backend-webgpu does not request it; without it, this op falls back to
 *   a JS dot loop (still correct, slow, may sync GPU tensors via dataSync).
 */

import { engine, registerKernel, util } from '@tensorflow/tfjs';

/** @type {string} */
export const PACKED_U8_DENSE_RELU_KERNEL = 'PackedU8DenseRelu';

/** WebGPU optional feature (requestDevice). */
export const PACKED_DOT4_GPU_FEATURE = 'packed-4x8-integer-dot-product';

let kernelsRegistered = false;
/** @type {WeakMap<GPUDevice, GPUComputePipeline>} */
const pipelineCache = new WeakMap();

export function packU8x4(a, b, c, d) {
  return (
    (a & 0xff) |
    ((b & 0xff) << 8) |
    ((c & 0xff) << 16) |
    ((d & 0xff) << 24)
  ) >>> 0;
}

export function packVectorU8ToInt32Array(vec) {
  const packedLen = Math.ceil(vec.length / 4);
  const out = new Int32Array(packedLen);
  for (let i = 0; i < packedLen; i++) {
    const base = i * 4;
    out[i] = packU8x4(
      vec[base] ?? 0,
      vec[base + 1] ?? 0,
      vec[base + 2] ?? 0,
      vec[base + 3] ?? 0
    ) | 0;
  }
  return out;
}

export function packWeightsU8RowMajorInt32(weights, outDim, inDim) {
  const inDimPacked = Math.ceil(inDim / 4);
  const out = new Int32Array(outDim * inDimPacked);
  for (let j = 0; j < outDim; j++) {
    for (let k = 0; k < inDimPacked; k++) {
      const base = j * inDim + k * 4;
      out[j * inDimPacked + k] = packU8x4(
        weights[base] ?? 0,
        weights[base + 1] ?? 0,
        weights[base + 2] ?? 0,
        weights[base + 3] ?? 0
      ) | 0;
    }
  }
  return out;
}

export function hasWgslPackedDot4() {
  return (
    typeof navigator !== 'undefined' &&
    navigator.gpu &&
    navigator.gpu.wgslLanguageFeatures &&
    navigator.gpu.wgslLanguageFeatures.has('packed_4x8_integer_dot_product')
  );
}

export function webgpuDeviceSupportsPackedDot4(device) {
  return !!(device && device.features && device.features.has(PACKED_DOT4_GPU_FEATURE));
}

function dot4U8PackedJs(a, b) {
  const au = a >>> 0;
  const bu = b >>> 0;
  let acc = 0;
  for (let i = 0; i < 4; i++) {
    acc += ((au >>> (8 * i)) & 0xff) * ((bu >>> (8 * i)) & 0xff);
  }
  return acc;
}

const WGSL_MAIN = `
enable packed_4x8_integer_dot_product;

struct Meta {
  in_dim: u32,
  out_dim: u32,
  in_dim_packed: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> yOut: array<f32>;
@group(0) @binding(1) var<storage, read> xPacked: array<u32>;
@group(0) @binding(2) var<storage, read> wPacked: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<uniform> meta: Meta;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= meta.out_dim) { return; }

  var acc: f32 = 0.0;
  let base = j * meta.in_dim_packed;
  for (var k: u32 = 0u; k < meta.in_dim_packed; k = k + 1u) {
    let x4 = xPacked[k];
    let w4 = wPacked[base + k];
    acc = acc + f32(dot4U8Packed(x4, w4));
  }
  yOut[j] = max(acc * scales[j] + bias[j], 0.0);
}
`;

function getOrCreatePipeline(device) {
  let p = pipelineCache.get(device);
  if (!p) {
    const module = device.createShaderModule({ code: WGSL_MAIN, label: 'PackedU8DenseRelu' });
    p = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' },
      label: 'PackedU8DenseRelu',
    });
    pipelineCache.set(device, p);
  }
  return p;
}

function assertShapes(xPacked, wPacked, scales, bias, inDim, outDim) {
  const inDimPacked = Math.ceil(inDim / 4);
  util.assert(xPacked.dtype === 'int32', () => 'xPacked must be int32 (packed u8 lanes as bits)');
  util.assert(wPacked.dtype === 'int32', () => 'wPacked must be int32');
  util.assert(scales.dtype === 'float32' && bias.dtype === 'float32', () => 'scales/bias must be float32');
  util.assert(
    util.arraysEqual(xPacked.shape, [inDimPacked]),
    () => `xPacked shape expected [${inDimPacked}], got [${xPacked.shape}]`
  );
  util.assert(
    util.arraysEqual(wPacked.shape, [outDim, inDimPacked]),
    () => `wPacked shape expected [${outDim}, ${inDimPacked}], got [${wPacked.shape}]`
  );
  util.assert(
    util.arraysEqual(scales.shape, [outDim]) && util.arraysEqual(bias.shape, [outDim]),
    () => 'scales and bias must have shape [outDim]'
  );
}

function packedU8DenseReluCpu(args) {
  const { inputs, backend, attrs } = args;
  const { xPacked, wPacked, scales, bias } = inputs;
  const inDim = attrs.inDim;
  const outDim = attrs.outDim;
  assertShapes(xPacked, wPacked, scales, bias, inDim, outDim);

  const inDimPacked = Math.ceil(inDim / 4);
  const xd = xPacked.dataSync();
  const wd = wPacked.dataSync();
  const sd = scales.dataSync();
  const bd = bias.dataSync();
  const out = new Float32Array(outDim);

  for (let j = 0; j < outDim; j++) {
    let acc = 0;
    const row = j * inDimPacked;
    for (let k = 0; k < inDimPacked; k++) {
      acc += dot4U8PackedJs(xd[k], wd[row + k]);
    }
    out[j] = Math.max(0, acc * sd[j] + bd[j]);
  }
  return backend.makeTensorInfo([outDim], 'float32', out);
}

function packedU8DenseReluWebGPU(args) {
  const { inputs, backend, attrs } = args;
  const { xPacked, wPacked, scales, bias } = inputs;
  const inDim = attrs.inDim;
  const outDim = attrs.outDim;
  assertShapes(xPacked, wPacked, scales, bias, inDim, outDim);

  if (!webgpuDeviceSupportsPackedDot4(backend.device)) {
    const inDimPacked = Math.ceil(inDim / 4);
    const xd = xPacked.dataSync();
    const wd = wPacked.dataSync();
    const sd = scales.dataSync();
    const bd = bias.dataSync();
    const out = new Float32Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let acc = 0;
      const row = j * inDimPacked;
      for (let k = 0; k < inDimPacked; k++) {
        acc += dot4U8PackedJs(xd[k], wd[row + k]);
      }
      out[j] = Math.max(0, acc * sd[j] + bd[j]);
    }
    return backend.makeTensorInfo([outDim], 'float32', out);
  }

  const inDimPacked = Math.ceil(inDim / 4);
  const output = backend.makeTensorInfo([outDim], 'float32');
  backend.uploadToGPU(output.dataId);
  backend.uploadToGPU(xPacked.dataId);
  backend.uploadToGPU(wPacked.dataId);
  backend.uploadToGPU(scales.dataId);
  backend.uploadToGPU(bias.dataId);

  const yBuf = backend.tensorMap.get(output.dataId).resource;
  const xBuf = backend.tensorMap.get(xPacked.dataId).resource;
  const wBuf = backend.tensorMap.get(wPacked.dataId).resource;
  const sBuf = backend.tensorMap.get(scales.dataId).resource;
  const bBuf = backend.tensorMap.get(bias.dataId).resource;

  const metaUniform = backend.makeUniforms([
    { type: 'uint32', data: [inDim >>> 0, outDim >>> 0, inDimPacked >>> 0, 0] },
  ]);

  const pipeline = getOrCreatePipeline(backend.device);
  const bindGroup = backend.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: yBuf } },
      { binding: 1, resource: { buffer: xBuf } },
      { binding: 2, resource: { buffer: wBuf } },
      { binding: 3, resource: { buffer: sBuf } },
      { binding: 4, resource: { buffer: bBuf } },
      {
        binding: 5,
        resource: { buffer: metaUniform.buffer, offset: metaUniform.offset, size: metaUniform.size },
      },
    ],
  });

  backend.ensureCommandEncoderReady();
  backend.endComputePassEncoder();
  const pass = backend.commandEncoder.beginComputePass({ label: 'PackedU8DenseRelu' });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(outDim / 64), 1, 1);
  pass.end();
  backend.submitQueue();

  return output;
}

/** Call once after backends are loaded (and before or after setBackend). Idempotent. */
export function registerPackedU8DenseReluKernels() {
  if (kernelsRegistered) {
    return;
  }
  kernelsRegistered = true;

  registerKernel({
    kernelName: PACKED_U8_DENSE_RELU_KERNEL,
    backendName: 'cpu',
    kernelFunc: packedU8DenseReluCpu,
  });

  registerKernel({
    kernelName: PACKED_U8_DENSE_RELU_KERNEL,
    backendName: 'webgpu',
    kernelFunc: packedU8DenseReluWebGPU,
  });
}

/**
 * @param {import('@tensorflow/tfjs').Tensor} xPacked int32 [ceil(inDim/4)]
 * @param {import('@tensorflow/tfjs').Tensor} wPacked int32 [outDim, ceil(inDim/4)]
 * @param {import('@tensorflow/tfjs').Tensor} scales float32 [outDim]
 * @param {import('@tensorflow/tfjs').Tensor} bias float32 [outDim]
 */
export function packedU8DenseRelu(xPacked, wPacked, scales, bias, inDim, outDim) {
  return engine().runKernel(PACKED_U8_DENSE_RELU_KERNEL, { xPacked, wPacked, scales, bias }, { inDim, outDim });
}
