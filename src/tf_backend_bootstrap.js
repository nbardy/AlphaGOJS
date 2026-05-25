/**
 * Prefer TensorFlow.js WebGPU when the browser exposes navigator.gpu, then WebGL, then CPU.
 * @tensorflow/tfjs-backend-webgpu is patched (patch-package): uploadToGPU uses queue.writeBuffer;
 * BufferManager never uses mappedAtCreation (WebKit/Safari rejects STORAGE+mappable create).
 * The default @tensorflow/tfjs bundle registers WebGL but does not load the WebGPU backend;
 * importing @tensorflow/tfjs-backend-webgpu here registers it so setBackend('webgpu') can succeed.
 *
 * URL override: ?tfBackend=webgpu|webgl|cpu|auto (aliases: wgpu, gl, wasm).
 *
 * In a dedicated Worker, skips WebGL (no canvas). Uses one export so webpack worker chunks
 * cannot drop a separate "worker-only" symbol during split-chunk / HMR.
 */
import './webgpu_queue_write_shim.js';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

var backendChosen = false;
var globalPreference = 'auto';

/**
 * @param {URLSearchParams|string|null|undefined} params
 * @returns {'auto'|'webgpu'|'webgl'|'cpu'}
 */
export function parseTfBackendQueryParam(params) {
  if (!params) return 'auto';
  var p = params;
  if (typeof params === 'string') {
    p = new URLSearchParams(params.replace(/^\?/, ''));
  }
  var raw = p.get('tfBackend');
  if (!raw) raw = p.get('tfbackend');
  if (!raw) return 'auto';
  var v = String(raw).trim().toLowerCase();
  if (v === 'webgpu' || v === 'wgpu' || v === 'gpu') return 'webgpu';
  if (v === 'webgl' || v === 'gl') return 'webgl';
  if (v === 'cpu' || v === 'wasm') return 'cpu';
  if (v === 'auto') return 'auto';
  return 'auto';
}

export function setTfBackendPreference(pref) {
  globalPreference = pref || 'auto';
  backendChosen = false;
}

/** Force re-probing backends (e.g. worker WebGPU → CPU fallback). */
export function resetTfBackendChoice() {
  backendChosen = false;
}

function tfBootstrapInDedicatedWorker() {
  try {
    return (
      typeof WorkerGlobalScope !== 'undefined' &&
      typeof self !== 'undefined' &&
      self instanceof WorkerGlobalScope
    );
  } catch (e) {
    return false;
  }
}

function normalizePreference(pref) {
  var p = pref || globalPreference || 'auto';
  if (p === 'webgpu' || p === 'webgl' || p === 'cpu' || p === 'auto') return p;
  return 'auto';
}

function logBackendInfo(name, inWorker) {
  if (typeof console !== 'undefined' && console.info) {
    console.info('[tf] backend: ' + name + (inWorker ? ' (worker)' : ''));
  }
}

function logBackendWarn(label, err) {
  if (typeof console !== 'undefined' && console.warn) {
    console.warn(
      '[tf] ' + label + ':',
      err && err.message ? err.message : err
    );
  }
}

/**
 * Run a tensor op sized like a PPO batch upload (batch × board matmul).
 * @param {{ rows?: number, cols?: number, trainBatchSize?: number }} [probeOptions]
 * @returns {Promise<boolean>}
 */
async function webGpuUploadProbePasses(probeOptions) {
  probeOptions = probeOptions || {};
  var rows = probeOptions.rows || 20;
  var cols = probeOptions.cols || 20;
  var batchSize = probeOptions.trainBatchSize || 512;
  var boardSize = rows * cols;
  try {
    await tf.ready();
    var result = tf.tidy(function () {
      var x = tf.randomUniform([batchSize, boardSize]);
      var w = tf.randomUniform([boardSize, Math.min(64, boardSize)]);
      return x.matMul(w);
    });
    await result.data();
    result.dispose();
    return true;
  } catch (e) {
    logBackendWarn('WebGPU upload probe failed', e);
    return false;
  }
}

async function trySetWebGpuBackend(inWorker, probeOptions) {
  var nav = typeof navigator !== 'undefined' ? navigator : {};
  if (!nav.gpu) return false;
  try {
    var adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return false;
    var okWg = await tf.setBackend('webgpu');
    if (!okWg) return false;
    await tf.ready();
    if (!(await webGpuUploadProbePasses(probeOptions))) {
      return false;
    }
    logBackendInfo('webgpu', inWorker);
    return true;
  } catch (e) {
    logBackendWarn('WebGPU backend failed', e);
    return false;
  }
}

async function trySetWebGlBackend() {
  try {
    var okGl = await tf.setBackend('webgl');
    if (!okGl) return false;
    await tf.ready();
    logBackendInfo('webgl', false);
    return true;
  } catch (e) {
    logBackendWarn('WebGL backend failed', e);
    return false;
  }
}

async function trySetCpuBackend(inWorker) {
  await tf.setBackend('cpu');
  await tf.ready();
  logBackendInfo('cpu', inWorker);
  return true;
}

/**
 * WebGPU → (WebGL on main thread only) → CPU.
 * `preference`: force order start; `auto` probes WebGPU with an upload smoke test first.
 * @param {'auto'|'webgpu'|'webgl'|'cpu'} [preference]
 * @param {{ rows?: number, cols?: number, trainBatchSize?: number }} [probeOptions]
 * @returns {Promise<string>}
 */
export async function ensureBestTfBackendOnce(preference, probeOptions) {
  if (backendChosen) {
    return tf.getBackend();
  }
  backendChosen = true;

  var pref = normalizePreference(preference);
  var inWorker = tfBootstrapInDedicatedWorker();
  var order = [];

  if (pref === 'webgpu') {
    order = ['webgpu', 'cpu'];
  } else if (pref === 'webgl') {
    order = inWorker ? ['cpu'] : ['webgl', 'cpu'];
  } else if (pref === 'cpu') {
    order = ['cpu'];
  } else {
    order = inWorker ? ['webgpu', 'cpu'] : ['webgpu', 'webgl', 'cpu'];
  }

  for (var i = 0; i < order.length; i++) {
    var kind = order[i];
    if (kind === 'webgpu') {
      if (await trySetWebGpuBackend(inWorker, probeOptions)) return 'webgpu';
      continue;
    }
    if (kind === 'webgl' && !inWorker) {
      if (await trySetWebGlBackend()) return 'webgl';
      continue;
    }
    if (kind === 'cpu') {
      await trySetCpuBackend(inWorker);
      return 'cpu';
    }
  }

  await trySetCpuBackend(inWorker);
  return 'cpu';
}

/**
 * When TF.js is on the WebGPU backend, returns the same GPUDevice used for inference/train
 * so custom WGSL (e.g. WebGPUGameEngine) can share it instead of calling requestDevice again.
 */
export function getTfWebGpuDeviceIfAvailable() {
  try {
    if (tf.getBackend() !== 'webgpu') return null;
    var b = tf.backend();
    if (!b || !b.device) return null;
    return b.device;
  } catch (e) {
    return null;
  }
}
