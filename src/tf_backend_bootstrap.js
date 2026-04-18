/**
 * Prefer TensorFlow.js WebGPU when the browser exposes navigator.gpu, then WebGL, then CPU.
 * @tensorflow/tfjs-backend-webgpu is patched (patch-package): uploadToGPU uses queue.writeBuffer;
 * BufferManager never uses mappedAtCreation (WebKit/Safari rejects STORAGE+mappable create).
 * The default @tensorflow/tfjs bundle registers WebGL but does not load the WebGPU backend;
 * importing @tensorflow/tfjs-backend-webgpu here registers it so setBackend('webgpu') can succeed.
 *
 * In a dedicated Worker, skips WebGL (no canvas). Uses one export so webpack worker chunks
 * cannot drop a separate "worker-only" symbol during split-chunk / HMR.
 */
import './webgpu_queue_write_shim.js';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

var backendChosen = false;

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

/**
 * WebGPU → (WebGL on main thread only) → CPU.
 * Probes requestAdapter() before setBackend('webgpu'): TF's factory uses adapter.features
 * and throws if adapter is null.
 */
export async function ensureBestTfBackendOnce() {
  if (backendChosen) {
    return tf.getBackend();
  }
  backendChosen = true;

  var inWorker = tfBootstrapInDedicatedWorker();
  var nav = typeof navigator !== 'undefined' ? navigator : {};
  try {
    if (nav.gpu) {
      var adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        var okWg = await tf.setBackend('webgpu');
        if (okWg) {
          await tf.ready();
          if (typeof console !== 'undefined' && console.info) {
            console.info('[tf] backend: webgpu' + (inWorker ? ' (worker)' : ''));
          }
          return 'webgpu';
        }
      }
    }
  } catch (e) {
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('[tf] WebGPU backend failed:', e && e.message ? e.message : e);
    }
  }

  if (!inWorker) {
    try {
      var okGl = await tf.setBackend('webgl');
      if (okGl) {
        await tf.ready();
        if (typeof console !== 'undefined' && console.info) {
          console.info('[tf] backend: webgl');
        }
        return 'webgl';
      }
    } catch (e2) {
      if (typeof console !== 'undefined' && console.warn) {
        console.warn('[tf] WebGL backend failed:', e2 && e2.message ? e2.message : e2);
      }
    }
  }

  await tf.setBackend('cpu');
  await tf.ready();
  if (typeof console !== 'undefined' && console.info) {
    console.info(
      '[tf] backend: cpu' +
        (inWorker ? ' (worker — WebGL unavailable in workers)' : '')
    );
  }
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
