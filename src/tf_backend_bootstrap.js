/**
 * Prefer TensorFlow.js WebGPU when the browser exposes navigator.gpu, then WebGL, then CPU.
 * The default @tensorflow/tfjs bundle registers WebGL but does not load the WebGPU backend;
 * importing @tensorflow/tfjs-backend-webgpu here registers it so setBackend('webgpu') can succeed.
 *
 * This is separate from WebGPUGameEngine / ?webgpuEnv=1 (custom WGSL plague sim).
 */
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

var backendChosen = false;

export async function ensureBestTfBackendOnce() {
  if (backendChosen) {
    return tf.getBackend();
  }
  backendChosen = true;

  var nav = typeof navigator !== 'undefined' ? navigator : {};
  try {
    if (nav.gpu) {
      var okWg = await tf.setBackend('webgpu');
      if (okWg) {
        await tf.ready();
        if (typeof console !== 'undefined' && console.info) {
          console.info('[tf] backend: webgpu');
        }
        return 'webgpu';
      }
    }
  } catch (e) {
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('[tf] WebGPU backend failed:', e && e.message ? e.message : e);
    }
  }

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

  await tf.setBackend('cpu');
  await tf.ready();
  if (typeof console !== 'undefined' && console.info) {
    console.info('[tf] backend: cpu');
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
