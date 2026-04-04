/**
 * Shared WebGPU bootstrap for Node benchmarks (Dawn `webgpu` package or native globals).
 */

export async function ensureWebGPUGlobals() {
  if (globalThis?.navigator?.gpu || globalThis?.gpu) return 'native';
  let mod = null;
  try {
    mod = await import('webgpu');
  } catch (e) {
    return null;
  }
  if (!mod || typeof mod.create !== 'function') return null;
  if (mod.globals && typeof mod.globals === 'object') Object.assign(globalThis, mod.globals);
  if (!globalThis.navigator) globalThis.navigator = {};
  if (!globalThis.navigator.gpu) globalThis.navigator.gpu = mod.create([]);
  return 'node-webgpu';
}

export function getGPU() {
  if (globalThis?.navigator?.gpu) return globalThis.navigator.gpu;
  if (globalThis?.gpu) return globalThis.gpu;
  return null;
}

export async function initWebGPUDevice() {
  const provider = await ensureWebGPUGlobals();
  const gpu = getGPU();
  if (!gpu) {
    return { ok: false, provider: provider || 'none', adapter: null, device: null };
  }
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    return { ok: false, provider: provider || 'unknown', adapter: null, device: null };
  }
  const device = await adapter.requestDevice();
  return { ok: true, provider: provider || 'unknown', adapter, device };
}
