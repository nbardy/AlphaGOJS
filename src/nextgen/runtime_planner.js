// Chooses the best available runtime tier based on probed capabilities.

var TIER_A = 'A'; // Worker WebGPU + OffscreenCanvas WebGPU
var TIER_B = 'B'; // Main-thread WebGPU + worker staging
var TIER_C = 'C'; // Worker OffscreenCanvas WebGL2 fallback
var TIER_D = 'D'; // CPU/WASM fallback

function supportsTier(tier, cap) {
  if (tier === TIER_A) {
    return !!(cap.workerWebGPU && cap.offscreenCanvas && cap.offscreenWebGPU);
  }
  if (tier === TIER_B) {
    return !!cap.webgpu;
  }
  if (tier === TIER_C) {
    return !!(cap.offscreenCanvas && cap.offscreenWebGL2);
  }
  if (tier === TIER_D) {
    return true;
  }
  return false;
}

function defaultsForTier(tier, cap) {
  if (tier === TIER_A) {
    return {
      tier: tier,
      useGpuOwnerWorker: true,
      useSimWorker: !!cap.sab,
      useSharedInferenceWorker: false,
      useSAB: !!cap.sab,
      renderBackend: 'webgpu'
    };
  }
  if (tier === TIER_B) {
    return {
      tier: tier,
      useGpuOwnerWorker: false,
      useSimWorker: !!cap.sab,
      useSharedInferenceWorker: false,
      useSAB: !!cap.sab,
      renderBackend: 'webgpu'
    };
  }
  if (tier === TIER_C) {
    return {
      tier: tier,
      useGpuOwnerWorker: true,
      useSimWorker: !!cap.sab,
      useSharedInferenceWorker: false,
      useSAB: !!cap.sab,
      renderBackend: 'webgl2'
    };
  }
  return {
    tier: TIER_D,
    useGpuOwnerWorker: false,
    useSimWorker: false,
    useSharedInferenceWorker: false,
    useSAB: false,
    renderBackend: 'cpu'
  };
}

export function chooseRuntimeTier(cap, prefs) {
  prefs = prefs || {};
  var preferTier = prefs.tier || null;

  if (preferTier && supportsTier(preferTier, cap)) {
    var preferred = defaultsForTier(preferTier, cap);
    return Object.assign(preferred, prefs);
  }

  var tier = TIER_D;
  if (supportsTier(TIER_A, cap)) tier = TIER_A;
  else if (supportsTier(TIER_B, cap)) tier = TIER_B;
  else if (supportsTier(TIER_C, cap)) tier = TIER_C;

  var plan = defaultsForTier(tier, cap);
  return Object.assign(plan, prefs);
}

export var RuntimeTiers = {
  A: TIER_A,
  B: TIER_B,
  C: TIER_C,
  D: TIER_D
};

