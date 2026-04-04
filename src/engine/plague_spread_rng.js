/**
 * Deterministic spread RNG — bit-for-bit companion to src/engine/wgsl/plague_spread.wgsl
 * (hash_u32 + rand01). Use in tests / CPU reference for WebGPU parity.
 */

export function hashU32(x) {
  var v = x >>> 0;
  v = (v ^ (v >>> 16)) >>> 0;
  v = Math.imul(v, 0x7feb352d) >>> 0;
  v = (v ^ (v >>> 15)) >>> 0;
  v = Math.imul(v, 0x846ca68b) >>> 0;
  v = (v ^ (v >>> 16)) >>> 0;
  return v >>> 0;
}

export function rand01(gameId, tick, localIdx, dir) {
  var mix = (gameId * 0x9e3779b9) ^ (tick * 0x85ebca6b) ^ (localIdx * 0xc2b2ae35) ^ (dir * 0x27d4eb2d);
  var h = hashU32(mix >>> 0);
  return (h & 0xffffff) / 0x1000000;
}
