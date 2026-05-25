// js_inference.ts — CPU forward pass mirroring the WGSL forward_pass function.
// Used for cross-architecture tournament eval (infrequent, correctness > speed).

import { type WeightLayout, computeWeightLayout } from './arch_config';
import { K, N, getCellState } from './js_game';
import { pcg } from './js_game';

const MASK_FILL = -3.4e38;
const EPS = 1.0e-12;

export interface ArchWeights {
  dense: Float32Array;  // length = layout.denseWeightCount
  embed: Float16Array;  // length = layout.embedWeightCount
}

// --- Weight accessors (mirror WGSL exactly) ---

function cellEmbed(embed: Float16Array, D: number, state: number, d: number): number {
  // E_CELL = 0, so embed[state * D + d]
  return embed[state * D + d];
}

function patchEmbed(embed: Float16Array, D: number, row: number, d: number): number {
  const E_PATCH = 4 * D;
  return embed[E_PATCH + row * D + d];
}

function c1w(dense: Float32Array, D: number, ky: number, kx: number, c: number, o: number): number {
  // W_CONV1 = 0
  return dense[((ky * 3 + kx) * D + c) * D + o];
}

function c2w(dense: Float32Array, D: number, PATCH_CH: number, ky: number, kx: number, c: number, o: number): number {
  const W_CONV2 = 9 * D * D;
  return dense[W_CONV2 + ((ky * 3 + kx) * D + c) * PATCH_CH + o];
}

function fw(dense: Float32Array, D: number, PATCH_CH: number, c: number, o: number): number {
  const W_FUSE = 9 * D * D + 9 * D * PATCH_CH;
  return dense[W_FUSE + c * D + o];
}

function pw(dense: Float32Array, D: number, PATCH_CH: number, d: number): number {
  const W_FUSE = 9 * D * D + 9 * D * PATCH_CH;
  const W_POLICY = W_FUSE + 2 * D * D;
  return dense[W_POLICY + d];
}

function vw(dense: Float32Array, D: number, PATCH_CH: number, d: number): number {
  const W_FUSE = 9 * D * D + 9 * D * PATCH_CH;
  const W_POLICY = W_FUSE + 2 * D * D;
  const W_VALUE = W_POLICY + D;
  return dense[W_VALUE + d];
}

function vbias(dense: Float32Array, D: number, PATCH_CH: number): number {
  const W_FUSE = 9 * D * D + 9 * D * PATCH_CH;
  const W_POLICY = W_FUSE + 2 * D * D;
  const W_VALUE = W_POLICY + D;
  return dense[W_VALUE + D];
}

// --- Board cell reader with perspective inversion ---

function getCell(board: Uint32Array, cell: number, invert: boolean): number {
  const s = getCellState(board, cell);
  if (invert) {
    if (s === 1) return 2;
    if (s === 2) return 1;
  }
  return s;
}

// --- Conv2 at a patch position (with dilation=2) ---
// Mirrors conv2_at_patch in WGSL. The sh_a array stores P*P*D values (conv1 output).
// Conv2 uses dilation=2 over the 8x8 patch grid.

function conv2AtPatch(
  shA: Float32Array, dense: Float32Array,
  D: number, P: number, PATCH_CH: number,
  py: number, px: number, o: number,
): number {
  let acc = 0.0;
  for (let c = 0; c < D; c++) {
    for (let ky = 0; ky < 3; ky++) {
      for (let kx = 0; kx < 3; kx++) {
        const y = py + 2 * (ky - 1);
        const x = px + 2 * (kx - 1);
        if (y >= 0 && y < P && x >= 0 && x < P) {
          acc += c2w(dense, D, PATCH_CH, ky, kx, c, o) * shA[y * P * D + x * D + c];
        }
      }
    }
  }
  return Math.max(acc, 0.0); // ReLU
}

// --- Forward pass + action sampling ---

export function inferAction(
  weights: ArchWeights,
  layout: WeightLayout,
  board: Uint32Array,
  player: number,
  seed: number,
): number {
  const { D, P, PATCH_CH } = layout;
  const { dense, embed } = weights;
  const invert = player === 2;

  // Step 1: Patch embedding — 8x8 grid of 3x3 patches
  const shA = new Float32Array(P * P * D);
  for (let lid = 0; lid < P * P; lid++) {
    const py = Math.floor(lid / P);
    const px = lid % P;
    let pi = 0;
    for (let dy = 0; dy < 3; dy++) {
      for (let dx = 0; dx < 3; dx++) {
        pi |= getCell(board, (py * 3 + dy) * K + (px * 3 + dx), invert) << (2 * (dy * 3 + dx));
      }
    }
    pi = pi >>> 0;
    for (let d = 0; d < D; d++) {
      shA[lid * D + d] = patchEmbed(embed, D, pi, d);
    }
  }

  // Step 2: Conv1 — 3x3 over 8x8 patch grid, D→D, ReLU
  const conv1Out = new Float32Array(P * P * D);
  for (let lid = 0; lid < P * P; lid++) {
    const py = Math.floor(lid / P);
    const px = lid % P;
    for (let o = 0; o < D; o++) {
      let acc = 0.0;
      for (let c = 0; c < D; c++) {
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const y = py + ky - 1;
            const x = px + kx - 1;
            if (y >= 0 && y < P && x >= 0 && x < P) {
              acc += c1w(dense, D, ky, kx, c, o) * shA[y * P * D + x * D + c];
            }
          }
        }
      }
      conv1Out[lid * D + o] = Math.max(acc, 0.0); // ReLU
    }
  }
  // Conv1 output replaces shA for subsequent conv2 reads
  shA.set(conv1Out);

  // Step 3: Conv2 + pixel shuffle + fuse + heads (per-cell)
  const logits = new Float32Array(N);
  const pool = new Float32Array(D); // global average pool accumulator

  for (let cell = 0; cell < N; cell++) {
    const y = Math.floor(cell / K);
    const x = cell % K;
    const py = Math.floor(y / 3);
    const px = Math.floor(x / 3);
    const sub = (y % 3) * 3 + (x % 3);
    const state = getCell(board, cell, invert);

    // Conv2 + pixel shuffle: for this cell's sub-position within its patch
    const decoded = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      decoded[d] = conv2AtPatch(shA, dense, D, P, PATCH_CH, py, px, sub * D + d);
    }

    // Fuse: linear(decoded, cell_embed) → ReLU
    const fused = new Float32Array(D);
    for (let o = 0; o < D; o++) {
      let acc = 0.0;
      for (let c = 0; c < D; c++) {
        acc += fw(dense, D, PATCH_CH, c, o) * decoded[c]
             + fw(dense, D, PATCH_CH, D + c, o) * cellEmbed(embed, D, state, c);
      }
      fused[o] = Math.max(acc, 0.0); // ReLU
      pool[o] += fused[o];
    }

    // Policy head: dot(policy_w, fused)
    let logit = 0.0;
    for (let d = 0; d < D; d++) {
      logit += pw(dense, D, PATCH_CH, d) * fused[d];
    }
    // Mask non-empty cells
    logits[cell] = state === 0 ? logit : MASK_FILL;
  }

  // Value head: global average pool → dot(value_w, pool/N) + bias → tanh
  // (We compute this for completeness but only use the action from policy)
  // const valuePool = pool.map(v => v / N);
  // let val = vbias(dense, D, PATCH_CH);
  // for (let d = 0; d < D; d++) val += vw(dense, D, PATCH_CH, d) * valuePool[d];
  // const value = Math.tanh(val);

  // Softmax over valid cells (numerically stable: subtract max, exp, normalize)
  let maxLogit = MASK_FILL;
  for (let i = 0; i < N; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  const probs = new Float32Array(N);
  let sumExp = 0.0;
  for (let i = 0; i < N; i++) {
    if (getCellState(board, i) === 0) { // only valid (empty) cells, reading raw state
      // For invert: state==0 is the same either way (empty is invariant to inversion)
      const p = Math.exp(logits[i] - maxLogit);
      probs[i] = p;
      sumExp += p;
    }
  }

  // Normalize
  if (sumExp > EPS) {
    for (let i = 0; i < N; i++) {
      probs[i] /= sumExp;
    }
  }

  // Sample from CDF using PCG RNG
  const rand = rngFloat(seed);

  let cumsum = 0.0;
  let chosen = 0;
  let chosenP = 0.0;
  for (let i = 0; i < N; i++) {
    if (probs[i] <= 0.0) continue;
    cumsum += probs[i];
    if (rand < cumsum) {
      chosen = i;
      chosenP = probs[i];
      break;
    }
  }
  // Fallback: if floating point issues prevented selection, pick first valid
  if (chosenP === 0.0) {
    for (let i = 0; i < N; i++) {
      if (probs[i] > 0.0) {
        chosen = i;
        break;
      }
    }
  }

  return chosen;
}

// Simple RNG float from a single seed (for sampling).
// Matches the WGSL rng_float(seed, a, b) pattern but simplified for single-value use.
function rngFloat(seed: number): number {
  const hash = pcg(seed >>> 0);
  return (hash & 0x00FFFFFF) / 16777216.0;
}

// --- Value inference (useful for analysis, not needed for action selection) ---

export function inferValue(
  weights: ArchWeights,
  layout: WeightLayout,
  board: Uint32Array,
  player: number,
): number {
  const { D, P, PATCH_CH } = layout;
  const { dense, embed } = weights;
  const invert = player === 2;

  // Patch embedding
  const shA = new Float32Array(P * P * D);
  for (let lid = 0; lid < P * P; lid++) {
    const py = Math.floor(lid / P);
    const px = lid % P;
    let pi = 0;
    for (let dy = 0; dy < 3; dy++) {
      for (let dx = 0; dx < 3; dx++) {
        pi |= getCell(board, (py * 3 + dy) * K + (px * 3 + dx), invert) << (2 * (dy * 3 + dx));
      }
    }
    pi = pi >>> 0;
    for (let d = 0; d < D; d++) {
      shA[lid * D + d] = patchEmbed(embed, D, pi, d);
    }
  }

  // Conv1
  const conv1Out = new Float32Array(P * P * D);
  for (let lid = 0; lid < P * P; lid++) {
    const py = Math.floor(lid / P);
    const px = lid % P;
    for (let o = 0; o < D; o++) {
      let acc = 0.0;
      for (let c = 0; c < D; c++) {
        for (let ky = 0; ky < 3; ky++) {
          for (let kx = 0; kx < 3; kx++) {
            const y = py + ky - 1;
            const x = px + kx - 1;
            if (y >= 0 && y < P && x >= 0 && x < P) {
              acc += c1w(dense, D, ky, kx, c, o) * shA[y * P * D + x * D + c];
            }
          }
        }
      }
      conv1Out[lid * D + o] = Math.max(acc, 0.0);
    }
  }
  shA.set(conv1Out);

  // Conv2 + fuse → accumulate pool
  const pool = new Float32Array(D);
  for (let cell = 0; cell < N; cell++) {
    const y = Math.floor(cell / K);
    const x = cell % K;
    const py = Math.floor(y / 3);
    const px = Math.floor(x / 3);
    const sub = (y % 3) * 3 + (x % 3);
    const state = getCell(board, cell, invert);

    const decoded = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      decoded[d] = conv2AtPatch(shA, dense, D, P, PATCH_CH, py, px, sub * D + d);
    }

    for (let o = 0; o < D; o++) {
      let acc = 0.0;
      for (let c = 0; c < D; c++) {
        acc += fw(dense, D, PATCH_CH, c, o) * decoded[c]
             + fw(dense, D, PATCH_CH, D + c, o) * cellEmbed(embed, D, state, c);
      }
      pool[o] += Math.max(acc, 0.0);
    }
  }

  // Value head: global avg pool → linear → tanh
  let val = vbias(dense, D, PATCH_CH);
  for (let d = 0; d < D; d++) {
    val += vw(dense, D, PATCH_CH, d) * (pool[d] / N);
  }
  return Math.tanh(val);
}
