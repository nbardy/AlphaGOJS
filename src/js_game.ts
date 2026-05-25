// js_game.ts — CPU-side game logic for plague spread board game.
// Mirrors the WGSL kernel semantics exactly for cross-architecture eval.

export const K = 24;
export const N = K * K; // 576
export const WPB = Math.ceil(N / 16); // 36 — words per board (2-bit packed, 16 cells per u32)

// --- PCG hash (must match WGSL pcg exactly) ---
// Uses `>>> 0` everywhere to clamp to uint32.
export function pcg(s: number): number {
  let st = (Math.imul(s, 747796405) + 2891336453) >>> 0;
  const shift = ((st >>> 28) + 4) >>> 0;
  const w = Math.imul(((st >>> shift) ^ st) >>> 0, 277803737) >>> 0;
  return ((w >>> 22) ^ w) >>> 0;
}

export function rngFloat(seed: number, a: number, b: number): number {
  const hash = pcg((seed ^ pcg((a ^ pcg(b)) >>> 0)) >>> 0);
  return (hash & 0x00FFFFFF) / 16777216.0;
}

// --- Cell access (2-bit packed) ---

export function getCellState(packed: Uint32Array, cell: number): number {
  const wi = cell >>> 4;
  const bo = (cell & 15) << 1;
  return (packed[wi] >>> bo) & 3;
}

export function setCellState(packed: Uint32Array, cell: number, state: number): void {
  const wi = cell >>> 4;
  const bo = (cell & 15) << 1;
  packed[wi] = ((packed[wi] & ~((3 << bo) >>> 0)) | ((state & 3) << bo)) >>> 0;
}

// --- Territory counting ---

export function countTerritory(packed: Uint32Array): { p1: number; p2: number } {
  let p1 = 0;
  let p2 = 0;
  for (let i = 0; i < N; i++) {
    const s = getCellState(packed, i);
    if (s === 1) p1++;
    else if (s === 2) p2++;
  }
  return { p1, p2 };
}

// --- Terminal check ---

export function isTerminal(packed: Uint32Array): boolean {
  for (let i = 0; i < N; i++) {
    const s = getCellState(packed, i);
    if (s === 0) return false; // found an empty non-wall cell
  }
  return true;
}

// --- Plague spread ---
// Matches the WGSL rollout_step plague spread logic exactly.
//
// Key WGSL semantics:
//   - For each empty cell (not the just-placed action cell), look at 4 neighbors.
//   - Each occupied neighbor contributes ±rng_float to a sum (+1 if P1, -1 if P2).
//   - The result is clamp(trunc(sum * 2), -1, 1): if > 0 → P1, if < 0 → P2, else stays empty.
//   - The RNG base `rb` for each cell is `seed ^ step ^ cell`.
//   - Neighbor directions use constants 0..3 as the second arg to rng_float.
//   - Reads from old state, writes to new buffer (no read-after-write hazard).
//
// NOTE: `actionCell` is the cell just placed this turn — it is excluded from spread.
export function plagueSpread(
  packed: Uint32Array,
  seed: number,
  step: number,
  actionCell: number,
  batchIdx: number,
): void {
  // Read the full board into a float buffer (mirrors sh_b in WGSL).
  const buf = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    buf[i] = getCellState(packed, i);
  }

  // WGSL semantics: the spread loop reads neighbor states from sh_board (the board
  // state BEFORE the action was placed), not from sh_b. The action cell was empty
  // in the original board. Since the caller has already placed the action in `packed`,
  // we temporarily restore it to empty for neighbor reads, then set it back.

  const actionState = getCellState(packed, actionCell);
  setCellState(packed, actionCell, 0); // restore to empty for neighbor reads

  for (let cell = 0; cell < N; cell++) {
    const r = Math.floor(cell / K);
    const c = cell % K;
    const state = getCellState(packed, cell);

    if (state !== 0 || cell === actionCell) {
      // Not empty, or is the action cell — skip spread
      continue;
    }

    let sum = 0.0;
    const rb = (seed ^ step ^ cell) >>> 0;

    // Up neighbor
    if (r > 0) {
      const n = getCellState(packed, (r - 1) * K + c);
      if (n === 1) sum += rngFloat(rb, 0, batchIdx);
      else if (n === 2) sum -= rngFloat(rb, 0, batchIdx);
    }
    // Down neighbor
    if (r + 1 < K) {
      const n = getCellState(packed, (r + 1) * K + c);
      if (n === 1) sum += rngFloat(rb, 1, batchIdx);
      else if (n === 2) sum -= rngFloat(rb, 1, batchIdx);
    }
    // Left neighbor
    if (c > 0) {
      const n = getCellState(packed, r * K + (c - 1));
      if (n === 1) sum += rngFloat(rb, 2, batchIdx);
      else if (n === 2) sum -= rngFloat(rb, 2, batchIdx);
    }
    // Right neighbor
    if (c + 1 < K) {
      const n = getCellState(packed, r * K + (c + 1));
      if (n === 1) sum += rngFloat(rb, 3, batchIdx);
      else if (n === 2) sum -= rngFloat(rb, 3, batchIdx);
    }

    const v = Math.max(-1.0, Math.min(1.0, Math.trunc(sum * 2.0)));
    if (v > 0) buf[cell] = 1;
    else if (v < 0) buf[cell] = 2;
    // else stays 0
  }

  // Restore action cell state
  setCellState(packed, actionCell, actionState);
  // Also set it in buf so it gets repacked
  buf[actionCell] = actionState;

  // Repack buf back to packed (mirrors WGSL step 4)
  for (let wi = 0; wi < WPB; wi++) {
    let word = 0;
    for (let i = 0; i < 16; i++) {
      const cell = wi * 16 + i;
      if (cell < N) {
        word |= (buf[cell] << (i * 2));
      }
    }
    packed[wi] = word >>> 0;
  }
}

// --- Board initialization (matching init_boards kernel) ---

export function initBoard(seed: number, wallDensity: number): Uint32Array {
  const packed = new Uint32Array(WPB);

  const area = N;
  const baseChains = 5.0 + Math.trunc(rngFloat(seed, 100, 0) * 8.0);
  const numChains = Math.round((baseChains * area) / 100.0 * wallDensity);

  const DR = [0, 1, 0, -1];
  const DC = [1, 0, -1, 0];

  let rngState = (seed ^ 0) >>> 0; // b=0 for single board init

  for (let chain = 0; chain < numChains; chain++) {
    rngState = pcg(rngState);
    let r = Math.trunc(rngFloat(rngState, chain, 0) * K);
    let c = Math.trunc(rngFloat(rngState, chain, 1) * K);
    const length = 1 + Math.trunc(rngFloat(rngState, chain, 2) * 4.0);
    let dir = Math.trunc(rngFloat(rngState, chain, 3) * 4.0);

    for (let seg = 0; seg < length; seg++) {
      if (r < 0 || r >= K || c < 0 || c >= K) break;

      // Wall state = 3
      setCellState(packed, r * K + c, 3);

      r += DR[dir];
      c += DC[dir];

      if (rngFloat(rngState, chain, 4 + seg) < 0.3) {
        const turn = rngFloat(rngState, chain, 5 + seg) < 0.5 ? 1 : 3;
        dir = (dir + turn) % 4;
      }
    }
  }

  // Clear center 3x3 for fair start
  const mid = Math.trunc(K / 2);
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      const rr = mid + dr;
      const cc = mid + dc;
      if (rr >= 0 && rr < K && cc >= 0 && cc < K) {
        setCellState(packed, rr * K + cc, 0);
      }
    }
  }

  return packed;
}
