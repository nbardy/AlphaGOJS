/**
 * CPU reference spread with the same neighbor order and RNG as plague_env.wgsl `spread_pass`
 * (via plague_spread_rng.rand01). For parity tests against WebGPU.
 */

import { rand01 } from './plague_spread_rng.js';

var EMPTY = 0;
var P1 = 1;
var P2 = 2;
var WALL = 3;

/**
 * Read src, write dst (same semantics as spreadPackedAllGames). No allocation.
 */
export function spreadPackedAllGamesInOut(src, dst, rows, cols, numGames, tick) {
  var boardSize = rows * cols;
  for (var g = 0; g < numGames; g++) {
    var base = g * boardSize;
    for (var li = 0; li < boardSize; li++) {
      var idx = base + li;
      var cur = src[idx];
      if (cur !== EMPTY) {
        dst[idx] = cur;
        continue;
      }
      var r = (li / cols) | 0;
      var c = li - r * cols;
      var sum = 0;

      var n0 = r > 0 ? src[base + (r - 1) * cols + c] : WALL;
      if (n0 === P1) sum += rand01(g, tick, li, 0);
      else if (n0 === P2) sum -= rand01(g, tick, li, 0);

      var n1 = r + 1 < rows ? src[base + (r + 1) * cols + c] : WALL;
      if (n1 === P1) sum += rand01(g, tick, li, 1);
      else if (n1 === P2) sum -= rand01(g, tick, li, 1);

      var n2 = c > 0 ? src[base + r * cols + (c - 1)] : WALL;
      if (n2 === P1) sum += rand01(g, tick, li, 2);
      else if (n2 === P2) sum -= rand01(g, tick, li, 2);

      var n3 = c + 1 < cols ? src[base + r * cols + (c + 1)] : WALL;
      if (n3 === P1) sum += rand01(g, tick, li, 3);
      else if (n3 === P2) sum -= rand01(g, tick, li, 3);

      var t = Math.trunc(sum * 2);
      var cl = t < -1 ? -1 : t > 1 ? 1 : t;
      dst[idx] = cl > 0 ? P1 : cl < 0 ? P2 : EMPTY;
    }
  }
}

export function spreadPackedAllGames(packed, rows, cols, numGames, tick) {
  var out = new Uint32Array(packed.length);
  spreadPackedAllGamesInOut(packed, out, rows, cols, numGames, tick);
  return out;
}
