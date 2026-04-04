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

/**
 * Same spread semantics on **2-bit packed** layout (16 cells / u32, flat row-major index).
 * Matches `plague_spread_packed.wgsl` `spread_packed_pass` for parity benches.
 */
export function spreadPacked2BitWordsInOut(src, dst, rows, cols, numGames, tick) {
  var boardSize = rows * cols;
  var wpg = (boardSize + 15) >> 4;

  function readCode(wb, li) {
    if (li < 0 || li >= boardSize) {
      return WALL;
    }
    var w = (li / 16) | 0;
    var sh = (li % 16) * 2;
    return (src[wb + w] >>> sh) & 3;
  }

  for (var g = 0; g < numGames; g++) {
    var wb = g * wpg;
    for (var wi = 0; wi < wpg; wi++) {
      var oldWord = src[wb + wi];
      var combined = (oldWord | (oldWord >>> 1)) & 0x55555555;
      if (combined === 0x55555555) {
        dst[wb + wi] = oldWord;
        continue;
      }
      var newWord = 0;
      for (var i = 0; i < 16; i++) {
        var li = wi * 16 + i;
        if (li >= boardSize) {
          newWord |= oldWord & (3 << (i * 2));
          continue;
        }
        var code = (oldWord >>> (i * 2)) & 3;
        if (code !== EMPTY) {
          newWord |= code << (i * 2);
          continue;
        }
        var r = (li / cols) | 0;
        var c = li - r * cols;
        var sum = 0;

        if (r > 0) {
          var n0 = readCode(wb, li - cols);
          if (n0 === P1) sum += rand01(g, tick, li, 0);
          else if (n0 === P2) sum -= rand01(g, tick, li, 0);
        }
        if (r + 1 < rows) {
          var n1 = readCode(wb, li + cols);
          if (n1 === P1) sum += rand01(g, tick, li, 1);
          else if (n1 === P2) sum -= rand01(g, tick, li, 1);
        }
        if (c > 0) {
          var n2 = readCode(wb, li - 1);
          if (n2 === P1) sum += rand01(g, tick, li, 2);
          else if (n2 === P2) sum -= rand01(g, tick, li, 2);
        }
        if (c + 1 < cols) {
          var n3 = readCode(wb, li + 1);
          if (n3 === P1) sum += rand01(g, tick, li, 3);
          else if (n3 === P2) sum -= rand01(g, tick, li, 3);
        }

        var t = Math.trunc(sum * 2);
        var cl = t < -1 ? -1 : t > 1 ? 1 : t;
        var outC = cl > 0 ? P1 : cl < 0 ? P2 : EMPTY;
        newWord |= outC << (i * 2);
      }
      dst[wb + wi] = newWord;
    }
  }
}
