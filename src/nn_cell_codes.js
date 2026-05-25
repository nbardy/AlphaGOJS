// Perspective NN cell codes for discrete-input models (matches float getBoardForNN semantics).
// 0 empty, 1 own, 2 opponent, 3 wall (wall is not flipped by player).

import { PLAGUE_WALL_CELL } from './engine/plague_walls_layout.js';
import { CELL_EMPTY, CELL_P1, CELL_P2, CELL_WALL } from './engine/webgpu_plague_spread_engine.js';

/**
 * @param {number} u - packed WGSL cell (CELL_*)
 * @param {number} player - +1 or -1
 * @returns {number} 0..3
 */
export function packedUintToNnCode(u, player) {
  if (u === CELL_WALL) return 3;
  if (u === CELL_EMPTY) return 0;
  var v = u === CELL_P1 ? 1 : -1;
  if (v * player === 1) return 1;
  return 2;
}

/**
 * @param {number} v - GPU tensor cell: 0 empty, ±1 players, PLAGUE_WALL_CELL wall
 * @param {number} player
 * @returns {number} 0..3
 */
export function floatEngineCellToNnCode(v, player) {
  if (v === PLAGUE_WALL_CELL) return 3;
  if (v === 0) return 0;
  if (v * player === 1) return 1;
  return 2;
}

/**
 * @param {Int8Array|number[]} boardRow - classic / walls int8 cell
 * @param {number} player
 * @param {number} i - flat index
 * @returns {number} 0..3
 */
export function int8BoardCellToNnCode(boardRow, player, i) {
  var v = boardRow[i];
  if (v === PLAGUE_WALL_CELL) return 3;
  if (v === 0) return 0;
  if (v * player === 1) return 1;
  return 2;
}

var _WALL_EPS = 1e-3;

/**
 * Inverse of plague getBoardForNN: float row (empty=0, own=+1, opp=-1, wall=0.5) → NN codes 0..3.
 * @param {ArrayLike<number>} floatRow
 * @param {number} boardSize
 * @returns {Int32Array}
 */
export function nnPerspectiveFloatBoardToCodes(floatRow, boardSize) {
  var out = new Int32Array(boardSize);
  for (var i = 0; i < boardSize; i++) {
    var v = floatRow[i];
    if (Math.abs(v - 0.5) < _WALL_EPS) out[i] = 3;
    else if (Math.abs(v) < _WALL_EPS) out[i] = 0;
    else if (v > 0) out[i] = 1;
    else out[i] = 2;
  }
  return out;
}

/**
 * Same layout as plague getBoardForNN (perspective already baked into codes).
 * @param {ArrayLike<number>} codeRow
 * @param {number} boardSize
 * @returns {Float32Array}
 */
export function nnCodesToFloatBoard(codeRow, boardSize) {
  var out = new Float32Array(boardSize);
  for (var i = 0; i < boardSize; i++) {
    var c = codeRow[i];
    if (c === 3) out[i] = 0.5;
    else if (c === 0) out[i] = 0;
    else if (c === 1) out[i] = 1;
    else out[i] = -1;
  }
  return out;
}
