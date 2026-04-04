// Perspective NN cell codes for discrete-input models (matches float getBoardForNN semantics).
// 0 empty, 1 own, 2 opponent, 3 wall (wall is not flipped by player).

import { PLAGUE_WALL_CELL } from './engine/plague_walls_layout';
import { CELL_EMPTY, CELL_P1, CELL_P2, CELL_WALL } from './engine/webgpu_plague_spread_engine';

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
