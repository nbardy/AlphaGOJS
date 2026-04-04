// Plague Walls: plague territory game WITH random wall chains.
// Cell values: 0=empty, 1=P1, -1=P2, 2=wall.
// Walls block plague spread; encoded as 0.5 in getBoardForNN.
// Self-registers as 'plague_walls' on import.

import { registerGame } from './registry';
import { generatePlagueWallsInto, PLAGUE_WALL_CELL } from '../engine/plague_walls_layout';

var WALL = PLAGUE_WALL_CELL;

function PlagueWalls(rows, cols) {
  this.rows = rows || 10;
  this.cols = cols || 10;
  this.size = this.rows * this.cols;
  this.board = new Int8Array(this.size);
  this._generateWalls();
}

PlagueWalls.prototype._generateWalls = function () {
  generatePlagueWallsInto(this.board, this.rows, this.cols, Math.random);
};

PlagueWalls.prototype.reset = function () {
  this.board.fill(0);
  this._generateWalls();
};

PlagueWalls.prototype.getValidMoves = function () {
  var moves = [];
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) moves.push(i);
  }
  return moves;
};

PlagueWalls.prototype.getValidMovesMask = function () {
  var mask = new Float32Array(this.size);
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) mask[i] = 1;
  }
  return mask;
};

PlagueWalls.prototype.makeMove = function (player, index) {
  if (index < 0 || index >= this.size || this.board[index] !== 0) return false;
  this.board[index] = player;
  return true;
};

PlagueWalls.prototype.spreadPlague = function () {
  var newBoard = Int8Array.from(this.board);
  var rows = this.rows, cols = this.cols, board = this.board;
  for (var r = 0; r < rows; r++) {
    for (var c = 0; c < cols; c++) {
      var i = r * cols + c;
      if (board[i] !== 0) continue;
      var sum = 0;
      var n;
      // Walls block plague spread — only sum neighbors that aren't walls
      if (r > 0) { n = board[(r - 1) * cols + c]; if (n !== WALL) sum += n * Math.random(); }
      if (r < rows - 1) { n = board[(r + 1) * cols + c]; if (n !== WALL) sum += n * Math.random(); }
      if (c > 0) { n = board[r * cols + (c - 1)]; if (n !== WALL) sum += n * Math.random(); }
      if (c < cols - 1) { n = board[r * cols + (c + 1)]; if (n !== WALL) sum += n * Math.random(); }
      newBoard[i] = Math.max(-1, Math.min(1, Math.trunc(sum * 2)));
    }
  }
  this.board = newBoard;
};

PlagueWalls.prototype.isGameOver = function () {
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) return false;
  }
  return true;
};

PlagueWalls.prototype.countCells = function () {
  var p1 = 0, p2 = 0, empty = 0;
  for (var i = 0; i < this.size; i++) {
    var v = this.board[i];
    if (v === 1) p1++;
    else if (v === -1) p2++;
    else if (v === 0) empty++;
    // walls (v === WALL) don't count for either side
  }
  return { p1: p1, p2: p2, empty: empty };
};

PlagueWalls.prototype.getWinner = function () {
  var c = this.countCells();
  if (c.p1 > c.p2) return 1;
  if (c.p2 > c.p1) return -1;
  return 0;
};

PlagueWalls.prototype.getBoardForNN = function (player) {
  // Single-channel encoding: own=+1, opponent=-1, empty=0, wall=0.5.
  // Wall value is perspective-independent so the model learns spatial terrain.
  var state = new Float32Array(this.size);
  for (var i = 0; i < this.size; i++) {
    var v = this.board[i];
    if (v === WALL) state[i] = 0.5;
    else state[i] = v * player;
  }
  return state;
};

// Renderer: colors for plague with walls
var renderer = {
  label: 'Walls',
  cellColor: function (val) {
    if (val === 1) return [0, 255, 136];
    if (val === -1) return [255, 51, 102];
    if (val === 2) return [80, 70, 90];   // wall: dark purple
    return [30, 30, 58];
  },
  humanCellColor: function (val) {
    if (val === 1) return '#00ff88';
    if (val === -1) return '#ff3366';
    if (val === 2) return '#504660';       // wall
    return '#1e1e3a';
  }
};

registerGame('plague_walls', function (rows, cols) {
  return new PlagueWalls(rows, cols);
}, renderer);
