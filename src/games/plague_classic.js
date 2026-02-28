import { registerGame } from './registry';

// Cell values: 0=empty, 1=player1, -1=player2

function PlagueClassic(rows, cols) {
  this.rows = rows;
  this.cols = cols;
  this.size = rows * cols;
  this.board = new Int8Array(this.size);
}

PlagueClassic.prototype.getValidMoves = function () {
  var moves = [];
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) moves.push(i);
  }
  return moves;
};

PlagueClassic.prototype.getValidMovesMask = function () {
  var mask = new Float32Array(this.size);
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) mask[i] = 1;
  }
  return mask;
};

PlagueClassic.prototype.makeMove = function (player, index) {
  if (index < 0 || index >= this.size || this.board[index] !== 0) return false;
  this.board[index] = player;
  return true;
};

PlagueClassic.prototype.step = function () {
  var newBoard = Int8Array.from(this.board);
  var rows = this.rows, cols = this.cols;
  for (var r = 0; r < rows; r++) {
    for (var c = 0; c < cols; c++) {
      var i = r * cols + c;
      if (this.board[i] !== 0) continue;
      var sum = 0;
      if (r > 0) sum += this.board[(r - 1) * cols + c] * Math.random();
      if (r < rows - 1) sum += this.board[(r + 1) * cols + c] * Math.random();
      if (c > 0) sum += this.board[r * cols + (c - 1)] * Math.random();
      if (c < cols - 1) sum += this.board[r * cols + (c + 1)] * Math.random();
      newBoard[i] = Math.max(-1, Math.min(1, Math.trunc(sum * 2)));
    }
  }
  this.board = newBoard;
};

PlagueClassic.prototype.isGameOver = function () {
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) return false;
  }
  return true;
};

PlagueClassic.prototype.countCells = function () {
  var p1 = 0, p2 = 0, empty = 0;
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 1) p1++;
    else if (this.board[i] === -1) p2++;
    else empty++;
  }
  return { p1: p1, p2: p2, empty: empty };
};

PlagueClassic.prototype.getWinner = function () {
  var c = this.countCells();
  if (c.p1 > c.p2) return 1;
  if (c.p2 > c.p1) return -1;
  return 0;
};

PlagueClassic.prototype.getBoardForNN = function (player) {
  var state = new Float32Array(this.size);
  for (var i = 0; i < this.size; i++) {
    state[i] = this.board[i] * player;
  }
  return state;
};

PlagueClassic.prototype.getBoardChannels = function () { return 1; };

// Renderer
var classicRenderer = {
  label: 'Plague Classic',
  cellColor: function (val) {
    if (val === 1) return [0, 255, 136];
    if (val === -1) return [255, 51, 102];
    return [30, 30, 58];
  },
  humanCellColor: function (val) {
    if (val === 1) return '#00ff88';
    if (val === -1) return '#ff3366';
    return '#1e1e3a';
  }
};

registerGame('plague_classic', function (rows, cols) {
  return new PlagueClassic(rows, cols);
}, classicRenderer);
