import { registerGame } from './registry';

// Cell values: 0=empty, 1=player1, -1=player2, 2=wall (blocker)
var WALL = 2;

function PlagueAdvanced(rows, cols) {
  this.rows = rows;
  this.cols = cols;
  this.size = rows * cols;
  this.board = new Int8Array(this.size);
  this._generateWalls();
}

// Generate random wall chains: 5-12 chains of 1-4 segments with possible turns
PlagueAdvanced.prototype._generateWalls = function () {
  var rows = this.rows, cols = this.cols;
  var numChains = 5 + Math.floor(Math.random() * 8);

  for (var chain = 0; chain < numChains; chain++) {
    var r = Math.floor(Math.random() * rows);
    var c = Math.floor(Math.random() * cols);
    var len = 1 + Math.floor(Math.random() * 4);

    // Pick an initial direction: 0=right, 1=down, 2=left, 3=up
    var dir = Math.floor(Math.random() * 4);
    var DR = [0, 1, 0, -1];
    var DC = [1, 0, -1, 0];

    for (var seg = 0; seg < len; seg++) {
      if (r < 0 || r >= rows || c < 0 || c >= cols) break;
      this.board[r * cols + c] = WALL;
      r += DR[dir];
      c += DC[dir];

      // Possible turn after each segment (30% chance)
      if (Math.random() < 0.3) {
        // Turn 90 degrees (clockwise or counterclockwise)
        dir = (dir + (Math.random() < 0.5 ? 1 : 3)) % 4;
      }
    }
  }

  // Make sure center area is clear (don't block the whole middle)
  var midR = Math.floor(rows / 2), midC = Math.floor(cols / 2);
  for (var dr = -1; dr <= 1; dr++) {
    for (var dc = -1; dc <= 1; dc++) {
      var idx = (midR + dr) * cols + (midC + dc);
      if (idx >= 0 && idx < this.size) this.board[idx] = 0;
    }
  }
};

PlagueAdvanced.prototype.getValidMoves = function () {
  var moves = [];
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) moves.push(i);
  }
  return moves;
};

PlagueAdvanced.prototype.getValidMovesMask = function () {
  var mask = new Float32Array(this.size);
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) mask[i] = 1;
  }
  return mask;
};

PlagueAdvanced.prototype.makeMove = function (player, index) {
  if (index < 0 || index >= this.size || this.board[index] !== 0) return false;
  this.board[index] = player;
  return true;
};

PlagueAdvanced.prototype.step = function () {
  var newBoard = Int8Array.from(this.board);
  var rows = this.rows, cols = this.cols, board = this.board;
  for (var r = 0; r < rows; r++) {
    for (var c = 0; c < cols; c++) {
      var i = r * cols + c;
      if (board[i] !== 0) continue; // skip walls, occupied
      var sum = 0;
      var n;
      // Walls block spread — only consider neighbors that aren't walls
      if (r > 0) { n = board[(r - 1) * cols + c]; if (n !== WALL) sum += n * Math.random(); }
      if (r < rows - 1) { n = board[(r + 1) * cols + c]; if (n !== WALL) sum += n * Math.random(); }
      if (c > 0) { n = board[r * cols + (c - 1)]; if (n !== WALL) sum += n * Math.random(); }
      if (c < cols - 1) { n = board[r * cols + (c + 1)]; if (n !== WALL) sum += n * Math.random(); }
      newBoard[i] = Math.max(-1, Math.min(1, Math.trunc(sum * 2)));
    }
  }
  this.board = newBoard;
};

PlagueAdvanced.prototype.isGameOver = function () {
  for (var i = 0; i < this.size; i++) {
    if (this.board[i] === 0) return false;
  }
  return true;
};

PlagueAdvanced.prototype.countCells = function () {
  var p1 = 0, p2 = 0, empty = 0;
  for (var i = 0; i < this.size; i++) {
    var v = this.board[i];
    if (v === 1) p1++;
    else if (v === -1) p2++;
    else if (v === 0) empty++;
    // walls don't count for either side
  }
  return { p1: p1, p2: p2, empty: empty };
};

PlagueAdvanced.prototype.getWinner = function () {
  var c = this.countCells();
  if (c.p1 > c.p2) return 1;
  if (c.p2 > c.p1) return -1;
  return 0;
};

PlagueAdvanced.prototype.getBoardForNN = function (player) {
  // 2-channel encoding: channel 0 = pieces (flipped), channel 1 = walls
  // Flattened: [pieces_0..pieces_n, walls_0..walls_n]
  var state = new Float32Array(this.size * 2);
  for (var i = 0; i < this.size; i++) {
    var v = this.board[i];
    if (v === WALL) {
      state[this.size + i] = 1; // wall channel
    } else {
      state[i] = v * player; // piece channel (flipped perspective)
    }
  }
  return state;
};

PlagueAdvanced.prototype.getBoardChannels = function () { return 2; };

// Renderer
var advancedRenderer = {
  label: 'Plague Advanced',
  cellColor: function (val) {
    if (val === 1) return [0, 255, 136];    // green
    if (val === -1) return [255, 51, 102];   // red
    if (val === WALL) return [80, 70, 90];   // dark purple/gray wall
    return [30, 30, 58];                     // empty
  },
  humanCellColor: function (val) {
    if (val === 1) return '#00ff88';
    if (val === -1) return '#ff3366';
    if (val === WALL) return '#504660';
    return '#1e1e3a';
  }
};

registerGame('plague_advanced', function (rows, cols) {
  return new PlagueAdvanced(rows, cols);
}, advancedRenderer);
