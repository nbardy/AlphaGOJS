// Cell values: 0=empty, 1=player1, -1=player2, 2=wall (blocker)
var WALL = 2;

export class Game {
  constructor(rows, cols, walls) {
    this.rows = rows || 10;
    this.cols = cols || 10;
    this.size = this.rows * this.cols;
    this.walls = walls !== false; // default: walls on
    this.board = new Int8Array(this.size);
    if (this.walls) this._generateWalls();
  }

  // Random wall chains scaled to board area.
  // ~5-12 chains per 100 cells, each 1-4 segments with 30% turn chance.
  // Center 3x3 always cleared so the board stays playable.
  _generateWalls() {
    var rows = this.rows, cols = this.cols;
    var area = rows * cols;
    var baseChains = 5 + Math.floor(Math.random() * 8); // 5-12 for 100 cells
    var numChains = Math.round(baseChains * area / 100);
    var DR = [0, 1, 0, -1];
    var DC = [1, 0, -1, 0];

    for (var chain = 0; chain < numChains; chain++) {
      var r = Math.floor(Math.random() * rows);
      var c = Math.floor(Math.random() * cols);
      var len = 1 + Math.floor(Math.random() * 4);
      var dir = Math.floor(Math.random() * 4);

      for (var seg = 0; seg < len; seg++) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) break;
        this.board[r * cols + c] = WALL;
        r += DR[dir];
        c += DC[dir];
        if (Math.random() < 0.3) {
          dir = (dir + (Math.random() < 0.5 ? 1 : 3)) % 4;
        }
      }
    }

    // Clear center 3x3 so mid-board is always playable
    var midR = Math.floor(rows / 2), midC = Math.floor(cols / 2);
    for (var dr = -1; dr <= 1; dr++) {
      for (var dc = -1; dc <= 1; dc++) {
        var idx = (midR + dr) * cols + (midC + dc);
        if (idx >= 0 && idx < this.size) this.board[idx] = 0;
      }
    }
  }

  reset() {
    this.board.fill(0);
    if (this.walls) this._generateWalls();
  }

  getValidMoves() {
    var moves = [];
    for (var i = 0; i < this.size; i++) {
      if (this.board[i] === 0) moves.push(i);
    }
    return moves;
  }

  getValidMovesMask() {
    var mask = new Float32Array(this.size);
    for (var i = 0; i < this.size; i++) {
      if (this.board[i] === 0) mask[i] = 1;
    }
    return mask;
  }

  makeMove(player, index) {
    if (index < 0 || index >= this.size || this.board[index] !== 0) return false;
    this.board[index] = player;
    return true;
  }

  spreadPlague() {
    var newBoard = Int8Array.from(this.board);
    var rows = this.rows, cols = this.cols, board = this.board;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var i = r * cols + c;
        if (board[i] !== 0) continue; // skip walls, occupied
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
  }

  isGameOver() {
    for (var i = 0; i < this.size; i++) {
      if (this.board[i] === 0) return false;
    }
    return true;
  }

  countCells() {
    var p1 = 0, p2 = 0, empty = 0;
    for (var i = 0; i < this.size; i++) {
      var v = this.board[i];
      if (v === 1) p1++;
      else if (v === -1) p2++;
      else if (v === 0) empty++;
      // walls (v === WALL) don't count for either side
    }
    return { p1: p1, p2: p2, empty: empty };
  }

  getWinner() {
    var c = this.countCells();
    if (c.p1 > c.p2) return 1;
    if (c.p2 > c.p1) return -1;
    return 0;
  }

  getBoardForNN(player) {
    // Single-channel encoding: own=+1, opponent=-1, empty=0, wall=0.5
    // Wall value is perspective-independent so the model learns spatial terrain.
    var state = new Float32Array(this.size);
    for (var i = 0; i < this.size; i++) {
      var v = this.board[i];
      if (v === WALL) state[i] = 0.5;
      else state[i] = v * player;
    }
    return state;
  }
}
