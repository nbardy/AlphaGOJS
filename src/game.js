export class Game {
  constructor(rows, cols) {
    this.rows = rows || 10;
    this.cols = cols || 10;
    this.size = this.rows * this.cols;
    this.board = new Int8Array(this.size);
  }

  reset() {
    this.board.fill(0);
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
      if (this.board[i] === 1) p1++;
      else if (this.board[i] === -1) p2++;
      else empty++;
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
    var state = new Float32Array(this.size);
    for (var i = 0; i < this.size; i++) {
      state[i] = this.board[i] * player;
    }
    return state;
  }
}
