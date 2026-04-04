// Shared random wall layout for plague_walls — used by CPU game and GPU engine
// so worker and main-thread rules stay aligned.

export var PLAGUE_WALL_CELL = 2;

/**
 * Fill `board` (length rows*cols) with 0 and PLAGUE_WALL_CELL wall segments.
 * Same algorithm as legacy PlagueWalls._generateWalls (center 3×3 cleared).
 * @param {Int8Array} board
 * @param {number} rows
 * @param {number} cols
 * @param {function(): number} [random] — defaults to Math.random
 */
export function generatePlagueWallsInto(board, rows, cols, random) {
  var WALL = PLAGUE_WALL_CELL;
  random = random || Math.random;
  var area = rows * cols;
  board.fill(0);

  var baseChains = 5 + Math.floor(random() * 8);
  var numChains = Math.round((baseChains * area) / 100);
  var DR = [0, 1, 0, -1];
  var DC = [1, 0, -1, 0];

  for (var chain = 0; chain < numChains; chain++) {
    var r = Math.floor(random() * rows);
    var c = Math.floor(random() * cols);
    var len = 1 + Math.floor(random() * 4);
    var dir = Math.floor(random() * 4);

    for (var seg = 0; seg < len; seg++) {
      if (r < 0 || r >= rows || c < 0 || c >= cols) break;
      board[r * cols + c] = WALL;
      r += DR[dir];
      c += DC[dir];
      if (random() < 0.3) {
        dir = (dir + (random() < 0.5 ? 1 : 3)) % 4;
      }
    }
  }

  var midR = Math.floor(rows / 2);
  var midC = Math.floor(cols / 2);
  for (var dr = -1; dr <= 1; dr++) {
    for (var dc = -1; dc <= 1; dc++) {
      var idx = (midR + dr) * cols + (midC + dc);
      if (idx >= 0 && idx < area) board[idx] = 0;
    }
  }
}
