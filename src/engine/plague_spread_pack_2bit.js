/**
 * Pack / unpack plague cell codes (0–3) into 2 bits × 16 cells per u32.
 * Layout: flat row-major index i → word floor(i/16), shift (i%16)*2 (works for any cols).
 */

export function wordsPerGameBoard(rows, cols) {
  var n = rows * cols;
  return (n + 15) >> 4;
}

export function packedBufferLength(rows, cols, numGames) {
  return wordsPerGameBoard(rows, cols) * numGames;
}

/** One game: unpacked length rows*cols, values 0–3 */
export function packUint32CellsTo2Bit(unpacked, boardSize) {
  var words = (boardSize + 15) >> 4;
  var out = new Uint32Array(words);
  for (var i = 0; i < boardSize; i++) {
    var w = (i / 16) | 0;
    var sh = (i % 16) * 2;
    out[w] |= (unpacked[i] & 3) << sh;
  }
  return out;
}

export function unpack2BitToUint32(packed, boardSize) {
  var out = new Uint32Array(boardSize);
  for (var i = 0; i < boardSize; i++) {
    var w = (i / 16) | 0;
    var sh = (i % 16) * 2;
    out[i] = (packed[w] >>> sh) & 3;
  }
  return out;
}

/** All games contiguous: unpacked layout [game][cell] flattened */
export function packAllGamesUint32To2Bit(unpackedAllGames, rows, cols, numGames) {
  var boardSize = rows * cols;
  var wpg = wordsPerGameBoard(rows, cols);
  var out = new Uint32Array(wpg * numGames);
  for (var g = 0; g < numGames; g++) {
    var slice = unpackedAllGames.subarray(g * boardSize, g * boardSize + boardSize);
    var pw = packUint32CellsTo2Bit(slice, boardSize);
    out.set(pw, g * wpg);
  }
  return out;
}

export function unpackAllGames2BitToUint32(packed, rows, cols, numGames) {
  var boardSize = rows * cols;
  var wpg = wordsPerGameBoard(rows, cols);
  var out = new Uint32Array(boardSize * numGames);
  for (var g = 0; g < numGames; g++) {
    var sub = packed.subarray(g * wpg, g * wpg + wpg);
    var u = unpack2BitToUint32(sub, boardSize);
    out.set(u, g * boardSize);
  }
  return out;
}
