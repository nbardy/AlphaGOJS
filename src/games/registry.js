// Game Registry
// Each game type registers a factory and a renderer.
//
// Game instance interface (what trainer consumes):
//   .rows, .cols, .size          — dimensions
//   .board                       — Int8Array of cell values
//   .getValidMoves()             — returns array of valid cell indices
//   .getValidMovesMask()         — returns Float32Array, 1 for valid cells
//   .makeMove(player, index)     — place piece, returns boolean
//   .step()                      — post-move phase (plague spread, etc.)
//   .isGameOver()                — boolean
//   .countCells()                — { p1, p2, empty }
//   .getWinner()                 — 1, -1, or 0
//   .getBoardForNN(player)       — Float32Array, flipped perspective
//   .getBoardChannels()          — number of NN input channels per cell (default 1)
//
// Renderer interface (what UI consumes):
//   .cellColor(value)            — returns [r, g, b] for a cell value
//   .humanCellColor(value)       — returns CSS color string
//   .label                       — display name

var registry = {};

export function registerGame(id, factory, renderer) {
  registry[id] = { factory: factory, renderer: renderer };
}

export function createGame(id, rows, cols) {
  if (!registry[id]) throw new Error('Unknown game: ' + id);
  return registry[id].factory(rows, cols);
}

export function getRenderer(id) {
  if (!registry[id]) throw new Error('Unknown game: ' + id);
  return registry[id].renderer;
}

export function listGames() {
  var ids = [];
  for (var id in registry) ids.push({ id: id, label: registry[id].renderer.label });
  return ids;
}
