// Game Registry: thin dispatcher for game creation.
// Each game file self-registers on import via registerGame().
//
// One Clean Path: the registry is a pure lookup table.
// No conditional logic, no fallbacks — unknown IDs throw.

var games = {};

// registerGame(id, factory, renderer)
//   id       — unique string key (e.g. 'plague_classic')
//   factory  — (rows, cols) => game instance conforming to the game interface
//   renderer — { label, cellColor(val) => [R,G,B], humanCellColor(val) => CSS string }
export function registerGame(id, factory, renderer) {
  if (games[id]) throw new Error('Game already registered: ' + id);
  games[id] = { factory: factory, renderer: renderer };
}

// createGame(id, rows, cols) => game instance
// Exhaustive: unknown id throws (no silent fallback).
export function createGame(id, rows, cols) {
  var entry = games[id];
  if (!entry) throw new Error('Unknown game type: ' + id + '. Registered: ' + Object.keys(games).join(', '));
  return entry.factory(rows, cols);
}

// getRenderer(id) => renderer object
export function getRenderer(id) {
  var entry = games[id];
  if (!entry) throw new Error('Unknown game type: ' + id);
  return entry.renderer;
}

// listGames() => [{id, label}]
export function listGames() {
  var list = [];
  var ids = Object.keys(games);
  for (var i = 0; i < ids.length; i++) {
    list.push({ id: ids[i], label: games[ids[i]].renderer.label });
  }
  return list;
}
