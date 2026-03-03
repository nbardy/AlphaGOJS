// Game facade: imports both game variants (triggering self-registration)
// and re-exports registry functions.
//
// Backward compat: Game(rows, cols, walls) creates the appropriate variant.

import './games/plague_classic';
import './games/plague_walls';
import { createGame, getRenderer, listGames } from './games/registry';

export { createGame, getRenderer, listGames };

// Backward-compatible constructor alias.
// walls=true (default) => 'plague_walls', walls=false => 'plague_classic'.
export function Game(rows, cols, walls) {
  return createGame(walls !== false ? 'plague_walls' : 'plague_classic', rows, cols);
}
