import { Game } from './game';
import { sampleFromProbs } from './action';

// Evaluate model against a random agent.
// Plays numGames as each side (2*numGames total), returns aggregate stats.

export function evaluateVsRandom(algo, rows, cols, numGames) {
  numGames = numGames || 10;
  var wins = 0;
  var losses = 0;
  var draws = 0;
  var boardSize = rows * cols;

  for (var side = 0; side < 2; side++) {
    var modelPlayer = side === 0 ? 1 : -1;

    for (var g = 0; g < numGames; g++) {
      var game = new Game(rows, cols);
      var maxTurns = boardSize * 2;

      for (var turn = 0; turn < maxTurns; turn++) {
        // Player 1 move
        var mask1 = game.getValidMovesMask();
        var hasValid1 = false;
        for (var j = 0; j < boardSize; j++) { if (mask1[j] > 0) { hasValid1 = true; break; } }
        if (!hasValid1) break;

        if (modelPlayer === 1) {
          var state1 = game.getBoardForNN(1);
          var action1 = algo.selectAction(state1, mask1);
          game.makeMove(1, action1);
        } else {
          game.makeMove(1, _randomAction(mask1));
        }

        // Player -1 move
        var mask2 = game.getValidMovesMask();
        var hasValid2 = false;
        for (var j = 0; j < boardSize; j++) { if (mask2[j] > 0) { hasValid2 = true; break; } }
        if (!hasValid2) break;

        if (modelPlayer === -1) {
          var state2 = game.getBoardForNN(-1);
          var action2 = algo.selectAction(state2, mask2);
          game.makeMove(-1, action2);
        } else {
          game.makeMove(-1, _randomAction(mask2));
        }

        game.spreadPlague();
        if (game.isGameOver()) break;
      }

      var winner = game.getWinner();
      if (winner === modelPlayer) wins++;
      else if (winner === 0) draws++;
      else losses++;
    }
  }

  var total = wins + losses + draws;
  return {
    winRate: wins / total,
    lossRate: losses / total,
    drawRate: draws / total
  };
}

function _randomAction(mask) {
  var valid = [];
  for (var i = 0; i < mask.length; i++) {
    if (mask[i] > 0) valid.push(i);
  }
  return valid[Math.floor(Math.random() * valid.length)];
}
