import { Game } from './game';

export class SelfPlayTrainer {
  constructor(model, config) {
    this.model = model;
    this.numGames = config.numGames || 80;
    this.rows = config.rows || 20;
    this.cols = config.cols || 20;
    this.boardSize = this.rows * this.cols;
    this.trainBatchSize = config.trainBatchSize || 512;
    this.trainInterval = config.trainInterval || 30;
    this.snapshotInterval = config.snapshotInterval || 50;
    this.evalInterval = config.evalInterval || 100;
    this.evalGames = config.evalGames || 20;

    this.games = [];
    this.experienceBuffer = [];
    this.maxBufferSize = 20000;
    this.gamesCompleted = 0;
    this.gamesSinceLastTrain = 0;
    this.generation = 0;
    this.lastLoss = 0;
    this.p1Wins = 0;
    this.p2Wins = 0;
    this.draws = 0;
    this.recentLengths = [];

    // Snapshot trail
    this.trail = []; // { gen, vsRandom }
    this.lastEvalGen = -1;

    for (var i = 0; i < this.numGames; i++) {
      this.games.push(this._newSlot());
    }
  }

  _newSlot() {
    return { game: new Game(this.rows, this.cols), history: [], turn: 0, done: false, winner: 0 };
  }

  tick() {
    var i, j, gi;

    // Player 1 moves
    var s1 = [], m1 = [], idx1 = [];
    for (i = 0; i < this.numGames; i++) {
      if (this.games[i].done) continue;
      var mask = this.games[i].game.getValidMovesMask();
      var hasValid = false;
      for (j = 0; j < this.boardSize; j++) { if (mask[j] > 0) { hasValid = true; break; } }
      if (!hasValid) { this._finish(i); continue; }
      s1.push(this.games[i].game.getBoardForNN(1));
      m1.push(mask);
      idx1.push(i);
    }
    if (idx1.length > 0) {
      var res1 = this.model.getActions(s1, m1);
      for (j = 0; j < idx1.length; j++) {
        gi = idx1[j];
        this.games[gi].game.makeMove(1, res1.actions[j]);
        this.games[gi].history.push({ state: s1[j], action: res1.actions[j], player: 1, value: res1.values[j] });
      }
    }

    // Player -1 moves
    var s2 = [], m2 = [], idx2 = [];
    for (i = 0; i < this.numGames; i++) {
      if (this.games[i].done) continue;
      var mask2 = this.games[i].game.getValidMovesMask();
      var hasValid2 = false;
      for (j = 0; j < this.boardSize; j++) { if (mask2[j] > 0) { hasValid2 = true; break; } }
      if (!hasValid2) { this._finish(i); continue; }
      s2.push(this.games[i].game.getBoardForNN(-1));
      m2.push(mask2);
      idx2.push(i);
    }
    if (idx2.length > 0) {
      var res2 = this.model.getActions(s2, m2);
      for (j = 0; j < idx2.length; j++) {
        gi = idx2[j];
        this.games[gi].game.makeMove(-1, res2.actions[j]);
        this.games[gi].history.push({ state: s2[j], action: res2.actions[j], player: -1, value: res2.values[j] });
      }
    }

    // Spread plague and check game over
    for (i = 0; i < this.numGames; i++) {
      if (this.games[i].done) continue;
      this.games[i].game.spreadPlague();
      this.games[i].turn++;
      if (this.games[i].game.isGameOver()) this._finish(i);
    }

    this._maybeTrainAndRestart();
  }

  _finish(index) {
    var gs = this.games[index];
    if (gs.done) return;
    gs.done = true;
    gs.winner = gs.game.getWinner();

    var counts = gs.game.countCells();
    var total = counts.p1 + counts.p2;
    if (total === 0) total = 1;

    for (var k = 0; k < gs.history.length; k++) {
      var exp = gs.history[k];
      // Margin-based reward from this player's perspective
      var margin = exp.player === 1
        ? (counts.p1 - counts.p2) / total
        : (counts.p2 - counts.p1) / total;
      this.experienceBuffer.push({ state: exp.state, action: exp.action, reward: margin, value: exp.value });
    }

    if (this.experienceBuffer.length > this.maxBufferSize) {
      this.experienceBuffer = this.experienceBuffer.slice(-this.maxBufferSize);
    }

    this.gamesCompleted++;
    this.gamesSinceLastTrain++;
    if (gs.winner === 1) this.p1Wins++;
    else if (gs.winner === -1) this.p2Wins++;
    else this.draws++;

    this.recentLengths.push(gs.turn);
    if (this.recentLengths.length > 200) this.recentLengths.shift();
  }

  _maybeTrainAndRestart() {
    if (this.gamesSinceLastTrain >= this.trainInterval && this.experienceBuffer.length >= this.trainBatchSize) {
      var batch = this.experienceBuffer.splice(0, this.trainBatchSize);
      this.lastLoss = this.model.train(batch);
      this.generation++;
      this.gamesSinceLastTrain = 0;

      // Snapshot
      if (this.generation % this.snapshotInterval === 0) {
        this.model.saveSnapshot(this.generation);
      }

      // Evaluate vs random
      if (this.generation % this.evalInterval === 0 && this.generation !== this.lastEvalGen) {
        this.lastEvalGen = this.generation;
        var wr = this._evalVsRandom(this.evalGames);
        this.trail.push({ gen: this.generation, vsRandom: wr });
        if (this.trail.length > 50) this.trail.shift();
      }
    }

    // Immediately restart finished games (no dead slot wait)
    for (var i = 0; i < this.numGames; i++) {
      if (this.games[i].done) this.games[i] = this._newSlot();
    }
  }

  _evalVsRandom(numGames) {
    var wins = 0;
    for (var g = 0; g < numGames; g++) {
      var game = new Game(this.rows, this.cols);
      var modelPlayer = (g % 2 === 0) ? 1 : -1;
      var safety = 0;
      while (!game.isGameOver() && safety < 200) {
        safety++;
        // Player 1
        var moves = game.getValidMoves();
        if (moves.length === 0) break;
        if (modelPlayer === 1) {
          var state = game.getBoardForNN(1);
          var mask = game.getValidMovesMask();
          var action = this.model.getAction(state, mask);
          game.makeMove(1, action);
        } else {
          game.makeMove(1, moves[Math.floor(Math.random() * moves.length)]);
        }

        if (game.isGameOver()) break;
        moves = game.getValidMoves();
        if (moves.length === 0) break;

        // Player -1
        if (modelPlayer === -1) {
          var state2 = game.getBoardForNN(-1);
          var mask2 = game.getValidMovesMask();
          var action2 = this.model.getAction(state2, mask2);
          game.makeMove(-1, action2);
        } else {
          game.makeMove(-1, moves[Math.floor(Math.random() * moves.length)]);
        }

        game.spreadPlague();
      }
      if (game.getWinner() === modelPlayer) wins++;
    }
    return wins / numGames;
  }

  getStats() {
    var avgLen = 0;
    if (this.recentLengths.length > 0) {
      var sum = 0;
      for (var i = 0; i < this.recentLengths.length; i++) sum += this.recentLengths[i];
      avgLen = sum / this.recentLengths.length;
    }
    return {
      gamesCompleted: this.gamesCompleted,
      generation: this.generation,
      loss: this.lastLoss,
      p1Wins: this.p1Wins,
      p2Wins: this.p2Wins,
      draws: this.draws,
      avgGameLength: avgLen,
      bufferSize: this.experienceBuffer.length,
      trail: this.trail,
      snapshots: this.model.snapshots.length
    };
  }
}
