import { Game } from './game';

export class SelfPlayTrainer {
  constructor(model, config) {
    this.model = model;
    this.numGames = config.numGames || 40;
    this.rows = config.rows || 10;
    this.cols = config.cols || 10;
    this.boardSize = this.rows * this.cols;
    this.trainBatchSize = config.trainBatchSize || 256;
    this.trainInterval = config.trainInterval || 20;

    this.games = [];
    this.experienceBuffer = [];
    this.maxBufferSize = 10000;
    this.gamesCompleted = 0;
    this.gamesSinceLastTrain = 0;
    this.generation = 0;
    this.lastLoss = 0;
    this.p1Wins = 0;
    this.p2Wins = 0;
    this.draws = 0;
    this.recentLengths = [];

    for (var i = 0; i < this.numGames; i++) {
      this.games.push(this._newSlot());
    }
  }

  _newSlot() {
    return {
      game: new Game(this.rows, this.cols),
      history: [],
      turn: 0,
      done: false,
      doneTime: 0,
      winner: 0
    };
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
      var acts1 = this.model.getActions(s1, m1);
      for (j = 0; j < idx1.length; j++) {
        gi = idx1[j];
        this.games[gi].game.makeMove(1, acts1[j]);
        this.games[gi].history.push({ state: s1[j], action: acts1[j], player: 1 });
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
      var acts2 = this.model.getActions(s2, m2);
      for (j = 0; j < idx2.length; j++) {
        gi = idx2[j];
        this.games[gi].game.makeMove(-1, acts2[j]);
        this.games[gi].history.push({ state: s2[j], action: acts2[j], player: -1 });
      }
    }

    // Spread plague and check game over
    for (i = 0; i < this.numGames; i++) {
      if (this.games[i].done) continue;
      this.games[i].game.spreadPlague();
      this.games[i].turn++;
      if (this.games[i].game.isGameOver()) {
        this._finish(i);
      }
    }

    this._maybeTrainAndRestart();
  }

  _finish(index) {
    var gs = this.games[index];
    if (gs.done) return;
    gs.done = true;
    gs.doneTime = Date.now();
    gs.winner = gs.game.getWinner();

    for (var k = 0; k < gs.history.length; k++) {
      var exp = gs.history[k];
      var reward = (exp.player === gs.winner) ? 1 : (gs.winner === 0 ? 0 : -1);
      this.experienceBuffer.push({ state: exp.state, action: exp.action, reward: reward });
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
    if (this.recentLengths.length > 100) this.recentLengths.shift();
  }

  _maybeTrainAndRestart() {
    if (this.gamesSinceLastTrain >= this.trainInterval && this.experienceBuffer.length >= this.trainBatchSize) {
      var batch = this.experienceBuffer.splice(0, this.trainBatchSize);
      this.lastLoss = this.model.train(batch);
      this.generation++;
      this.gamesSinceLastTrain = 0;
    }

    var now = Date.now();
    for (var i = 0; i < this.numGames; i++) {
      if (this.games[i].done && now - this.games[i].doneTime > 300) {
        this.games[i] = this._newSlot();
      }
    }
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
      bufferSize: this.experienceBuffer.length
    };
  }
}
