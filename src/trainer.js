import { Game } from './game';

// SelfPlayTrainer orchestrates parallel self-play games.
// It is algorithm-agnostic: it calls algo.selectActions() and algo.onGameFinished().
// History stores full ActionResult metadata via Object.assign (no branching on algo type).

export class SelfPlayTrainer {
  constructor(algo, config) {
    this.algo = algo;
    this.numGames = config.numGames || 40;
    this.rows = config.rows || 10;
    this.cols = config.cols || 10;
    this.boardSize = this.rows * this.cols;
    this.trainBatchSize = config.trainBatchSize || 256;
    this.trainInterval = config.trainInterval || 20;

    this.games = [];
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
      var results1 = this.algo.selectActions(s1, m1);
      for (j = 0; j < idx1.length; j++) {
        gi = idx1[j];
        this.games[gi].game.makeMove(1, results1[j].action);
        // Store state + player + all algo metadata (logProb, value for PPO, etc.)
        var entry = { state: s1[j], player: 1 };
        Object.assign(entry, results1[j]);
        this.games[gi].history.push(entry);
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
      var results2 = this.algo.selectActions(s2, m2);
      for (j = 0; j < idx2.length; j++) {
        gi = idx2[j];
        this.games[gi].game.makeMove(-1, results2[j].action);
        var entry2 = { state: s2[j], player: -1 };
        Object.assign(entry2, results2[j]);
        this.games[gi].history.push(entry2);
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

    // Delegate experience processing to the algorithm
    this.algo.onGameFinished(gs.history, gs.winner);

    this.gamesCompleted++;
    this.gamesSinceLastTrain++;
    if (gs.winner === 1) this.p1Wins++;
    else if (gs.winner === -1) this.p2Wins++;
    else this.draws++;

    this.recentLengths.push(gs.turn);
    if (this.recentLengths.length > 100) this.recentLengths.shift();
  }

  _maybeTrainAndRestart() {
    if (this.algo.shouldTrain(this.gamesSinceLastTrain, this.trainInterval, this.trainBatchSize)) {
      this.lastLoss = this.algo.train(this.trainBatchSize);
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
      bufferSize: this.algo.getBufferSize()
    };
  }
}
