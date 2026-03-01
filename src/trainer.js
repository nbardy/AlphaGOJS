import { Game } from './game';

// SelfPlayTrainer orchestrates parallel self-play games.
// It is algorithm-agnostic: it calls algo.selectActions() and algo.onGameFinished().
// History stores full ActionResult metadata via Object.assign (no branching on algo type).
//
// Checkpoint integration: when a checkpointPool is provided, ~10% of game slots
// play current model (P1) vs a past checkpoint (P2). These games update Elo
// ratings and still contribute P1 moves to the training buffer.

export class SelfPlayTrainer {
  constructor(algo, config) {
    this.algo = algo;
    this.numGames = config.numGames || 40;
    this.rows = config.rows || 10;
    this.cols = config.cols || 10;
    this.boardSize = this.rows * this.cols;
    this.trainBatchSize = config.trainBatchSize || 256;
    this.trainInterval = config.trainInterval || 20;
    this.checkpointPool = config.checkpointPool || null;

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
    var vs = this.checkpointPool && this.checkpointPool.shouldBeCheckpointGame();
    return {
      game: new Game(this.rows, this.cols),
      history: [],
      turn: 0,
      done: false,
      doneTime: 0,
      winner: 0,
      vsCheckpoint: vs
    };
  }

  tick() {
    var i, j, gi;

    // Player 1 moves — always current model
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
        // Store P1 moves in history (used for training)
        var entry = { state: s1[j], player: 1, mask: m1[j] };
        Object.assign(entry, results1[j]);
        this.games[gi].history.push(entry);
      }
    }

    // Player -1 moves — split self-play vs checkpoint
    var s2 = [], m2 = [], idx2 = [];
    var s2c = [], m2c = [], idx2c = [];
    for (i = 0; i < this.numGames; i++) {
      if (this.games[i].done) continue;
      var mask2 = this.games[i].game.getValidMovesMask();
      var hasValid2 = false;
      for (j = 0; j < this.boardSize; j++) { if (mask2[j] > 0) { hasValid2 = true; break; } }
      if (!hasValid2) { this._finish(i); continue; }
      var boardState = this.games[i].game.getBoardForNN(-1);
      if (this.games[i].vsCheckpoint && this.checkpointPool) {
        s2c.push(boardState);
        m2c.push(mask2);
        idx2c.push(i);
      } else {
        s2.push(boardState);
        m2.push(mask2);
        idx2.push(i);
      }
    }

    // Self-play P2 — current model
    if (idx2.length > 0) {
      var results2 = this.algo.selectActions(s2, m2);
      for (j = 0; j < idx2.length; j++) {
        gi = idx2[j];
        this.games[gi].game.makeMove(-1, results2[j].action);
        var entry2 = { state: s2[j], player: -1, mask: m2[j] };
        Object.assign(entry2, results2[j]);
        this.games[gi].history.push(entry2);
      }
    }

    // Checkpoint P2 — opponent model plays, no history (not training data)
    if (idx2c.length > 0 && this.checkpointPool) {
      var results2c = this.checkpointPool.selectActions(s2c, m2c);
      for (j = 0; j < idx2c.length; j++) {
        gi = idx2c[j];
        this.games[gi].game.makeMove(-1, results2c[j].action);
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

    if (gs.vsCheckpoint && this.checkpointPool) {
      // Checkpoint game: train on P1 (current model) moves only, update Elo
      this.algo.onGameFinished(gs.history, gs.winner);
      var currentWon = gs.winner === 1;
      var draw = gs.winner === 0;
      this.checkpointPool.updateElo(currentWon, draw);
    } else {
      // Self-play: train on both sides
      this.algo.onGameFinished(gs.history, gs.winner);
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
    if (this.algo.shouldTrain(this.gamesSinceLastTrain, this.trainInterval, this.trainBatchSize)) {
      this.lastLoss = this.algo.train(this.trainBatchSize);
      this.generation++;
      this.gamesSinceLastTrain = 0;

      // Save checkpoint + load new opponent for next batch
      if (this.checkpointPool) {
        if (this.checkpointPool.shouldSave(this.generation)) {
          this.checkpointPool.save(this.algo.model.model, this.generation);
        }
        if (this.checkpointPool.hasCheckpoints()) {
          this.checkpointPool.loadRandomOpponent();
        }
      }
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
      bufferSize: this.algo.getBufferSize(),
      elo: this.checkpointPool ? this.checkpointPool.getCurrentElo() : 0,
      checkpointWinRate: this.checkpointPool ? this.checkpointPool.getRecentWinRate() : 0
    };
  }
}
