import * as tf from '@tensorflow/tfjs';
import { maskedSoftmax, sampleFromProbs } from './action';

// Checkpoint pool for Elo-rated self-play.
// Saves model weight snapshots periodically. During training, some games
// pit the current model (P1) against a past checkpoint (P2).
// Elo ratings update from those game outcomes — no separate eval needed.
//
// Weights stored as CPU Float32Arrays to avoid GPU memory pressure.

export class CheckpointPool {
  constructor(createModelFn) {
    this.checkpoints = [];       // { weights: [{shape,data}], elo, gen }
    this.currentElo = 1000;
    this.maxCheckpoints = 20;
    this.saveInterval = 20;      // save checkpoint every N generations
    this.checkpointFraction = 0.1; // 10% of game slots play vs checkpoint
    this.eloK = 32;
    this.createModelFn = createModelFn;
    this.opponentModel = null;
    this.loadedIdx = -1;

    // Track recent checkpoint match results for UI
    this.recentWins = 0;
    this.recentGames = 0;
  }

  save(kerasModel, generation) {
    var weights = kerasModel.getWeights().map(function (w) {
      var d = w.dataSync();
      return { shape: w.shape.slice(), data: new Float32Array(d) };
    });
    this.checkpoints.push({
      weights: weights,
      elo: this.currentElo,
      gen: generation
    });
    if (this.checkpoints.length > this.maxCheckpoints) {
      this.checkpoints.shift();
      if (this.loadedIdx >= 0) this.loadedIdx--;
    }
  }

  hasCheckpoints() {
    return this.checkpoints.length > 0;
  }

  shouldSave(generation) {
    return generation > 0 && generation % this.saveInterval === 0;
  }

  // Should this game slot play against a checkpoint?
  shouldBeCheckpointGame() {
    return this.checkpoints.length > 0 && Math.random() < this.checkpointFraction;
  }

  _ensureOpponent() {
    if (!this.opponentModel) {
      this.opponentModel = this.createModelFn();
    }
  }

  // Load a random checkpoint into the opponent model.
  // Call once per generation/batch, not per game.
  loadRandomOpponent() {
    if (this.checkpoints.length === 0) return -1;
    this._ensureOpponent();
    var idx = Math.floor(Math.random() * this.checkpoints.length);
    var ckpt = this.checkpoints[idx];
    var tensors = ckpt.weights.map(function (w) {
      return tf.tensor(w.data, w.shape);
    });
    this.opponentModel.model.setWeights(tensors);
    tensors.forEach(function (t) { t.dispose(); });
    this.loadedIdx = idx;
    return idx;
  }

  // Batch action selection for the checkpoint opponent.
  selectActions(states, masks) {
    if (!this.opponentModel || states.length === 0) return [];
    var boardSize = this.opponentModel.boardSize;
    var n = states.length;

    var flat = new Float32Array(n * boardSize);
    for (var i = 0; i < n; i++) flat.set(states[i], i * boardSize);
    var statesTensor = tf.tensor2d(flat, [n, boardSize]);
    var out = this.opponentModel.forward(statesTensor);
    var logitsData = out.policy.dataSync();
    out.policy.dispose();
    out.value.dispose();
    statesTensor.dispose();

    var results = [];
    for (var i = 0; i < n; i++) {
      var logits = new Float32Array(boardSize);
      for (var j = 0; j < boardSize; j++) logits[j] = logitsData[i * boardSize + j];
      var probs = maskedSoftmax(logits, masks[i]);
      var action = sampleFromProbs(probs);
      results.push({ action: action });
    }
    return results;
  }

  // Update Elo after a checkpoint game finishes.
  // currentWon: true if current model (P1) won.
  updateElo(currentWon, draw) {
    if (this.loadedIdx < 0 || this.loadedIdx >= this.checkpoints.length) return;
    var ckpt = this.checkpoints[this.loadedIdx];
    var expected = 1 / (1 + Math.pow(10, (ckpt.elo - this.currentElo) / 400));
    var actual = draw ? 0.5 : (currentWon ? 1 : 0);
    this.currentElo += this.eloK * (actual - expected);
    ckpt.elo += this.eloK * ((1 - actual) - (1 - expected));

    this.recentGames++;
    if (currentWon) this.recentWins++;
  }

  getCurrentElo() {
    return this.currentElo;
  }

  getRecentWinRate() {
    return this.recentGames > 0 ? this.recentWins / this.recentGames : 0;
  }

  resetRecentStats() {
    this.recentWins = 0;
    this.recentGames = 0;
  }
}
