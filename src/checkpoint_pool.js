import * as tf from '@tensorflow/tfjs';
import { maskedSoftmax, sampleFromProbs } from './action';

// Checkpoint pool for Elo-rated self-play.
// Saves model weight snapshots periodically. During training, some games
// pit the current model (P1) against a past checkpoint (P2).
// Elo ratings update from those game outcomes — no separate eval needed.
//
// Weights stored as CPU Float32Arrays to avoid GPU memory pressure.

export class CheckpointPool {
  constructor(createModelFn, config) {
    config = config || {};
    this.checkpoints = [];       // { id, weights: [{shape,data}], elo, gen }
    this.nextCheckpointId = 1;
    this.currentElo = 1000;
    this.maxCheckpoints = config.maxCheckpoints || 50;
    this.saveInterval = config.saveInterval || 20;      // save checkpoint every N generations
    this.checkpointFraction = typeof config.checkpointFraction === 'number' ? config.checkpointFraction : 0.25;
    this.sampleMode = config.sampleMode || 'uniform_recent';
    this.recentWindow = config.recentWindow || 50;
    this.recentBiasDecay = typeof config.recentBiasDecay === 'number' ? config.recentBiasDecay : 3;
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
      id: this.nextCheckpointId++,
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

  _pickCheckpointIndex() {
    var n = this.checkpoints.length;
    if (n === 0) return -1;

    if (this.sampleMode === 'uniform_all') {
      return Math.floor(Math.random() * n);
    }

    // League-style: uniformly sample from a recent opponent window.
    // This prevents catastrophic forgetting against older strategies while
    // keeping difficulty close to the current policy.
    if (this.sampleMode === 'uniform_recent') {
      var window = Math.max(1, Math.min(this.recentWindow, n));
      var start = n - window;
      return start + Math.floor(Math.random() * window);
    }

    // Legacy option: recency-biased exponential sampling.
    if (this.sampleMode === 'recent_bias') {
      var weights = new Float64Array(n);
      var total = 0;
      for (var i = 0; i < n; i++) {
        weights[i] = Math.exp(this.recentBiasDecay * i / n);
        total += weights[i];
      }
      var r = Math.random() * total;
      var cumul = 0;
      var idx = n - 1;
      for (var i = 0; i < n; i++) {
        cumul += weights[i];
        if (r <= cumul) { idx = i; break; }
      }
      return idx;
    }

    // Fallback: uniform over entire pool.
    return Math.floor(Math.random() * n);
  }

  pickRandomOpponentIndex() {
    return this._pickCheckpointIndex();
  }

  _findIndexById(id) {
    for (var i = 0; i < this.checkpoints.length; i++) {
      if (this.checkpoints[i].id === id) return i;
    }
    return -1;
  }

  pickRandomOpponentId() {
    var idx = this._pickCheckpointIndex();
    if (idx < 0) return -1;
    return this.checkpoints[idx].id;
  }

  loadOpponentByIndex(idx) {
    if (idx < 0 || idx >= this.checkpoints.length) return -1;
    this._ensureOpponent();
    var ckpt = this.checkpoints[idx];
    var tensors = ckpt.weights.map(function (w) {
      return tf.tensor(w.data, w.shape);
    });
    this.opponentModel.model.setWeights(tensors);
    tensors.forEach(function (t) { t.dispose(); });
    this.loadedIdx = idx;
    return idx;
  }

  loadOpponentById(id) {
    var idx = this._findIndexById(id);
    if (idx < 0) return -1;
    return this.loadOpponentByIndex(idx);
  }

  // Load a checkpoint into the opponent model.
  loadRandomOpponent() {
    if (this.checkpoints.length === 0) return -1;
    var idx = this._pickCheckpointIndex();
    return this.loadOpponentByIndex(idx);
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
    this.updateEloForIndex(this.loadedIdx, currentWon, draw);
  }

  updateEloForIndex(idx, currentWon, draw) {
    if (idx < 0 || idx >= this.checkpoints.length) return;
    var ckpt = this.checkpoints[idx];
    var expected = 1 / (1 + Math.pow(10, (ckpt.elo - this.currentElo) / 400));
    var actual = draw ? 0.5 : (currentWon ? 1 : 0);
    this.currentElo += this.eloK * (actual - expected);
    ckpt.elo += this.eloK * ((1 - actual) - (1 - expected));

    this.recentGames++;
    if (currentWon) this.recentWins++;
  }

  updateEloForId(id, currentWon, draw) {
    var idx = this._findIndexById(id);
    if (idx < 0) return;
    this.updateEloForIndex(idx, currentWon, draw);
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
