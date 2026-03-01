import * as tf from '@tensorflow/tfjs';
import { maskedSoftmax, sampleFromProbs, flattenStates } from './action';
import { Game } from './game';

// GPU-Native Trainer: entire game loop on GPU, ~320 bytes/tick transfer.
//
// State lives as tf.Tensor2D [N, boardSize] on GPU.
// Only game-over flags (160B) and canPlay mask (160B) cross the bus per tick.
// Trajectory snapshots are GPU tensors (tf.keep'd), downloaded on game end.
//
// Spread uses conv2d (distributional approximation — sums neighbors then
// multiplies by one random, vs CPU which multiplies each neighbor independently).
// Acceptable for training: model learns from observed dynamics either way.
//
// tf.multinomial WebGPU bug: first sample always 0. Workaround: request 2, take second.

export class GPUTrainer {
  constructor(model, config) {
    this.model = model;
    this.numGames = config.numGames || 40;
    this.rows = config.rows || 10;
    this.cols = config.cols || 10;
    this.boardSize = this.rows * this.cols;
    this.trainBatchSize = config.trainBatchSize || 256;
    this.trainInterval = config.trainInterval || 20;
    this.walls = config.walls !== false;

    // GPU state tensor [N, boardSize] — the core of zero-transfer
    this.state = tf.zeros([this.numGames, this.boardSize]);

    // Conv2d kernel for plague spread: 3x3 cross pattern
    this.neighborKernel = tf.tensor4d(
      [0, 1, 0, 1, 0, 1, 0, 1, 0],
      [3, 3, 1, 1]
    );

    // Optimizer (same as Reinforce)
    this.optimizer = tf.train.adam(0.001);

    // CPU metadata — trivial scalars, no GPU transfer needed
    this.turns = new Int32Array(this.numGames);
    this.done = new Uint8Array(this.numGames);
    this.winners = new Int8Array(this.numGames);
    this.doneTimes = new Float64Array(this.numGames);

    // Trajectory recording: GPU snapshots + generation tracking
    this._stepSnapshots = [];
    this._gameGenerations = new Int32Array(this.numGames);
    this._nextGeneration = 1;

    // CPU experience buffer (same format as Reinforce)
    this.buffer = [];
    this.maxBufferSize = 20000;

    // Checkpoint pool for Elo tracking (optional, injected from app.js)
    this.checkpointPool = config.checkpointPool || null;
    this._eloUpdateInFlight = false;

    // Stats
    this.gamesCompleted = 0;
    this.gamesSinceLastTrain = 0;
    this.generation = 0;
    this.lastLoss = 0;
    this.trainSteps = 0;
    this.p1Wins = 0;
    this.p2Wins = 0;
    this.draws = 0;
    this.recentLengths = [];
  }

  // ── Main tick: both players move, spread, check game over ──

  tick() {
    this._selectAndApplyMoves(1);
    this._selectAndApplyMoves(-1);
    this._spreadPlague();

    // Increment turns for active games (CPU scalars)
    for (var i = 0; i < this.numGames; i++) {
      if (!this.done[i]) this.turns[i]++;
    }

    this._checkGameOver();
    this._maybeTrainAndRestart();
  }

  // ── GPU action pipeline: mask → forward → softmax → multinomial → oneHot apply ──

  _selectAndApplyMoves(player) {
    var self = this;
    var N = this.numGames;
    var boardSize = this.boardSize;

    // Build canPlay float array from CPU done[] — ~160 bytes CPU→GPU
    var canPlayArr = new Float32Array(N);
    for (var i = 0; i < N; i++) canPlayArr[i] = this.done[i] ? 0 : 1;

    // Record PRE-MOVE state for trajectory — this is the state the agent sees
    // when deciding. Must be captured before tidy applies the move.
    var preMoveState = tf.keep(this.state.clone());

    var actionsKept;
    var oldState = this.state;
    this.state = tf.tidy(function () {
      var canPlay = tf.tensor1d(canPlayArr);

      // Empty cells mask [N, boardSize]
      var mask = oldState.equal(0).cast('float32');
      var hasValid = mask.sum(1).greater(0).cast('float32');
      var activeAndValid = canPlay.mul(hasValid);

      // NN forward — NO UPLOAD, state is already on GPU
      var perspective = oldState.mul(player);
      var combined = self.model.model.predict(perspective);
      var logits = combined.slice([0, 0], [-1, boardSize]);

      // GPU action selection: mask invalid → softmax → multinomial
      var maskedLogits = logits.add(mask.sub(1).mul(1e9));
      // tf.multinomial WebGPU bug: first sample always 0. Request 2, take second.
      var actions = tf.multinomial(maskedLogits, 2).slice([0, 1], [-1, 1]).squeeze([1]);

      // Keep action indices for trajectory recording (~320 bytes, not disposed by tidy)
      actionsKept = tf.keep(actions.clone());

      // Apply move: oneHot scatter, only for active games with valid moves
      var moveOneHot = tf.oneHot(actions.cast('int32'), boardSize).mul(player);
      var newState = oldState.add(moveOneHot.mul(activeAndValid.expandDims(1)));

      return newState;
    });
    oldState.dispose();

    // Record pre-move state + actions for trajectory (outside tidy)
    this._recordStep(player, preMoveState, actionsKept);
  }

  // ── GPU plague spread: conv2d neighbor sum + random ──
  // Distributional note: CPU version multiplies each neighbor independently
  // by random before summing. GPU version sums first, then multiplies by
  // one random. This is a distributional approximation — acceptable for
  // training since the model learns from observed dynamics either way,
  // and conv2d is much more GPU-efficient (1 kernel vs 4 slices).

  _spreadPlague() {
    var self = this;
    var N = this.numGames;
    var rows = this.rows;
    var cols = this.cols;

    var oldState = this.state;
    this.state = tf.tidy(function () {
      var grid = oldState.reshape([N, rows, cols, 1]);
      var neighborSum = tf.conv2d(grid, self.neighborKernel, 1, 'same');
      var rand = tf.randomUniform(neighborSum.shape);
      var weighted = neighborSum.mul(rand).mul(2);
      var emptyMask = grid.equal(0).cast('float32');
      var newVals = weighted.sign().mul(emptyMask);
      var occupied = grid.mul(tf.scalar(1).sub(emptyMask));
      return occupied.add(newVals).reshape([N, self.boardSize]);
    });
    oldState.dispose();
  }

  // ── Game-over check: the only mandatory GPU→CPU transfer (~160 bytes) ──

  _checkGameOver() {
    var emptyCounts = tf.tidy(function () {
      return this.state.equal(0).cast('float32').sum(1);
    }.bind(this));
    var data = emptyCounts.dataSync();
    emptyCounts.dispose();

    var doneIndices = [];
    for (var i = 0; i < this.numGames; i++) {
      if (!this.done[i] && data[i] === 0) {
        doneIndices.push(i);
      }
    }

    if (doneIndices.length > 0) {
      this._finishGames(doneIndices);
    }
  }

  // ── Trajectory recording: GPU snapshots, CPU buffer ──

  _recordStep(player, preMoveState, actions) {
    // Store pre-move state + action indices for correct REINFORCE training.
    // Pre-move state = what the agent saw when deciding (not post-move).
    // Actions tensor = actual indices chosen by tf.multinomial.
    var snap = {
      state: preMoveState,
      actions: actions,
      player: player,
      generations: Int32Array.from(this._gameGenerations)
    };
    this._stepSnapshots.push(snap);
  }

  _downloadTrajectory(gameIndex) {
    var gen = this._gameGenerations[gameIndex];
    var trajectory = [];
    var boardSize = this.boardSize;

    for (var s = 0; s < this._stepSnapshots.length; s++) {
      var snap = this._stepSnapshots[s];
      if (snap.generations[gameIndex] !== gen) continue;

      // Download just this game's state from the snapshot
      var stateSlice = tf.tidy(function () {
        return snap.state.slice([gameIndex, 0], [1, boardSize]).reshape([boardSize]);
      });
      var stateData = stateSlice.dataSync();
      stateSlice.dispose();

      // Apply perspective transform: model expects own pieces as positive.
      // Raw board has P1=+1, P2=-1. Multiply by player so the agent's
      // own pieces are always +1 (matches how inference does it).
      var state = new Float32Array(boardSize);
      var mask = new Float32Array(boardSize);
      for (var j = 0; j < boardSize; j++) {
        state[j] = stateData[j] * snap.player;
        mask[j] = stateData[j] === 0 ? 1 : 0;
      }

      // Download actual action index from the actions tensor
      var actionData = snap.actions.dataSync();
      var action = actionData[gameIndex];

      trajectory.push({
        state: state,
        action: action,
        player: snap.player,
        mask: mask
      });
    }

    return trajectory;
  }

  _pruneSnapshots() {
    // Find the minimum generation still active
    var minGen = Infinity;
    for (var i = 0; i < this.numGames; i++) {
      if (!this.done[i] && this._gameGenerations[i] < minGen) {
        minGen = this._gameGenerations[i];
      }
    }

    // Remove snapshots that no active game references
    var kept = [];
    for (var s = 0; s < this._stepSnapshots.length; s++) {
      var snap = this._stepSnapshots[s];
      var anyActive = false;
      for (var i = 0; i < this.numGames; i++) {
        if (!this.done[i] && snap.generations[i] === this._gameGenerations[i]) {
          anyActive = true;
          break;
        }
      }
      if (anyActive) {
        kept.push(snap);
      } else {
        snap.state.dispose();
        if (snap.actions) snap.actions.dispose();
      }
    }
    this._stepSnapshots = kept;
  }

  // ── Game lifecycle ──

  _finishGames(indices) {
    // Download winner info from GPU state
    var stateData = this.state.dataSync();

    for (var k = 0; k < indices.length; k++) {
      var i = indices[k];
      if (this.done[i]) continue;

      this.done[i] = 1;
      this.doneTimes[i] = Date.now();

      // Count cells to determine winner
      var p1 = 0, p2 = 0;
      var offset = i * this.boardSize;
      for (var j = 0; j < this.boardSize; j++) {
        var v = stateData[offset + j];
        if (v > 0) p1++;
        else if (v < 0) p2++;
      }
      var winner = p1 > p2 ? 1 : (p2 > p1 ? -1 : 0);
      this.winners[i] = winner;

      // Download trajectory and push to CPU buffer with real actions + masks.
      // Bug fix: previously hardcoded action: 0, corrupting REINFORCE gradient.
      var trajectory = this._downloadTrajectory(i);
      for (var t = 0; t < trajectory.length; t++) {
        var step = trajectory[t];
        var reward = (step.player === winner) ? 1 : (winner === 0 ? 0 : -1);
        this.buffer.push({ state: step.state, action: step.action, reward: reward, mask: step.mask });
      }
      if (this.buffer.length > this.maxBufferSize) {
        this.buffer = this.buffer.slice(-this.maxBufferSize);
      }

      // Stats
      this.gamesCompleted++;
      this.gamesSinceLastTrain++;
      if (winner === 1) this.p1Wins++;
      else if (winner === -1) this.p2Wins++;
      else this.draws++;
      this.recentLengths.push(this.turns[i]);
      if (this.recentLengths.length > 100) this.recentLengths.shift();
    }
  }

  _resetGames(indices) {
    if (indices.length === 0) return;

    // Build a new state tensor with reset games zeroed out
    var stateData = this.state.dataSync();
    var newData = Float32Array.from(stateData);

    for (var k = 0; k < indices.length; k++) {
      var i = indices[k];
      var offset = i * this.boardSize;
      for (var j = 0; j < this.boardSize; j++) {
        newData[offset + j] = 0;
      }
      this.done[i] = 0;
      this.winners[i] = 0;
      this.turns[i] = 0;
      this.doneTimes[i] = 0;
      this._gameGenerations[i] = this._nextGeneration++;
    }

    var oldState = this.state;
    this.state = tf.tensor2d(newData, [this.numGames, this.boardSize]);
    oldState.dispose();

    this._pruneSnapshots();
  }

  // ── Training: REINFORCE loss, same as reinforce.js ──

  _maybeTrainAndRestart() {
    if (this.gamesSinceLastTrain >= this.trainInterval &&
        this.buffer.length >= this.trainBatchSize) {
      this.lastLoss = this.train(this.trainBatchSize);
      this.generation++;
      this.gamesSinceLastTrain = 0;

      // Checkpoint pool: save weights periodically, run async Elo eval
      if (this.checkpointPool) {
        if (this.checkpointPool.shouldSave(this.generation)) {
          this.checkpointPool.save(this.model.model, this.generation);
        }
        if (this.checkpointPool.hasCheckpoints() && this.generation % 10 === 0) {
          this._asyncEloUpdate();
        }
      }
    }

    // Immediate restart — no dead-slot delay (was 300ms, bottleneck at scale).
    var toRestart = [];
    for (var i = 0; i < this.numGames; i++) {
      if (this.done[i]) {
        toRestart.push(i);
      }
    }
    if (toRestart.length > 0) {
      this._resetGames(toRestart);
    }
  }

  train(batchSize) {
    if (this.buffer.length === 0) return 0;

    var batch = this.buffer.splice(0, batchSize);
    var boardSize = this.model.boardSize;
    var n = batch.length;

    var statesArr = [];
    var actionsArr = [];
    var rewardsArr = [];
    var masksArr = [];
    for (var i = 0; i < n; i++) {
      statesArr.push(batch[i].state);
      actionsArr.push(batch[i].action);
      rewardsArr.push(batch[i].reward);
      masksArr.push(batch[i].mask);
    }

    var statesTensor = tf.tensor2d(flattenStates(statesArr, boardSize), [n, boardSize]);

    var actionMaskData = new Float32Array(n * boardSize);
    for (var i = 0; i < n; i++) {
      actionMaskData[i * boardSize + actionsArr[i]] = 1;
    }
    var actionMaskTensor = tf.tensor2d(actionMaskData, [n, boardSize]);
    var rewardsTensor = tf.tensor1d(rewardsArr);

    // Build valid-move mask tensor: invalid moves get logit = -1e9 before softmax.
    // Matches inference path (which uses maskedSoftmax). Previously missing,
    // causing train/inference mismatch that pushed probability onto occupied cells.
    var validMaskData = new Float32Array(n * boardSize);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < boardSize; j++) {
        validMaskData[i * boardSize + j] = masksArr[i][j];
      }
    }
    var validMaskTensor = tf.tensor2d(validMaskData, [n, boardSize]);

    var lossVal = 0;
    var self = this;
    try {
      var loss = self.optimizer.minimize(function () {
        var combined = self.model.model.predict(statesTensor);
        var policyLogits = combined.slice([0, 0], [-1, boardSize]);
        // Mask invalid moves before softmax so training matches inference
        var maskedLogits = policyLogits.add(validMaskTensor.sub(1).mul(1e9));
        var preds = maskedLogits.softmax();
        var selectedProbs = preds.mul(actionMaskTensor).sum(1);
        var logProbs = selectedProbs.add(tf.scalar(1e-8)).log();
        var policyLoss = logProbs.mul(rewardsTensor).mean().neg();
        var entropy = preds.add(tf.scalar(1e-8)).log().mul(preds).sum(1).mean().neg();
        return policyLoss.sub(entropy.mul(tf.scalar(0.01)));
      }, true);

      if (loss) {
        lossVal = loss.dataSync()[0];
        loss.dispose();
      }
    } catch (e) {
      console.warn('GPU trainer training error:', e.message);
    }

    statesTensor.dispose();
    actionMaskTensor.dispose();
    rewardsTensor.dispose();
    validMaskTensor.dispose();
    self.trainSteps++;
    return lossVal;
  }

  // ── UI interface: getBoardsForRender ──
  // Called once per animation frame (not per tick).
  // 16KB at 60fps = 960KB/s — negligible.

  getBoardsForRender() {
    var stateData = this.state.dataSync();
    return {
      boards: stateData,
      done: this.done,
      winners: this.winners
    };
  }

  // ── Human play: CPU path, same as Reinforce.selectAction ──

  selectAction(state, mask) {
    var boardSize = this.model.boardSize;
    var statesTensor = tf.tensor2d(state, [1, boardSize]);
    var out = this.model.forward(statesTensor);
    var logitsData = out.policy.dataSync();
    out.policy.dispose();
    out.value.dispose();
    statesTensor.dispose();

    var logits = new Float32Array(boardSize);
    for (var j = 0; j < boardSize; j++) logits[j] = logitsData[j];
    var probs = maskedSoftmax(logits, mask);
    return sampleFromProbs(probs);
  }

  // ── Algorithm interface compatibility ──

  shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize) {
    return gamesSinceLastTrain >= trainInterval && this.buffer.length >= trainBatchSize;
  }

  getBufferSize() {
    return this.buffer.length;
  }

  getTrainSteps() {
    return this.trainSteps;
  }

  // ── Async Elo: play 1 CPU-side game (current model vs checkpoint) via setTimeout ──
  // Runs off the GPU hot path so it doesn't block tick().
  // One game at a time (_eloUpdateInFlight gate) to avoid piling up work.

  _asyncEloUpdate() {
    if (this._eloUpdateInFlight) return;
    this._eloUpdateInFlight = true;
    var self = this;
    var pool = this.checkpointPool;

    pool.loadRandomOpponent();

    setTimeout(function () {
      try {
        var game = new Game(self.rows, self.cols, self.walls);
        var boardSize = self.boardSize;
        var maxTurns = boardSize * 2;

        for (var turn = 0; turn < maxTurns; turn++) {
          // P1 move (current model)
          var mask1 = game.getValidMovesMask();
          var hasValid1 = false;
          for (var j = 0; j < boardSize; j++) { if (mask1[j] > 0) { hasValid1 = true; break; } }
          if (!hasValid1) break;
          var state1 = game.getBoardForNN(1);
          var action1 = self.selectAction(state1, mask1);
          game.makeMove(1, action1);

          // P2 move (checkpoint opponent)
          var mask2 = game.getValidMovesMask();
          var hasValid2 = false;
          for (var j = 0; j < boardSize; j++) { if (mask2[j] > 0) { hasValid2 = true; break; } }
          if (!hasValid2) break;
          var state2 = game.getBoardForNN(-1);
          var results2 = pool.selectActions([state2], [mask2]);
          game.makeMove(-1, results2[0].action);

          game.spreadPlague();
          if (game.isGameOver()) break;
        }

        var winner = game.getWinner();
        pool.updateElo(winner === 1, winner === 0);
      } catch (e) {
        console.warn('Async Elo update error:', e.message);
      }
      self._eloUpdateInFlight = false;
    }, 0);
  }

  // ── Stats interface (same shape as SelfPlayTrainer.getStats) ──

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
      bufferSize: this.buffer.length,
      elo: this.checkpointPool ? this.checkpointPool.getCurrentElo() : 0,
      checkpointWinRate: this.checkpointPool ? this.checkpointPool.getRecentWinRate() : 0
    };
  }

  // ── Cleanup ──

  dispose() {
    if (this.state) { this.state.dispose(); this.state = null; }
    if (this.neighborKernel) { this.neighborKernel.dispose(); this.neighborKernel = null; }
    for (var s = 0; s < this._stepSnapshots.length; s++) {
      this._stepSnapshots[s].state.dispose();
      if (this._stepSnapshots[s].actions) this._stepSnapshots[s].actions.dispose();
    }
    this._stepSnapshots = [];
    if (this.optimizer) { this.optimizer.dispose(); this.optimizer = null; }
  }
}
