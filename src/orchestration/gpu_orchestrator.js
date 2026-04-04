import * as tf from '@tensorflow/tfjs';
import { maskedSoftmax, sampleFromProbs } from '../action';
import { createGame } from '../game';
import { GPUGameEngine } from '../engine/gpu_game_engine';
import { AsyncJobQueue } from '../core/job_queue';

// GPUOrchestrator
// - Runs simulation on GPUGameEngine
// - Delegates action selection + training to pluggable algorithms
// - Uses async queues for training and maintenance jobs
// - Stores compact CPU-side snapshots (only acted slots) for trajectory reconstruction

export class GPUOrchestrator {
  constructor(model, algo, config) {
    this.model = model;
    this.algo = algo;

    this.numGames = config.numGames || 40;
    this.rows = config.rows || 10;
    this.cols = config.cols || 10;
    this.boardSize = this.rows * this.cols;
    this.trainBatchSize = config.trainBatchSize || 256;
    this.trainInterval = config.trainInterval || 20;
    this.gameType = config.gameType || 'plague_walls';
    this.checkpointPool = config.checkpointPool || null;
    this.enableAsyncEloEval = !!config.enableAsyncEloEval;

    var uim = config.uiSnapshotMaxGames;
    if (typeof uim === 'number' && uim > 0) {
      this.uiSnapshotMaxGames = Math.min(uim, this.numGames);
    } else {
      this.uiSnapshotMaxGames = Math.min(48, this.numGames);
    }
    this._boardRenderCache = new Float32Array(this.numGames * this.boardSize);
    this._gpuUiPrimed = false;
    this._boardSnapshotCounter = 0;

    this.engine = new GPUGameEngine({
      numGames: this.numGames,
      rows: this.rows,
      cols: this.cols,
      gameType: this.gameType
    });
    this.engine.seedInitialBoardsIfNeeded();

    this.trainQueue = new AsyncJobQueue(1);
    this.maintenanceQueue = new AsyncJobQueue(1);

    // Compact per-step snapshots:
    // { player, slotIds, states, actions, logProbs, values, generations }
    this._stepSnapshots = [];
    this._gameGenerations = new Int32Array(this.numGames);
    this._nextGeneration = 1;
    this._vsCheckpoint = new Uint8Array(this.numGames);
    this._slotCheckpointId = new Int32Array(this.numGames);
    for (var sid = 0; sid < this.numGames; sid++) this._slotCheckpointId[sid] = -1;
    this._disposed = false;

    // Stats
    this.gamesCompleted = 0;
    this.gamesSinceLastTrain = 0;
    this.generation = 0;
    this.lastLoss = 0;
    this.p1Wins = 0;
    this.p2Wins = 0;
    this.draws = 0;
    this.recentLengths = [];
    this.lastEntropy = 0;

    var initSlots = [];
    for (var i = 0; i < this.numGames; i++) initSlots.push(i);
    this._resetSlotMetadata(initSlots);
  }

  // --- Main loop ---

  tick() {
    if (this._disposed) return;

    var entropySum = 0;
    var entropyCount = 0;

    // P1 moves (always current algorithm)
    var p1Active = this.engine.getActiveSlots();
    var p1Batch = this._buildActionBatch(p1Active, 1);
    if (p1Batch.slotIds.length > 0) {
      var p1Sel = this._selectWithAlgorithm(p1Batch);
      this.engine.applyActions(1, p1Batch.slotIds, p1Sel.actionsBySlot);
      this._recordSnapshot(1, p1Batch, p1Sel);
      entropySum += p1Sel.entropySum;
      entropyCount += p1Sel.entropyCount;
    }

    // P2 moves (current algorithm for self-play slots, checkpoint model for league slots)
    var p2Active = this.engine.getActiveSlots();
    if (p2Active.length > 0) {
      var split = this._splitP2Slots(p2Active);
      var p2ActionsBySlot = new Int32Array(this.numGames);
      var p2ApplySlots = [];

      var selfBatch = this._buildActionBatch(split.selfSlots, -1);
      if (selfBatch.slotIds.length > 0) {
        var p2Sel = this._selectWithAlgorithm(selfBatch);
        for (var i = 0; i < selfBatch.slotIds.length; i++) {
          var slot = selfBatch.slotIds[i];
          p2ActionsBySlot[slot] = p2Sel.actionsBySlot[slot];
          p2ApplySlots.push(slot);
        }
        this._recordSnapshot(-1, selfBatch, p2Sel);
        entropySum += p2Sel.entropySum;
        entropyCount += p2Sel.entropyCount;
      }

      var ckptBatch = this._buildActionBatch(split.ckptSlots, -1);
      if (ckptBatch.slotIds.length > 0 && this.checkpointPool && this.checkpointPool.hasCheckpoints()) {
        var ckptSel = this._selectWithCheckpoint(ckptBatch);
        for (var j = 0; j < ckptSel.slotIds.length; j++) {
          var cslot = ckptSel.slotIds[j];
          p2ActionsBySlot[cslot] = ckptSel.actionsBySlot[cslot];
          p2ApplySlots.push(cslot);
        }
      }

      if (p2ApplySlots.length > 0) {
        this.engine.applyActions(-1, p2ApplySlots, p2ActionsBySlot);
      }
    }

    this.engine.spread();
    this.engine.incrementTurnsForActive();

    var finished = this.engine.resolveTerminals();
    if (finished.doneSlots.length > 0) {
      this._finishSlots(finished.doneSlots, finished.winners);
      this.engine.resetSlots(finished.doneSlots);
      this._resetSlotMetadata(finished.doneSlots);
      this._pruneSnapshots();
    }

    if (entropyCount > 0) {
      this.lastEntropy = entropySum / entropyCount;
      if (this.algo && this.algo.algo) this.algo.algo.lastEntropy = this.lastEntropy;
    }

    this._maybeScheduleTrain();
  }

  // --- Action batching and selection ---

  _buildActionBatch(slotIds, player) {
    var out = { slotIds: [], states: [], masks: [] };
    if (!slotIds || slotIds.length === 0) return out;
    var cpu = this.engine.extractStatesMasksCPU(slotIds, player);
    for (var i = 0; i < slotIds.length; i++) {
      var mask = cpu.masks[i];
      var hasValid = false;
      for (var j = 0; j < mask.length; j++) {
        if (mask[j] > 0) { hasValid = true; break; }
      }
      if (!hasValid) continue;
      out.slotIds.push(slotIds[i]);
      out.states.push(cpu.states[i]);
      out.masks.push(mask);
    }
    return out;
  }

  _selectWithAlgorithm(batch) {
    var N = this.numGames;
    var actionsBySlot = new Int32Array(N);
    var k = batch.slotIds.length;
    var actions = new Int32Array(k);
    var logProbs = new Float32Array(k);
    var values = new Float32Array(k);
    for (var z = 0; z < k; z++) {
      logProbs[z] = NaN;
      values[z] = NaN;
    }
    var entropySum = 0;
    var entropyCount = k;

    var results = this.algo.selectActions(batch.states, batch.masks);
    for (var i = 0; i < k; i++) {
      var slot = batch.slotIds[i];
      var r = results && results[i] ? results[i] : null;
      var action = r && Number.isFinite(r.action) ? r.action : this._fallbackAction(batch.masks[i]);
      actionsBySlot[slot] = action;
      actions[i] = action;
      if (r && Number.isFinite(r.logProb)) logProbs[i] = r.logProb;
      if (r && Number.isFinite(r.value)) values[i] = r.value;
    }

    if (this.algo && this.algo.algo && Number.isFinite(this.algo.algo.lastEntropy)) {
      entropySum = this.algo.algo.lastEntropy * k;
    } else {
      entropySum = 0;
      entropyCount = 0;
    }

    return {
      slotIds: batch.slotIds,
      actionsBySlot: actionsBySlot,
      actions: actions,
      logProbs: logProbs,
      values: values,
      entropySum: entropySum,
      entropyCount: entropyCount
    };
  }

  _selectWithCheckpoint(batch) {
    var N = this.numGames;
    var actionsBySlot = new Int32Array(N);
    var grouped = {};

    for (var i = 0; i < batch.slotIds.length; i++) {
      var slot = batch.slotIds[i];
      var checkpointId = this._slotCheckpointId[slot];
      if (checkpointId < 0 || !this.checkpointPool || !this.checkpointPool.hasCheckpoints()) {
        checkpointId = this.checkpointPool ? this.checkpointPool.pickRandomOpponentId() : -1;
        this._slotCheckpointId[slot] = checkpointId;
      }
      if (checkpointId < 0) {
        actionsBySlot[slot] = this._fallbackAction(batch.masks[i]);
        continue;
      }
      var key = '' + checkpointId;
      if (!grouped[key]) grouped[key] = { checkpointId: checkpointId, slotIds: [], states: [], masks: [] };
      grouped[key].slotIds.push(slot);
      grouped[key].states.push(batch.states[i]);
      grouped[key].masks.push(batch.masks[i]);
    }

    var keys = Object.keys(grouped);
    var appliedSlots = [];
    for (var g = 0; g < keys.length; g++) {
      var grp = grouped[keys[g]];
      var loaded = this.checkpointPool.loadOpponentById(grp.checkpointId);
      if (loaded < 0) {
        for (var m = 0; m < grp.slotIds.length; m++) {
          var missingSlot = grp.slotIds[m];
          var replacementId = this.checkpointPool.pickRandomOpponentId();
          this._slotCheckpointId[missingSlot] = replacementId;
          actionsBySlot[missingSlot] = this._fallbackAction(grp.masks[m]);
          appliedSlots.push(missingSlot);
        }
        continue;
      }
      var rs = this.checkpointPool.selectActions(grp.states, grp.masks);
      for (var j = 0; j < grp.slotIds.length; j++) {
        var slot = grp.slotIds[j];
        var a = rs[j] && Number.isFinite(rs[j].action) ? rs[j].action : this._fallbackAction(grp.masks[j]);
        actionsBySlot[slot] = a;
        appliedSlots.push(slot);
      }
    }

    return { actionsBySlot: actionsBySlot, slotIds: appliedSlots };
  }

  _fallbackAction(mask) {
    for (var i = 0; i < mask.length; i++) {
      if (mask[i] > 0) return i;
    }
    return 0;
  }

  _splitP2Slots(activeSlots) {
    var selfSlots = [];
    var ckptSlots = [];
    for (var i = 0; i < activeSlots.length; i++) {
      var slot = activeSlots[i];
      if (this._vsCheckpoint[slot]) ckptSlots.push(slot);
      else selfSlots.push(slot);
    }
    return { selfSlots: selfSlots, ckptSlots: ckptSlots };
  }

  _recordSnapshot(player, batch, selection) {
    var k = batch.slotIds.length;
    if (k === 0) return;
    var generations = new Int32Array(k);
    var slotIds = new Int32Array(k);
    var states = new Array(k);
    for (var i = 0; i < k; i++) {
      var slot = batch.slotIds[i];
      slotIds[i] = slot;
      generations[i] = this._gameGenerations[slot];
      states[i] = batch.states[i];
    }
    this._stepSnapshots.push({
      player: player,
      slotIds: slotIds,
      states: states,
      actions: selection.actions,
      logProbs: selection.logProbs,
      values: selection.values,
      generations: generations
    });
  }

  // --- Episode handling ---

  _finishSlots(doneSlots, winners) {
    for (var i = 0; i < doneSlots.length; i++) {
      var slot = doneSlots[i];
      var winner = winners[i];
      var vsCheckpoint = !!this._vsCheckpoint[slot];
      var trajectory = this._downloadTrajectory(slot, !vsCheckpoint);
      if (trajectory.length > 0) this.algo.onGameFinished(trajectory, winner);

      if (vsCheckpoint && this.checkpointPool) {
        var checkpointId = this._slotCheckpointId[slot];
        if (checkpointId >= 0) this.checkpointPool.updateEloForId(checkpointId, winner === 1, winner === 0);
        else this.checkpointPool.updateElo(winner === 1, winner === 0);
      }

      this.gamesCompleted++;
      this.gamesSinceLastTrain++;
      if (winner === 1) this.p1Wins++;
      else if (winner === -1) this.p2Wins++;
      else this.draws++;

      this.recentLengths.push(this.engine.turns[slot]);
      if (this.recentLengths.length > 100) this.recentLengths.shift();
    }
  }

  _downloadTrajectory(gameIndex, includeP2Moves) {
    var gen = this._gameGenerations[gameIndex];
    var B = this.boardSize;
    var trajectory = [];

    for (var s = 0; s < this._stepSnapshots.length; s++) {
      var snap = this._stepSnapshots[s];
      if (!includeP2Moves && snap.player === -1) continue;

      var found = -1;
      for (var i = 0; i < snap.slotIds.length; i++) {
        if (snap.slotIds[i] === gameIndex && snap.generations[i] === gen) {
          found = i;
          break;
        }
      }
      if (found < 0) continue;

      var raw = snap.states[found];
      var state = new Float32Array(B);
      var mask = new Float32Array(B);
      for (var j = 0; j < B; j++) {
        state[j] = raw[j];
        mask[j] = raw[j] === 0 ? 1 : 0;
      }

      var step = {
        state: state,
        action: snap.actions[found],
        player: snap.player,
        mask: mask
      };
      var lp = snap.logProbs[found];
      var vv = snap.values[found];
      if (Number.isFinite(lp)) step.logProb = lp;
      if (Number.isFinite(vv)) step.value = vv;
      trajectory.push(step);
    }
    return trajectory;
  }

  _resetSlotMetadata(indices) {
    for (var i = 0; i < indices.length; i++) {
      var slot = indices[i];
      this._gameGenerations[slot] = this._nextGeneration++;

      var vsCheckpoint = this._shouldUseCheckpointForSlot();
      var checkpointId = -1;
      if (vsCheckpoint && this.checkpointPool && this.checkpointPool.hasCheckpoints()) {
        checkpointId = this.checkpointPool.pickRandomOpponentId();
        if (checkpointId < 0) vsCheckpoint = false;
      } else {
        vsCheckpoint = false;
      }

      this._vsCheckpoint[slot] = vsCheckpoint ? 1 : 0;
      this._slotCheckpointId[slot] = checkpointId;
    }
  }

  _pruneSnapshots() {
    var kept = [];
    for (var s = 0; s < this._stepSnapshots.length; s++) {
      var snap = this._stepSnapshots[s];
      var anyActive = false;
      for (var i = 0; i < snap.slotIds.length; i++) {
        var slot = snap.slotIds[i];
        if (!this.engine.done[slot] && snap.generations[i] === this._gameGenerations[slot]) {
          anyActive = true;
          break;
        }
      }
      if (anyActive) kept.push(snap);
    }
    this._stepSnapshots = kept;
  }

  // --- Training / async jobs ---

  _maybeScheduleTrain() {
    if (this._disposed) return;
    if (this.trainQueue.hasKey('train')) return;
    if (!this.algo.shouldTrain(this.gamesSinceLastTrain, this.trainInterval, this.trainBatchSize)) return;
    var self = this;
    this.trainQueue.enqueue('train', function () {
      if (self._disposed) return;
      self.lastLoss = self.algo.train(self.trainBatchSize);
      self.generation++;
      // Consume one interval of game credit per train step so overflow
      // accumulated during asynchronous training is not discarded.
      var consumedGames = Math.max(1, self.trainInterval || 1);
      self.gamesSinceLastTrain = Math.max(0, self.gamesSinceLastTrain - consumedGames);

      if (self.checkpointPool) {
        if (self.checkpointPool.shouldSave(self.generation)) {
          self.checkpointPool.save(self.model.model, self.generation);
        }
        if (self.enableAsyncEloEval && self.generation % 10 === 0) {
          self.maintenanceQueue.enqueue('elo', function () {
            if (!self._disposed) self._runAsyncEloUpdate();
          });
        }
      }
    });
  }

  _runAsyncEloUpdate() {
    if (!this.checkpointPool || !this.checkpointPool.hasCheckpoints()) return;
    var checkpointId = this.checkpointPool.pickRandomOpponentId();
    if (checkpointId < 0) return;
    if (this.checkpointPool.loadOpponentById(checkpointId) < 0) return;

    try {
      var game = createGame(this.gameType, this.rows, this.cols);
      var B = this.boardSize;
      var maxTurns = B * 2;
      for (var turn = 0; turn < maxTurns; turn++) {
        var mask1 = game.getValidMovesMask();
        var has1 = false;
        for (var j = 0; j < B; j++) { if (mask1[j] > 0) { has1 = true; break; } }
        if (!has1) break;
        var state1 = game.getBoardForNN(1);
        var action1 = this.selectAction(state1, mask1);
        game.makeMove(1, action1);

        var mask2 = game.getValidMovesMask();
        var has2 = false;
        for (var j = 0; j < B; j++) { if (mask2[j] > 0) { has2 = true; break; } }
        if (!has2) break;
        var state2 = game.getBoardForNN(-1);
        var r2 = this.checkpointPool.selectActions([state2], [mask2]);
        game.makeMove(-1, r2[0].action);

        game.spreadPlague();
        if (game.isGameOver()) break;
      }
      var winner = game.getWinner();
      this.checkpointPool.updateEloForId(checkpointId, winner === 1, winner === 0);
    } catch (e) {
      console.warn('GPUOrchestrator Elo update error:', e.message);
    }
  }

  _shouldUseCheckpointForSlot() {
    return !!(this.checkpointPool && this.checkpointPool.shouldBeCheckpointGame());
  }

  // --- UI compatibility ---

  selectAction(state, mask) {
    if (this.algo && typeof this.algo.selectAction === 'function') {
      return this.algo.selectAction(state, mask);
    }

    // Fallback to direct model sampling.
    var B = this.model.boardSize;
    var statesTensor = tf.tensor2d(state, [1, B]);
    var out = this.model.forward(statesTensor);
    var logitsData = out.policy.dataSync();
    out.policy.dispose();
    out.value.dispose();
    statesTensor.dispose();
    var logits = new Float32Array(B);
    for (var j = 0; j < B; j++) logits[j] = logitsData[j];
    var probs = maskedSoftmax(logits, mask);
    return sampleFromProbs(probs);
  }

  _uiBoardSnapshotSlots(N, maxG, counter) {
    var m = Math.min(maxG, N);
    var pageCount = Math.ceil(N / m);
    var page = counter % pageCount;
    var start = page * m;
    var out = [];
    for (var j = 0; j < m && start + j < N; j++) out.push(start + j);
    return out;
  }

  getBoardsForRender() {
    var N = this.numGames;
    var B = this.boardSize;
    var maxG = this.uiSnapshotMaxGames;
    if (maxG >= N) {
      var full = this.engine.getBoardsForRender();
      this._boardRenderCache.set(full.boards);
      this._gpuUiPrimed = true;
      return { boards: this._boardRenderCache, done: full.done, winners: full.winners };
    }
    if (!this._gpuUiPrimed) {
      var fr = this.engine.getBoardsForRender();
      this._boardRenderCache.set(fr.boards);
      this._gpuUiPrimed = true;
      return { boards: this._boardRenderCache, done: fr.done, winners: fr.winners };
    }
    this._boardSnapshotCounter++;
    var slotIds = this._uiBoardSnapshotSlots(N, maxG, this._boardSnapshotCounter);
    var part = this.engine.getBoardsForRenderGather(slotIds);
    for (var i = 0; i < slotIds.length; i++) {
      var slot = slotIds[i];
      this._boardRenderCache.set(part.boards.subarray(i * B, (i + 1) * B), slot * B);
    }
    return { boards: this._boardRenderCache, done: part.done, winners: part.winners };
  }

  shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize) {
    return this.algo.shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize);
  }

  getBufferSize() {
    return this.algo.getBufferSize();
  }

  getTrainSteps() {
    return this.algo.getTrainSteps();
  }

  getStats() {
    var avgLen = 0;
    if (this.recentLengths.length > 0) {
      var sum = 0;
      for (var i = 0; i < this.recentLengths.length; i++) sum += this.recentLengths[i];
      avgLen = sum / this.recentLengths.length;
    }
    var ent = this.algo && this.algo.algo && typeof this.algo.algo.lastEntropy === 'number'
      ? this.algo.algo.lastEntropy
      : this.lastEntropy;
    return {
      gamesCompleted: this.gamesCompleted,
      generation: this.generation,
      loss: this.lastLoss,
      p1Wins: this.p1Wins,
      p2Wins: this.p2Wins,
      draws: this.draws,
      avgGameLength: avgLen,
      bufferSize: this.getBufferSize(),
      elo: this.checkpointPool ? this.checkpointPool.getCurrentElo() : 0,
      checkpointWinRate: this.checkpointPool ? this.checkpointPool.getRecentWinRate() : 0,
      entropy: Number.isFinite(ent) ? ent : 0
    };
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    this._stepSnapshots = [];
    if (this.trainQueue) this.trainQueue.close();
    if (this.maintenanceQueue) this.maintenanceQueue.close();
    if (this.engine) {
      this.engine.dispose();
      this.engine = null;
    }
    if (this.algo && typeof this.algo.dispose === 'function') this.algo.dispose();
  }
}
