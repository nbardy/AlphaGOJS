import { createModel } from '../../model_registry';
import { createAlgorithm } from '../../algo_registry';
import { CheckpointPool } from '../../checkpoint_pool';
import { GPUGameEngine } from '../../engine/gpu_game_engine';
import { AsyncJobQueue } from '../../core/job_queue';
import { MSG } from '../protocol/messages';

function defaultCheckpointConfig(cfg) {
  cfg = cfg || {};
  return {
    maxCheckpoints: cfg.maxCheckpoints || 50,
    recentWindow: cfg.recentWindow || 50,
    sampleMode: cfg.sampleMode || 'uniform_recent',
    checkpointFraction: typeof cfg.checkpointFraction === 'number' ? cfg.checkpointFraction : 0.25,
    saveInterval: cfg.saveInterval || 20
  };
}

export class GPUOwnerRuntime {
  constructor(postFn) {
    this._post = postFn;
    this._disposed = false;
    this._ready = false;
    this._ticks = 0;
    this._snapshotEveryTicks = 2;

    this.model = null;
    this.algo = null;
    this.pool = null;
    this.engine = null;
    this.trainQueue = null;

    this.numGames = 0;
    this.rows = 0;
    this.cols = 0;
    this.boardSize = 0;
    this.trainBatchSize = 0;
    this.trainInterval = 0;
    this.pauseTicksWhenTraining = false;
    this.gameType = 'plague_walls';

    this._stepSnapshots = [];
    this._gameGenerations = null;
    this._nextGeneration = 1;
    this._vsCheckpoint = null;
    this._slotCheckpointId = null;

    this.gamesCompleted = 0;
    this.gamesSinceLastTrain = 0;
    this.generation = 0;
    this.lastLoss = 0;
    this.p1Wins = 0;
    this.p2Wins = 0;
    this.draws = 0;
    this.recentLengths = [];
    this.lastEntropy = 0;
    this._trainInFlight = false;
  }

  async init(config) {
    config = config || {};
    await this.dispose();

    this._disposed = false;
    this._ready = false;
    this._ticks = 0;
    this._snapshotEveryTicks = Math.max(1, config.snapshotEveryTicks || 2);

    this.rows = config.rows || 20;
    this.cols = config.cols || 20;
    this.numGames = config.numGames || 80;
    this.boardSize = this.rows * this.cols;
    this.trainBatchSize = config.trainBatchSize || 512;
    this.trainInterval = config.trainInterval || 30;
    this.pauseTicksWhenTraining = !!config.pauseTicksWhenTraining;
    this.gameType = config.gameType || 'plague_walls';

    var modelType = config.modelType || 'dense';
    var algoType = config.algoType || 'ppo';
    this.model = createModel(modelType, this.rows, this.cols);
    this.algo = createAlgorithm(algoType, this.model);

    var poolCfg = defaultCheckpointConfig(config.checkpointPool);
    this.pool = new CheckpointPool(function () {
      return createModel(modelType, config.rows || 20, config.cols || 20);
    }, poolCfg);

    this.engine = new GPUGameEngine({
      numGames: this.numGames,
      rows: this.rows,
      cols: this.cols,
      gameType: this.gameType
    });
    this.trainQueue = new AsyncJobQueue(1);

    this._stepSnapshots = [];
    this._gameGenerations = new Int32Array(this.numGames);
    this._nextGeneration = 1;
    this._vsCheckpoint = new Uint8Array(this.numGames);
    this._slotCheckpointId = new Int32Array(this.numGames);
    for (var i = 0; i < this.numGames; i++) this._slotCheckpointId[i] = -1;

    this.gamesCompleted = 0;
    this.gamesSinceLastTrain = 0;
    this.generation = 0;
    this.lastLoss = 0;
    this.p1Wins = 0;
    this.p2Wins = 0;
    this.draws = 0;
    this.recentLengths = [];
    this.lastEntropy = 0;
    this._trainInFlight = false;

    var initSlots = [];
    for (var s = 0; s < this.numGames; s++) initSlots.push(s);
    this._resetSlotMetadata(initSlots);

    this._ready = true;
    this._post({ type: MSG.READY });
  }

  async restart(config) {
    return this.init(config);
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

  _buildActionBatch(slotIds, player) {
    var out = { slotIds: [], states: [], masks: [] };
    if (!slotIds || slotIds.length === 0) return out;
    var cpu = this.engine.extractStatesMasksCPU(slotIds, player);
    for (var i = 0; i < slotIds.length; i++) {
      var mask = cpu.masks[i];
      var hasValid = false;
      for (var j = 0; j < mask.length; j++) {
        if (mask[j] > 0) {
          hasValid = true;
          break;
        }
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

    var entropySum = 0;
    var entropyCount = 0;
    if (this.algo && Number.isFinite(this.algo.lastEntropy)) {
      entropySum = this.algo.lastEntropy * k;
      entropyCount = k;
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
      if (checkpointId < 0 || !this.pool || !this.pool.hasCheckpoints()) {
        checkpointId = this.pool ? this.pool.pickRandomOpponentId() : -1;
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
      var loaded = this.pool.loadOpponentById(grp.checkpointId);
      if (loaded < 0) {
        for (var m = 0; m < grp.slotIds.length; m++) {
          var missingSlot = grp.slotIds[m];
          var replacementId = this.pool.pickRandomOpponentId();
          this._slotCheckpointId[missingSlot] = replacementId;
          actionsBySlot[missingSlot] = this._fallbackAction(grp.masks[m]);
          appliedSlots.push(missingSlot);
        }
        continue;
      }

      var rs = this.pool.selectActions(grp.states, grp.masks);
      for (var j = 0; j < grp.slotIds.length; j++) {
        var slot = grp.slotIds[j];
        var a = rs[j] && Number.isFinite(rs[j].action) ? rs[j].action : this._fallbackAction(grp.masks[j]);
        actionsBySlot[slot] = a;
        appliedSlots.push(slot);
      }
    }

    return { actionsBySlot: actionsBySlot, slotIds: appliedSlots };
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

  _finishSlots(doneSlots, winners) {
    for (var i = 0; i < doneSlots.length; i++) {
      var slot = doneSlots[i];
      var winner = winners[i];
      var vsCheckpoint = !!this._vsCheckpoint[slot];
      var trajectory = this._downloadTrajectory(slot, !vsCheckpoint);
      if (trajectory.length > 0) this.algo.onGameFinished(trajectory, winner);

      if (vsCheckpoint && this.pool) {
        var checkpointId = this._slotCheckpointId[slot];
        if (checkpointId >= 0) this.pool.updateEloForId(checkpointId, winner === 1, winner === 0);
        else this.pool.updateElo(winner === 1, winner === 0);
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

  _shouldUseCheckpointForSlot() {
    return !!(this.pool && this.pool.shouldBeCheckpointGame());
  }

  _resetSlotMetadata(indices) {
    for (var i = 0; i < indices.length; i++) {
      var slot = indices[i];
      this._gameGenerations[slot] = this._nextGeneration++;

      var vsCheckpoint = this._shouldUseCheckpointForSlot();
      var checkpointId = -1;
      if (vsCheckpoint && this.pool && this.pool.hasCheckpoints()) {
        checkpointId = this.pool.pickRandomOpponentId();
        if (checkpointId < 0) vsCheckpoint = false;
      } else {
        vsCheckpoint = false;
      }

      this._vsCheckpoint[slot] = vsCheckpoint ? 1 : 0;
      this._slotCheckpointId[slot] = checkpointId;
    }
  }

  _maybeScheduleTrain() {
    if (this._disposed) return;
    if (!this.trainQueue || this.trainQueue.hasKey('train')) return;
    if (!this.algo.shouldTrain(this.gamesSinceLastTrain, this.trainInterval, this.trainBatchSize)) return;

    var self = this;
    this.trainQueue.enqueue('train', function () {
      if (self._disposed || !self.algo) return;
      self._trainInFlight = true;
      try {
        self.lastLoss = self.algo.train(self.trainBatchSize);
        self.generation++;
        // Consume exactly one train interval of game credit per train step.
        // This preserves overflow accrued while a train job was in flight.
        var consumedGames = Math.max(1, self.trainInterval || 1);
        self.gamesSinceLastTrain = Math.max(0, self.gamesSinceLastTrain - consumedGames);

        if (self.pool && self.pool.shouldSave(self.generation)) {
          self.pool.save(self.model.model, self.generation);
        }
      } catch (e) {
        console.warn('GPUOwnerRuntime train error:', e.message);
      } finally {
        self._trainInFlight = false;
      }
    });
  }

  _tickOne() {
    var entropySum = 0;
    var entropyCount = 0;

    // P1 moves (current algorithm)
    var p1Active = this.engine.getActiveSlots();
    var p1Batch = this._buildActionBatch(p1Active, 1);
    if (p1Batch.slotIds.length > 0) {
      var p1Sel = this._selectWithAlgorithm(p1Batch);
      this.engine.applyActions(1, p1Batch.slotIds, p1Sel.actionsBySlot);
      this._recordSnapshot(1, p1Batch, p1Sel);
      entropySum += p1Sel.entropySum;
      entropyCount += p1Sel.entropyCount;
    }

    // P2 moves (self-play + checkpoint slots)
    var p2Active = this.engine.getActiveSlots();
    if (p2Active.length > 0) {
      var p2ActionsBySlot = new Int32Array(this.numGames);
      var p2ApplySlots = [];
      var p2CPU = this.engine.extractStatesMasksCPU(p2Active, -1);
      var selfBatch = { slotIds: [], states: [], masks: [] };
      var ckptBatch = { slotIds: [], states: [], masks: [] };

      for (var bi = 0; bi < p2Active.length; bi++) {
        var slotForP2 = p2Active[bi];
        var maskForP2 = p2CPU.masks[bi];
        var hasValidP2 = false;
        for (var mv = 0; mv < maskForP2.length; mv++) {
          if (maskForP2[mv] > 0) {
            hasValidP2 = true;
            break;
          }
        }
        if (!hasValidP2) continue;

        if (this._vsCheckpoint[slotForP2]) {
          ckptBatch.slotIds.push(slotForP2);
          ckptBatch.states.push(p2CPU.states[bi]);
          ckptBatch.masks.push(maskForP2);
        } else {
          selfBatch.slotIds.push(slotForP2);
          selfBatch.states.push(p2CPU.states[bi]);
          selfBatch.masks.push(maskForP2);
        }
      }

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

      if (ckptBatch.slotIds.length > 0 && this.pool && this.pool.hasCheckpoints()) {
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
      if (this.algo) this.algo.lastEntropy = this.lastEntropy;
    }

    this._maybeScheduleTrain();
  }

  _collectStats() {
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
      bufferSize: this.algo && this.algo.getBufferSize ? this.algo.getBufferSize() : 0,
      trainSteps: this.algo && this.algo.getTrainSteps ? this.algo.getTrainSteps() : 0,
      elo: this.pool ? this.pool.getCurrentElo() : 0,
      checkpointWinRate: this.pool ? this.pool.getRecentWinRate() : 0,
      entropy: this.lastEntropy,
      trainInFlight: !!this._trainInFlight
    };
  }

  _emitTickResult() {
    var payload = {
      type: MSG.TICK_RESULT,
      stats: this._collectStats()
    };

    if (this._ticks % this._snapshotEveryTicks === 0) {
      var boards = this.engine.getBoardsForRender();
      var srcBoards = boards.boards;
      var packedBoards = new Int8Array(srcBoards.length);
      for (var i = 0; i < srcBoards.length; i++) {
        packedBoards[i] = srcBoards[i];
      }
      var done = new Uint8Array(boards.done.length);
      done.set(boards.done);
      var winners = new Int8Array(boards.winners.length);
      winners.set(boards.winners);
      payload.boards = packedBoards;
      payload.done = done;
      payload.winners = winners;
      this._post(payload, [payload.boards.buffer, payload.done.buffer, payload.winners.buffer]);
      return;
    }

    this._post(payload);
  }

  async tick(steps) {
    if (!this._ready || this._disposed || !this.engine) return;
    if (this.pauseTicksWhenTraining && this._trainInFlight) {
      this._emitTickResult();
      return;
    }
    var n = Math.max(1, steps || 1);
    for (var i = 0; i < n; i++) {
      this._tickOne();
      this._ticks++;
    }
    this._emitTickResult();
  }

  async inferAction(requestId, state, mask) {
    var action = 0;
    try {
      if (this.algo && typeof this.algo.selectAction === 'function') {
        action = this.algo.selectAction(state, mask);
      }
    } catch (e) {
      action = 0;
    }
    this._post({
      type: MSG.INFER_ACTION_RESULT,
      requestId: requestId,
      action: action
    });
  }

  async dispose() {
    if (this.trainQueue) {
      try { this.trainQueue.close(); } catch (e) {}
    }
    this.trainQueue = null;

    if (this.engine) {
      try { this.engine.dispose(); } catch (e) {}
    }
    this.engine = null;

    if (this.algo && typeof this.algo.dispose === 'function') {
      try { this.algo.dispose(); } catch (e) {}
    }
    this.algo = null;
    this.model = null;
    this.pool = null;

    this._stepSnapshots = [];
    this._gameGenerations = null;
    this._vsCheckpoint = null;
    this._slotCheckpointId = null;
    this._trainInFlight = false;

    this._ready = false;
    this._disposed = true;
  }
}
