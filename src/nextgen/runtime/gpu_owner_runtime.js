import * as tf from '@tensorflow/tfjs';
import { flattenStates } from '../../action';
import { createModel } from '../../model_registry';
import { createAlgorithm } from '../../algo_registry';
import { eloUpdatePair } from '../../league_elo';
import { CheckpointPool } from '../../checkpoint_pool';
import { GPUGameEngine } from '../../engine/gpu_game_engine';
import { PLAGUE_WALL_CELL } from '../../engine/plague_walls_layout';
import { AsyncJobQueue } from '../../core/job_queue';
import { MSG } from '../protocol/messages';

function defaultCheckpointConfig(cfg) {
  cfg = cfg || {};
  return {
    maxCheckpoints: cfg.maxCheckpoints || 50,
    recentWindow: cfg.recentWindow || 50,
    sampleMode: cfg.sampleMode || 'uniform_recent',
    checkpointFraction: typeof cfg.checkpointFraction === 'number' ? cfg.checkpointFraction : 0.3,
    saveInterval: cfg.saveInterval || 15
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
    this.models = null;
    this.algos = null;
    this.pools = null;
    this._multiModel = false;
    this._numModels = 1;
    this._modelTypeIds = null;
    this._slotModelIndex = null;
    this.lastLossByModel = null;
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
    this.leagueElo = null;
    this._p2Kind = null;
    this._p2OppArchIdx = null;
    this._p2CheckpointPoolIdx = null;

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

    var algoType = config.algoType || 'ppo';
    this._algoKind = algoType;
    this._useGpuBatchedPolicySelect = config.useGpuBatchedPolicySelect !== false;

    var poolCfg = defaultCheckpointConfig(config.checkpointPool);
    var modelTypes = config.modelTypes;
    var useMulti =
      !!config.multiModel &&
      Array.isArray(modelTypes) &&
      modelTypes.length > 1;
    this._multiModel = useMulti;
    this.models = null;
    this.algos = null;
    this.pools = null;
    this.lastLossByModel = null;

    if (useMulti) {
      this._modelTypeIds = modelTypes.slice();
      this._numModels = modelTypes.length;
      this.models = [];
      this.algos = [];
      this.pools = [];
      this.lastLossByModel = new Float32Array(this._numModels);
      var rowsR = this.rows;
      var colsR = this.cols;
      var selfInit = this;
      for (var mi = 0; mi < this._numModels; mi++) {
        (function (tid) {
          var m = createModel(tid, rowsR, colsR);
          selfInit.models.push(m);
          selfInit.algos.push(createAlgorithm(algoType, m));
          selfInit.pools.push(
            new CheckpointPool(function () {
              return createModel(tid, rowsR, colsR);
            }, poolCfg)
          );
        })(modelTypes[mi]);
      }
      this.model = this.models[0];
      this.algo = this.algos[0];
      this.pool = this.pools[0];
      this.gamesSinceLastTrain = new Int32Array(this._numModels);
      this.generation = new Int32Array(this._numModels);
    } else {
      this._modelTypeIds = null;
      this._numModels = 1;
      var modelType = config.modelType || 'spatial_lite';
      this.model = createModel(modelType, this.rows, this.cols);
      this.algo = createAlgorithm(algoType, this.model);
      this.pool = new CheckpointPool(function () {
        return createModel(modelType, config.rows || 20, config.cols || 20);
      }, poolCfg);
      this.gamesSinceLastTrain = 0;
      this.generation = 0;
    }

    this.engine = new GPUGameEngine({
      numGames: this.numGames,
      rows: this.rows,
      cols: this.cols,
      gameType: this.gameType
    });
    this.engine.seedInitialBoardsIfNeeded();
    this.trainQueue = new AsyncJobQueue(1);

    this._stepSnapshots = [];
    this._gameGenerations = new Int32Array(this.numGames);
    this._nextGeneration = 1;
    this._vsCheckpoint = new Uint8Array(this.numGames);
    this._slotCheckpointId = new Int32Array(this.numGames);
    for (var i = 0; i < this.numGames; i++) this._slotCheckpointId[i] = -1;

    if (this._multiModel) {
      this._slotModelIndex = new Int32Array(this.numGames);
      for (var sx = 0; sx < this.numGames; sx++) {
        this._slotModelIndex[sx] = sx % this._numModels;
      }
      this.leagueElo = new Float32Array(this._numModels);
      for (var li = 0; li < this._numModels; li++) {
        this.leagueElo[li] = 1000;
        this.pools[li].currentElo = 1000;
      }
      this._p2Kind = new Uint8Array(this.numGames);
      this._p2OppArchIdx = new Int32Array(this.numGames);
      this._p2CheckpointPoolIdx = new Int32Array(this.numGames);
    } else {
      this._slotModelIndex = null;
      this.leagueElo = null;
      this._p2Kind = null;
      this._p2OppArchIdx = null;
      this._p2CheckpointPoolIdx = null;
    }

    this.gamesCompleted = 0;
    if (!this._multiModel) {
      this.gamesSinceLastTrain = 0;
      this.generation = 0;
    }
    this.lastLoss = 0;
    this.p1Wins = 0;
    this.p2Wins = 0;
    this.draws = 0;
    this.recentLengths = [];
    this.lastEntropy = 0;
    this._trainInFlight = false;

    var uim = config.uiSnapshotMaxGames;
    if (typeof uim === 'number' && uim > 0) {
      this._uiSnapshotMaxGames = Math.min(uim, this.numGames);
    } else {
      this._uiSnapshotMaxGames = Math.min(48, this.numGames);
    }
    this._hasDoneFullBoardSync = false;
    this._boardSnapshotPage = 0;

    this._benchLoopMode = (config.benchLoopMode === 'sim_random' || config.benchLoopMode === 'sim_forward')
      ? config.benchLoopMode
      : 'off';
    this._benchInstrument = !!config.benchInstrument;
    this._benchDisableTrain = this._benchLoopMode !== 'off';
    this._benchTrainMsPending = 0;
    this._benchTrainCallsPending = 0;
    if (config.benchMinimalUi) {
      this._snapshotEveryTicks = 1000;
    }
    this._benchPolicyMsBatch = 0;
    this._benchPhysicsMsBatch = 0;
    this._benchTickSamplesBatch = 0;

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

  _benchMinimalReplay() {
    return this._benchLoopMode === 'sim_random' || this._benchLoopMode === 'sim_forward';
  }

  _randomActionFromMask(mask) {
    var legals = [];
    for (var j = 0; j < mask.length; j++) {
      if (mask[j] > 0) legals.push(j);
    }
    if (legals.length === 0) return this._fallbackAction(mask);
    var seed = (this._ticks * 1103515245 + legals.length * 17 + 12345) >>> 0;
    seed = (1664525 * seed + 1013904223) >>> 0;
    return legals[Math.floor((seed / 4294967296) * legals.length)];
  }

  _selectRandomLegal(batch) {
    var N = this.numGames;
    var k = batch.slotIds.length;
    var actionsBySlot = new Int32Array(N);
    var actions = new Int32Array(k);
    var logProbs = new Float32Array(k);
    var values = new Float32Array(k);
    for (var z = 0; z < k; z++) {
      logProbs[z] = NaN;
      values[z] = NaN;
    }
    for (var i = 0; i < k; i++) {
      var slot = batch.slotIds[i];
      var a = this._randomActionFromMask(batch.masks[i]);
      actions[i] = a;
      actionsBySlot[slot] = a;
    }
    return {
      slotIds: batch.slotIds,
      actionsBySlot: actionsBySlot,
      actions: actions,
      logProbs: logProbs,
      values: values,
      entropySum: 0,
      entropyCount: 0
    };
  }

  _selectRandomCheckpointBatch(batch) {
    var N = this.numGames;
    var actionsBySlot = new Int32Array(N);
    var applied = [];
    for (var i = 0; i < batch.slotIds.length; i++) {
      var slot = batch.slotIds[i];
      var a = this._randomActionFromMask(batch.masks[i]);
      actionsBySlot[slot] = a;
      applied.push(slot);
    }
    return { actionsBySlot: actionsBySlot, slotIds: applied };
  }

  _splitP2Slots(activeSlots) {
    var selfSlots = [];
    var ckptSlots = [];
    var liveCrossSlots = [];
    if (!this._multiModel) {
      for (var i = 0; i < activeSlots.length; i++) {
        var slot = activeSlots[i];
        if (this._vsCheckpoint[slot]) ckptSlots.push(slot);
        else selfSlots.push(slot);
      }
      return { selfSlots: selfSlots, ckptSlots: ckptSlots, liveCrossSlots: liveCrossSlots };
    }
    for (var j = 0; j < activeSlots.length; j++) {
      var sl = activeSlots[j];
      if (this._vsCheckpoint[sl]) ckptSlots.push(sl);
      else if (this._p2Kind[sl] === 2) liveCrossSlots.push(sl);
      else selfSlots.push(sl);
    }
    return { selfSlots: selfSlots, ckptSlots: ckptSlots, liveCrossSlots: liveCrossSlots };
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

  /**
   * Self-play GPU path: gather rows on GPU, filter rows with no legal moves via a
   * small (k-float) readback. Caller passes batch to _selectWithAlgorithm; tensors
   * are disposed inside GPU batched tensor select (or on fallback error).
   */
  _buildActionBatchGpu(slotIds, player) {
    var empty = { slotIds: [], states: [], masks: [], _obsTensor: null, _maskTensor: null, _gpuPerspective: player };
    if (!slotIds || slotIds.length === 0) return empty;

    var raw = this.engine.gatherSlotsTensor(slotIds);
    var mask = raw.equal(0).cast('float32');
    var rowSums = mask.sum(1);
    this.engine._trackReadback(slotIds.length);
    var sumData = rowSums.dataSync();
    rowSums.dispose();

    var keepIdx = [];
    for (var i = 0; i < slotIds.length; i++) {
      if (sumData[i] > 0) keepIdx.push(i);
    }
    if (keepIdx.length === 0) {
      raw.dispose();
      mask.dispose();
      return empty;
    }

    var idxT = tf.tensor1d(keepIdx, 'int32');
    var rawK = tf.gather(raw, idxT, 0);
    var maskK = tf.gather(mask, idxT, 0);
    raw.dispose();
    mask.dispose();
    idxT.dispose();

    var pSc = tf.scalar(player);
    // Match getBoardForNN(plague_walls): wall = 0.5, else v * player.
    var obs = tf.tidy(function () {
      var isWall = rawK.equal(PLAGUE_WALL_CELL).cast('float32');
      var nonWall = tf.sub(1, isWall);
      var perspective = rawK.mul(pSc);
      return isWall.mul(0.5).add(nonWall.mul(perspective));
    });
    rawK.dispose();
    pSc.dispose();

    var out = {
      slotIds: keepIdx.map(function (j) { return slotIds[j]; }),
      states: [],
      masks: [],
      _obsTensor: obs,
      _maskTensor: maskK,
      _gpuPerspective: player
    };
    return out;
  }

  _policySelectUsesGpuBatchedPath() {
    if (!this._useGpuBatchedPolicySelect) return false;
    if (this._multiModel) {
      if (!this.models || this.models.length === 0) return false;
      for (var mi = 0; mi < this.models.length; mi++) {
        if (!this.models[mi] || typeof this.models[mi].forward !== 'function') return false;
      }
    } else if (!this.model || typeof this.model.forward !== 'function') {
      return false;
    }
    var k = this._algoKind;
    return k === 'ppo' || k === 'reinforce' || k === 'ppg';
  }

  _modelIdxForSlot(slot) {
    if (!this._multiModel || !this._slotModelIndex) return 0;
    return this._slotModelIndex[slot];
  }

  /**
   * Multi-model: split batch rows by slot model index, forward each subgroup on its own net.
   */
  _selectWithAlgorithmGpuBatchedMulti(batch) {
    var self = this;
    var k = batch.slotIds.length;
    var boardSize = this.boardSize;
    var N = this.numGames;
    var flatStates = flattenStates(batch.states, boardSize);
    var maskFlat = new Float32Array(k * boardSize);
    for (var i = 0; i < k; i++) {
      maskFlat.set(batch.masks[i], i * boardSize);
    }

    var byModel = [];
    for (var m = 0; m < this._numModels; m++) {
      byModel.push([]);
    }
    for (var ri = 0; ri < k; ri++) {
      var mi = this._modelIdxForSlot(batch.slotIds[ri]);
      byModel[mi].push(ri);
    }

    var actionsBySlot = new Int32Array(N);
    var actions = new Int32Array(k);
    var logProbs = new Float32Array(k);
    var values = new Float32Array(k);
    var entropySum = 0;
    var entropyCount = 0;

    for (var mm = 0; mm < this._numModels; mm++) {
      var idxList = byModel[mm];
      if (idxList.length === 0) continue;
      var km = idxList.length;
      var subFlat = new Float32Array(km * boardSize);
      var subMaskFlat = new Float32Array(km * boardSize);
      for (var j = 0; j < km; j++) {
        var row = idxList[j];
        var off = row * boardSize;
        subFlat.set(flatStates.subarray(off, off + boardSize), j * boardSize);
        subMaskFlat.set(maskFlat.subarray(off, off + boardSize), j * boardSize);
      }

      var entropyScalar = 0;
      var actionsKept;
      var logProbKept;
      var valueKept;
      var mod = this.models[mm];
      tf.tidy(function () {
        var statesT = tf.tensor2d(subFlat, [km, boardSize]);
        var fwd = mod.forward(statesT);
        var logits = fwd.policy;
        var vals = fwd.value;
        var maskT = tf.tensor2d(subMaskFlat, [km, boardSize]);
        var maskedLogits = logits.add(maskT.sub(1).mul(1e9));
        var probs = tf.softmax(maskedLogits);
        var entMean = probs.mul(tf.log(probs.add(1e-8))).sum(1).neg().mean();
        entropyScalar = entMean.dataSync()[0];

        var multinom = tf.multinomial(maskedLogits, 2);
        var actionsT = multinom.slice([0, 1], [-1, 1]).squeeze([1]);
        var logSM = tf.logSoftmax(maskedLogits);
        var oh = tf.oneHot(actionsT.cast('int32'), boardSize);
        var logProbT = logSM.mul(oh).sum(1);

        actionsKept = tf.keep(actionsT.clone());
        logProbKept = tf.keep(logProbT.clone());
        valueKept = tf.keep(vals.clone());
      });

      var ad = actionsKept.dataSync();
      var lp = logProbKept.dataSync();
      var vd = valueKept.dataSync();
      actionsKept.dispose();
      logProbKept.dispose();
      valueKept.dispose();

      for (var jj = 0; jj < km; jj++) {
        var origRow = idxList[jj];
        var slot = batch.slotIds[origRow];
        var ai = Math.round(ad[jj]);
        if (ai < 0 || ai >= boardSize || batch.masks[origRow][ai] <= 0) {
          ai = self._fallbackAction(batch.masks[origRow]);
        }
        actionsBySlot[slot] = ai;
        actions[origRow] = ai;
        logProbs[origRow] = lp[jj];
        values[origRow] = vd[jj];
      }
      entropySum += entropyScalar * km;
      entropyCount += km;
    }

    return {
      slotIds: batch.slotIds,
      actionsBySlot: actionsBySlot,
      actions: actions,
      logProbs: logProbs,
      values: values,
      entropySum: entropySum,
      entropyCount: entropyCount > 0 ? entropyCount : k
    };
  }

  /**
   * Single fixed model: batched policy forward + multinomial (PPO/REINFORCE/PPG GPU path).
   */
  _gpuBatchedOneModel(batch, model) {
    var self = this;
    var k = batch.slotIds.length;
    var boardSize = this.boardSize;
    var N = this.numGames;
    var flatStates = flattenStates(batch.states, boardSize);
    var maskFlat = new Float32Array(k * boardSize);
    for (var i = 0; i < k; i++) {
      maskFlat.set(batch.masks[i], i * boardSize);
    }

    var entropyScalar = 0;
    var actionsKept;
    var logProbKept;
    var valueKept;

    tf.tidy(function () {
      var statesT = tf.tensor2d(flatStates, [k, boardSize]);
      var fwd = model.forward(statesT);
      var logits = fwd.policy;
      var vals = fwd.value;
      var maskT = tf.tensor2d(maskFlat, [k, boardSize]);
      var maskedLogits = logits.add(maskT.sub(1).mul(1e9));
      var probs = tf.softmax(maskedLogits);
      var entMean = probs.mul(tf.log(probs.add(1e-8))).sum(1).neg().mean();
      entropyScalar = entMean.dataSync()[0];

      var multinom = tf.multinomial(maskedLogits, 2);
      var actionsT = multinom.slice([0, 1], [-1, 1]).squeeze([1]);
      var logSM = tf.logSoftmax(maskedLogits);
      var oh = tf.oneHot(actionsT.cast('int32'), boardSize);
      var logProbT = logSM.mul(oh).sum(1);

      actionsKept = tf.keep(actionsT.clone());
      logProbKept = tf.keep(logProbT.clone());
      valueKept = tf.keep(vals.clone());
    });

    var ad = actionsKept.dataSync();
    var lp = logProbKept.dataSync();
    var vd = valueKept.dataSync();
    actionsKept.dispose();
    logProbKept.dispose();
    valueKept.dispose();

    var actionsBySlot = new Int32Array(N);
    var actions = new Int32Array(k);
    var logProbs = new Float32Array(k);
    var values = new Float32Array(k);
    for (var ii = 0; ii < k; ii++) {
      var slot = batch.slotIds[ii];
      var ai = Math.round(ad[ii]);
      if (ai < 0 || ai >= boardSize || batch.masks[ii][ai] <= 0) {
        ai = self._fallbackAction(batch.masks[ii]);
      }
      actionsBySlot[slot] = ai;
      actions[ii] = ai;
      logProbs[ii] = lp[ii];
      values[ii] = vd[ii];
    }

    return {
      slotIds: batch.slotIds,
      actionsBySlot: actionsBySlot,
      actions: actions,
      logProbs: logProbs,
      values: values,
      entropySum: entropyScalar * k,
      entropyCount: k
    };
  }

  /**
   * Batched policy forward + tf.multinomial on GPU, readback only actions, log π(a|s), and V(s)
   * per row (see gpu_trainer.js). Matches training: masked logits = logits + (mask-1)*1e9, then
   * logSoftmax for log probs. PPO/REINFORCE/PPG only; other algos use algo.selectActions.
   */
  _selectWithAlgorithmGpuBatched(batch) {
    if (this._multiModel) {
      return this._selectWithAlgorithmGpuBatchedMulti(batch);
    }
    return this._gpuBatchedOneModel(batch, this.model);
  }

  /**
   * GPU tensor path for multiple models: gather rows per model, forward separately.
   */
  _selectWithAlgorithmGpuBatchedTensorsMulti(obsTensor, maskTensor, batch) {
    var self = this;
    var k = batch.slotIds.length;
    var boardSize = this.boardSize;
    var N = this.numGames;

    var byModel = [];
    for (var m = 0; m < this._numModels; m++) {
      byModel.push([]);
    }
    for (var ri = 0; ri < k; ri++) {
      byModel[this._modelIdxForSlot(batch.slotIds[ri])].push(ri);
    }

    var entropySum = 0;
    var entropyCount = 0;
    var actions = new Int32Array(k);
    var logProbs = new Float32Array(k);
    var values = new Float32Array(k);

    try {
      for (var mm = 0; mm < this._numModels; mm++) {
        var idxList = byModel[mm];
        if (idxList.length === 0) continue;
        var km = idxList.length;
        var idxT = tf.tensor1d(idxList, 'int32');
        var obsSub = tf.gather(obsTensor, idxT, 0);
        var maskSub = tf.gather(maskTensor, idxT, 0);
        idxT.dispose();

        var entropyScalar = 0;
        var actionsKept;
        var logProbKept;
        var valueKept;
        var mod = this.models[mm];
        tf.tidy(function () {
          var fwd = mod.forward(obsSub);
          var logits = fwd.policy;
          var vals = fwd.value;
          var maskedLogits = logits.add(maskSub.sub(1).mul(1e9));
          var probs = tf.softmax(maskedLogits);
          var entMean = probs.mul(tf.log(probs.add(1e-8))).sum(1).neg().mean();
          entropyScalar = entMean.dataSync()[0];

          var multinom = tf.multinomial(maskedLogits, 2);
          var actionsT = multinom.slice([0, 1], [-1, 1]).squeeze([1]);
          var logSM = tf.logSoftmax(maskedLogits);
          var oh = tf.oneHot(actionsT.cast('int32'), boardSize);
          var logProbT = logSM.mul(oh).sum(1);

          actionsKept = tf.keep(actionsT.clone());
          logProbKept = tf.keep(logProbT.clone());
          valueKept = tf.keep(vals.clone());
        });
        obsSub.dispose();
        maskSub.dispose();

        var ad = actionsKept.dataSync();
        var lp = logProbKept.dataSync();
        var vd = valueKept.dataSync();
        actionsKept.dispose();
        logProbKept.dispose();
        valueKept.dispose();

        for (var jj = 0; jj < km; jj++) {
          var origRow = idxList[jj];
          var ai = Math.round(ad[jj]);
          actions[origRow] = ai;
          logProbs[origRow] = lp[jj];
          values[origRow] = vd[jj];
        }
        entropySum += entropyScalar * km;
        entropyCount += km;
      }
    } catch (err) {
      try {
        obsTensor.dispose();
      } catch (e0) { /* noop */ }
      try {
        maskTensor.dispose();
      } catch (e1) { /* noop */ }
      batch._obsTensor = null;
      batch._maskTensor = null;
      throw err;
    }

    this.engine._trackReadback(k * boardSize);
    var obsFlat = obsTensor.dataSync();
    obsTensor.dispose();
    maskTensor.dispose();
    batch._obsTensor = null;
    batch._maskTensor = null;

    batch.states = [];
    batch.masks = [];
    var EPS = 1e-6;
    for (var r = 0; r < k; r++) {
      var offset = r * boardSize;
      var row = new Float32Array(boardSize);
      row.set(obsFlat.subarray(offset, offset + boardSize));
      batch.states.push(row);
      var maskRow = new Float32Array(boardSize);
      for (var j = 0; j < boardSize; j++) {
        maskRow[j] = Math.abs(row[j]) < EPS ? 1 : 0;
      }
      batch.masks.push(maskRow);
    }

    var actionsBySlot = new Int32Array(N);
    for (var i = 0; i < k; i++) {
      var slot = batch.slotIds[i];
      var ai = Math.round(actions[i]);
      if (ai < 0 || ai >= boardSize || batch.masks[i][ai] <= 0) {
        ai = self._fallbackAction(batch.masks[i]);
      }
      actionsBySlot[slot] = ai;
      actions[i] = ai;
    }

    return {
      slotIds: batch.slotIds,
      actionsBySlot: actionsBySlot,
      actions: actions,
      logProbs: logProbs,
      values: values,
      entropySum: entropySum,
      entropyCount: entropyCount > 0 ? entropyCount : k
    };
  }

  _gpuBatchedOneModelTensors(obsTensor, maskTensor, batch, model) {
    var self = this;
    var k = batch.slotIds.length;
    var boardSize = this.boardSize;
    var N = this.numGames;

    var entropyScalar = 0;
    var actionsKept;
    var logProbKept;
    var valueKept;
    try {
      tf.tidy(function () {
        var fwd = model.forward(obsTensor);
        var logits = fwd.policy;
        var vals = fwd.value;
        var maskedLogits = logits.add(maskTensor.sub(1).mul(1e9));
        var probs = tf.softmax(maskedLogits);
        var entMean = probs.mul(tf.log(probs.add(1e-8))).sum(1).neg().mean();
        entropyScalar = entMean.dataSync()[0];

        var multinom = tf.multinomial(maskedLogits, 2);
        var actionsT = multinom.slice([0, 1], [-1, 1]).squeeze([1]);
        var logSM = tf.logSoftmax(maskedLogits);
        var oh = tf.oneHot(actionsT.cast('int32'), boardSize);
        var logProbT = logSM.mul(oh).sum(1);

        actionsKept = tf.keep(actionsT.clone());
        logProbKept = tf.keep(logProbT.clone());
        valueKept = tf.keep(vals.clone());
      });
    } catch (err) {
      obsTensor.dispose();
      maskTensor.dispose();
      batch._obsTensor = null;
      batch._maskTensor = null;
      throw err;
    }

    var ad = actionsKept.dataSync();
    var lp = logProbKept.dataSync();
    var vd = valueKept.dataSync();
    actionsKept.dispose();
    logProbKept.dispose();
    valueKept.dispose();

    this.engine._trackReadback(k * boardSize);
    var obsFlat = obsTensor.dataSync();
    obsTensor.dispose();
    maskTensor.dispose();
    batch._obsTensor = null;
    batch._maskTensor = null;

    batch.states = [];
    batch.masks = [];
    var EPS = 1e-6;
    for (var r = 0; r < k; r++) {
      var offset = r * boardSize;
      var row = new Float32Array(boardSize);
      row.set(obsFlat.subarray(offset, offset + boardSize));
      batch.states.push(row);
      var maskRow = new Float32Array(boardSize);
      for (var j = 0; j < boardSize; j++) {
        maskRow[j] = Math.abs(row[j]) < EPS ? 1 : 0;
      }
      batch.masks.push(maskRow);
    }

    var actionsBySlot = new Int32Array(N);
    var actions = new Int32Array(k);
    var logProbs = new Float32Array(k);
    var values = new Float32Array(k);
    for (var i = 0; i < k; i++) {
      var slot = batch.slotIds[i];
      var ai = Math.round(ad[i]);
      if (ai < 0 || ai >= boardSize || batch.masks[i][ai] <= 0) {
        ai = self._fallbackAction(batch.masks[i]);
      }
      actionsBySlot[slot] = ai;
      actions[i] = ai;
      logProbs[i] = lp[i];
      values[i] = vd[i];
    }

    return {
      slotIds: batch.slotIds,
      actionsBySlot: actionsBySlot,
      actions: actions,
      logProbs: logProbs,
      values: values,
      entropySum: entropyScalar * k,
      entropyCount: k
    };
  }

  /**
   * Same masked policy forward + multinomial as _selectWithAlgorithmGpuBatched, but obs/mask
   * stay on GPU (no flattenStates / CPU mask upload). Fills batch.states / batch.masks from one
   * obs readback after actions for snapshots and trajectory.
   */
  _selectWithAlgorithmGpuBatchedTensors(obsTensor, maskTensor, batch) {
    if (this._multiModel) {
      return this._selectWithAlgorithmGpuBatchedTensorsMulti(obsTensor, maskTensor, batch);
    }
    return this._gpuBatchedOneModelTensors(obsTensor, maskTensor, batch, this.model);
  }

  _selectWithAlgorithm(batch) {
    if (this._benchLoopMode === 'sim_random') {
      return this._selectRandomLegal(batch);
    }
    if (this._policySelectUsesGpuBatchedPath() && batch.slotIds.length > 0) {
      if (batch._obsTensor != null && batch._maskTensor != null) {
        try {
          return this._selectWithAlgorithmGpuBatchedTensors(batch._obsTensor, batch._maskTensor, batch);
        } catch (e) {
          if (batch._obsTensor) {
            try { batch._obsTensor.dispose(); } catch (e2) { /* noop */ }
          }
          if (batch._maskTensor) {
            try { batch._maskTensor.dispose(); } catch (e2) { /* noop */ }
          }
          batch._obsTensor = null;
          batch._maskTensor = null;
          console.warn('[gpu_owner] GPU batched tensor select failed, CPU rebuild:', e.message);
          try {
            var ex = this.engine.extractStatesMasksCPU(batch.slotIds, batch._gpuPerspective);
            batch.states = ex.states;
            batch.masks = ex.masks;
            return this._selectWithAlgorithmGpuBatched(batch);
          } catch (e3) {
            console.warn('[gpu_owner] GPU batched (CPU states) failed, using algo.selectActions:', e3.message);
            var ex2 = this.engine.extractStatesMasksCPU(batch.slotIds, batch._gpuPerspective);
            batch.states = ex2.states;
            batch.masks = ex2.masks;
          }
        }
      } else {
        try {
          return this._selectWithAlgorithmGpuBatched(batch);
        } catch (e) {
          console.warn('[gpu_owner] GPU batched policy select failed, using CPU path:', e.message);
        }
      }
    }
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

    if (this._multiModel) {
      var byModelRows = [];
      for (var bm = 0; bm < this._numModels; bm++) {
        byModelRows.push([]);
      }
      for (var ri = 0; ri < k; ri++) {
        byModelRows[this._modelIdxForSlot(batch.slotIds[ri])].push(ri);
      }
      var entropySumM = 0;
      var entropyCountM = 0;
      for (var mm = 0; mm < this._numModels; mm++) {
        var idxList = byModelRows[mm];
        if (idxList.length === 0) continue;
        var stSub = [];
        var maSub = [];
        for (var q = 0; q < idxList.length; q++) {
          var ridx = idxList[q];
          stSub.push(batch.states[ridx]);
          maSub.push(batch.masks[ridx]);
        }
        var resM = this.algos[mm].selectActions(stSub, maSub);
        for (var q2 = 0; q2 < idxList.length; q2++) {
          var origRow = idxList[q2];
          var slot = batch.slotIds[origRow];
          var r = resM && resM[q2] ? resM[q2] : null;
          var action = r && Number.isFinite(r.action) ? r.action : this._fallbackAction(batch.masks[origRow]);
          actionsBySlot[slot] = action;
          actions[origRow] = action;
          if (r && Number.isFinite(r.logProb)) logProbs[origRow] = r.logProb;
          if (r && Number.isFinite(r.value)) values[origRow] = r.value;
        }
        if (this.algos[mm] && Number.isFinite(this.algos[mm].lastEntropy)) {
          entropySumM += this.algos[mm].lastEntropy * idxList.length;
          entropyCountM += idxList.length;
        }
      }
      return {
        slotIds: batch.slotIds,
        actionsBySlot: actionsBySlot,
        actions: actions,
        logProbs: logProbs,
        values: values,
        entropySum: entropySumM,
        entropyCount: entropyCountM > 0 ? entropyCountM : k
      };
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
    if (this._benchLoopMode === 'sim_random') {
      return this._selectRandomCheckpointBatch(batch);
    }
    var N = this.numGames;
    var actionsBySlot = new Int32Array(N);
    var grouped = {};

    for (var i = 0; i < batch.slotIds.length; i++) {
      var slot = batch.slotIds[i];
      var poolIdx = this._multiModel ? this._p2CheckpointPoolIdx[slot] : 0;
      var pool = this._multiModel ? this.pools[poolIdx] : this.pool;
      var checkpointId = this._slotCheckpointId[slot];
      if (checkpointId < 0 || !pool || !pool.hasCheckpoints()) {
        checkpointId = pool ? pool.pickRandomOpponentId() : -1;
        this._slotCheckpointId[slot] = checkpointId;
      }
      if (checkpointId < 0) {
        actionsBySlot[slot] = this._fallbackAction(batch.masks[i]);
        continue;
      }

      var key = this._multiModel ? poolIdx + ':' + checkpointId : '' + checkpointId;
      if (!grouped[key]) {
        grouped[key] = {
          poolIdx: poolIdx,
          checkpointId: checkpointId,
          slotIds: [],
          states: [],
          masks: []
        };
      }
      grouped[key].slotIds.push(slot);
      grouped[key].states.push(batch.states[i]);
      grouped[key].masks.push(batch.masks[i]);
    }

    var keys = Object.keys(grouped);
    var appliedSlots = [];
    for (var g = 0; g < keys.length; g++) {
      var grp = grouped[keys[g]];
      var poolG = this._multiModel ? this.pools[grp.poolIdx] : this.pool;
      var loaded = poolG.loadOpponentById(grp.checkpointId);
      if (loaded < 0) {
        for (var m = 0; m < grp.slotIds.length; m++) {
          var missingSlot = grp.slotIds[m];
          var replacementId = poolG.pickRandomOpponentId();
          this._slotCheckpointId[missingSlot] = replacementId;
          actionsBySlot[missingSlot] = this._fallbackAction(grp.masks[m]);
          appliedSlots.push(missingSlot);
        }
        continue;
      }

      var rs = poolG.selectActions(grp.states, grp.masks);
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

  _findCkptById(pool, id) {
    for (var ci = 0; ci < pool.checkpoints.length; ci++) {
      if (pool.checkpoints[ci].id === id) return pool.checkpoints[ci];
    }
    return null;
  }

  /**
   * League Elo (multi-model): kind 0 self-play skips; 1 own ckpt; 2 live cross; 3 other-pool ckpt.
   */
  _applyMultiLeagueElo(slot, winner) {
    var mi = this._modelIdxForSlot(slot);
    var draw = winner === 0;
    var sa = draw ? 0.5 : winner === 1 ? 1 : 0;
    var kElo = this.pools[mi].eloK;
    var kind = this._p2Kind[slot];
    if (kind === 0) return;

    if (kind === 1) {
      var pool = this.pools[mi];
      var cid = this._slotCheckpointId[slot];
      if (cid >= 0) pool.updateEloForId(cid, winner === 1, draw);
      else pool.updateElo(winner === 1, draw);
      this.leagueElo[mi] = pool.currentElo;
      return;
    }

    if (kind === 2) {
      var mj = this._p2OppArchIdx[slot];
      if (mj < 0 || mj === mi) return;
      var out = eloUpdatePair(this.leagueElo[mi], this.leagueElo[mj], sa, kElo);
      this.leagueElo[mi] = out.a;
      this.leagueElo[mj] = out.b;
      this.pools[mi].currentElo = out.a;
      this.pools[mj].currentElo = out.b;
      return;
    }

    if (kind === 3) {
      var pj = this._p2CheckpointPoolIdx[slot];
      var ckid = this._slotCheckpointId[slot];
      var poolj = this.pools[pj];
      var ck = this._findCkptById(poolj, ckid);
      if (!ck) return;
      var out2 = eloUpdatePair(this.leagueElo[mi], ck.elo, sa, poolj.eloK);
      this.leagueElo[mi] = out2.a;
      ck.elo = out2.b;
      this.pools[mi].currentElo = out2.a;
    }
  }

  _selectWithAlgoBatchForArch(batch, archIdx) {
    var algo = this.algos[archIdx];
    var N = this.numGames;
    var k = batch.slotIds.length;
    var actionsBySlot = new Int32Array(N);
    var actions = new Int32Array(k);
    var logProbs = new Float32Array(k);
    var values = new Float32Array(k);
    for (var z = 0; z < k; z++) {
      logProbs[z] = NaN;
      values[z] = NaN;
    }
    var results = algo.selectActions(batch.states, batch.masks);
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
    if (algo && Number.isFinite(algo.lastEntropy)) {
      entropySum = algo.lastEntropy * k;
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
    var minimal = this._benchMinimalReplay();
    for (var i = 0; i < doneSlots.length; i++) {
      var slot = doneSlots[i];
      var winner = winners[i];
      var vsCheckpoint = !!this._vsCheckpoint[slot];
      var includeP2Train = !this._multiModel ? !vsCheckpoint : this._p2Kind[slot] === 0;
      var trajectory = minimal ? [] : this._downloadTrajectory(slot, includeP2Train);
      if (!minimal && trajectory.length > 0) {
        if (this._multiModel) {
          this.algos[this._modelIdxForSlot(slot)].onGameFinished(trajectory, winner);
        } else {
          this.algo.onGameFinished(trajectory, winner);
        }
      }

      if (!minimal && this._multiModel) {
        this._applyMultiLeagueElo(slot, winner);
      } else if (!minimal && vsCheckpoint && this.pool) {
        var checkpointId = this._slotCheckpointId[slot];
        if (checkpointId >= 0) this.pool.updateEloForId(checkpointId, winner === 1, winner === 0);
        else this.pool.updateElo(winner === 1, winner === 0);
      }

      this.gamesCompleted++;
      if (this._multiModel) {
        this.gamesSinceLastTrain[this._modelIdxForSlot(slot)]++;
      } else {
        this.gamesSinceLastTrain++;
      }
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

  _shouldUseCheckpointForSlot(slot) {
    var pool = this._multiModel ? this.pools[this._modelIdxForSlot(slot)] : this.pool;
    return !!(pool && pool.shouldBeCheckpointGame());
  }

  _pickRandomOtherArch(mi) {
    if (this._numModels < 2) return 0;
    var j = Math.floor(Math.random() * (this._numModels - 1));
    if (j >= mi) j++;
    return j;
  }

  _resetSlotMetadata(indices) {
    for (var i = 0; i < indices.length; i++) {
      var slot = indices[i];
      this._gameGenerations[slot] = this._nextGeneration++;

      if (!this._multiModel) {
        var vsCheckpoint = this._shouldUseCheckpointForSlot(slot);
        var checkpointId = -1;
        var pool = this.pool;
        if (vsCheckpoint && pool && pool.hasCheckpoints()) {
          checkpointId = pool.pickRandomOpponentId();
          if (checkpointId < 0) vsCheckpoint = false;
        } else {
          vsCheckpoint = false;
        }
        this._vsCheckpoint[slot] = vsCheckpoint ? 1 : 0;
        this._slotCheckpointId[slot] = checkpointId;
        continue;
      }

      var mi = this._modelIdxForSlot(slot);
      var r = Math.random();
      var kind = 0;
      var vsCkpt = false;
      var ckptId = -1;
      var ckptPoolIdx = mi;
      this._p2OppArchIdx[slot] = -1;

      if (r < 0.34) {
        kind = 0;
        vsCkpt = false;
      } else if (r < 0.52 && this.pools[mi].hasCheckpoints()) {
        kind = 1;
        vsCkpt = true;
        ckptPoolIdx = mi;
        ckptId = this.pools[mi].pickRandomOpponentId();
        if (ckptId < 0) {
          kind = 0;
          vsCkpt = false;
        }
      } else if (r < 0.74 && this._numModels > 1) {
        kind = 2;
        vsCkpt = false;
        this._p2OppArchIdx[slot] = this._pickRandomOtherArch(mi);
      } else {
        var candidates = [];
        for (var c = 0; c < this._numModels; c++) {
          if (this.pools[c].hasCheckpoints()) candidates.push(c);
        }
        if (candidates.length > 0) {
          kind = 3;
          vsCkpt = true;
          ckptPoolIdx = candidates[Math.floor(Math.random() * candidates.length)];
          ckptId = this.pools[ckptPoolIdx].pickRandomOpponentId();
          if (ckptId < 0) {
            kind = 0;
            vsCkpt = false;
            ckptPoolIdx = mi;
          }
        } else {
          kind = 0;
          vsCkpt = false;
        }
      }

      this._p2Kind[slot] = kind;
      this._vsCheckpoint[slot] = vsCkpt ? 1 : 0;
      this._slotCheckpointId[slot] = ckptId;
      this._p2CheckpointPoolIdx[slot] = ckptPoolIdx;
    }
  }

  _maybeScheduleTrain() {
    if (this._disposed) return;
    if (this._benchDisableTrain) return;
    if (!this.trainQueue || this.trainQueue.hasKey('train')) return;

    if (this._multiModel) {
      var anyMulti = false;
      for (var tm = 0; tm < this._numModels; tm++) {
        if (this.algos[tm].shouldTrain(this.gamesSinceLastTrain[tm], this.trainInterval, this.trainBatchSize)) {
          anyMulti = true;
          break;
        }
      }
      if (!anyMulti) return;
    } else if (!this.algo.shouldTrain(this.gamesSinceLastTrain, this.trainInterval, this.trainBatchSize)) {
      return;
    }

    var self = this;
    this.trainQueue.enqueue('train', function () {
      if (self._disposed) return;
      self._trainInFlight = true;
      var tTrain0 = self._benchInstrument ? performance.now() : 0;
      try {
        if (self._multiModel) {
          var trained = 0;
          var lossAcc = 0;
          for (var m = 0; m < self._numModels; m++) {
            if (!self.algos[m].shouldTrain(self.gamesSinceLastTrain[m], self.trainInterval, self.trainBatchSize)) {
              continue;
            }
            var Lm = self.algos[m].train(self.trainBatchSize);
            self.lastLossByModel[m] = Lm;
            self.generation[m]++;
            trained++;
            lossAcc += Lm;
            var consumedGm = Math.max(1, self.trainInterval || 1);
            self.gamesSinceLastTrain[m] = Math.max(0, self.gamesSinceLastTrain[m] - consumedGm);
            if (self.pools[m] && self.pools[m].shouldSave(self.generation[m])) {
              self.pools[m].save(self.models[m].model, self.generation[m]);
            }
          }
          if (trained > 0) {
            self.lastLoss = lossAcc / trained;
          }
        } else {
          if (!self.algo) return;
          self.lastLoss = self.algo.train(self.trainBatchSize);
          self.generation++;
          var consumedGames = Math.max(1, self.trainInterval || 1);
          self.gamesSinceLastTrain = Math.max(0, self.gamesSinceLastTrain - consumedGames);

          if (self.pool && self.pool.shouldSave(self.generation)) {
            self.pool.save(self.model.model, self.generation);
          }
        }
      } catch (e) {
        console.warn('GPUOwnerRuntime train error:', e.message);
      } finally {
        if (self._benchInstrument && tTrain0) {
          self._benchTrainMsPending += performance.now() - tTrain0;
          self._benchTrainCallsPending++;
        }
        self._trainInFlight = false;
      }
    });
  }

  _tickOne() {
    var ins = this._benchInstrument;
    var t0 = ins ? performance.now() : 0;

    var entropySum = 0;
    var entropyCount = 0;
    var minimal = this._benchMinimalReplay();

    var p1Active = this.engine.getActiveSlots();
    var p1Batch = (this._benchLoopMode === 'sim_random' || !this._policySelectUsesGpuBatchedPath())
      ? this._buildActionBatch(p1Active, 1)
      : this._buildActionBatchGpu(p1Active, 1);
    var p1Sel = null;
    if (p1Batch.slotIds.length > 0) {
      p1Sel = this._selectWithAlgorithm(p1Batch);
      entropySum += p1Sel.entropySum;
      entropyCount += p1Sel.entropyCount;
    }
    if (ins) {
      this._benchPolicyMsBatch += performance.now() - t0;
      t0 = performance.now();
    }

    if (p1Batch.slotIds.length > 0 && p1Sel) {
      this.engine.applyActions(1, p1Batch.slotIds, p1Sel.actionsBySlot);
      if (!minimal) this._recordSnapshot(1, p1Batch, p1Sel);
    }
    if (ins) {
      this._benchPhysicsMsBatch += performance.now() - t0;
      t0 = performance.now();
    }

    var p2ActionsBySlot = new Int32Array(this.numGames);
    var p2ApplySlots = [];
    var p2Active = this.engine.getActiveSlots();
    if (p2Active.length > 0) {
      var splitP2 = this._splitP2Slots(p2Active);
      var selfBatch = (this._benchLoopMode === 'sim_random' || !this._policySelectUsesGpuBatchedPath())
        ? this._buildActionBatch(splitP2.selfSlots, -1)
        : this._buildActionBatchGpu(splitP2.selfSlots, -1);
      var ckptBatch = this._buildActionBatch(splitP2.ckptSlots, -1);

      if (selfBatch.slotIds.length > 0) {
        var p2Sel = this._selectWithAlgorithm(selfBatch);
        for (var i = 0; i < selfBatch.slotIds.length; i++) {
          var slot = selfBatch.slotIds[i];
          p2ActionsBySlot[slot] = p2Sel.actionsBySlot[slot];
          p2ApplySlots.push(slot);
        }
        if (!minimal) this._recordSnapshot(-1, selfBatch, p2Sel);
        entropySum += p2Sel.entropySum;
        entropyCount += p2Sel.entropyCount;
      }

      if (this._multiModel && splitP2.liveCrossSlots.length > 0) {
        var byOpp = {};
        for (var lx = 0; lx < splitP2.liveCrossSlots.length; lx++) {
          var slc = splitP2.liveCrossSlots[lx];
          var ojx = this._p2OppArchIdx[slc];
          if (!byOpp[ojx]) byOpp[ojx] = [];
          byOpp[ojx].push(slc);
        }
        var kOpp = Object.keys(byOpp);
        for (var ko = 0; ko < kOpp.length; ko++) {
          var oj = parseInt(kOpp[ko], 10);
          var slist = byOpp[oj];
          var crossBatch = (this._benchLoopMode === 'sim_random' || !this._policySelectUsesGpuBatchedPath())
            ? this._buildActionBatch(slist, -1)
            : this._buildActionBatchGpu(slist, -1);
          if (crossBatch.slotIds.length === 0) continue;
          var csel;
          try {
            if (this._policySelectUsesGpuBatchedPath()) {
              if (crossBatch._obsTensor != null && crossBatch._maskTensor != null) {
                csel = this._gpuBatchedOneModelTensors(
                  crossBatch._obsTensor,
                  crossBatch._maskTensor,
                  crossBatch,
                  this.models[oj]
                );
              } else {
                csel = this._gpuBatchedOneModel(crossBatch, this.models[oj]);
              }
            } else {
              csel = this._selectWithAlgoBatchForArch(crossBatch, oj);
            }
          } catch (ex) {
            console.warn('[gpu_owner] live-cross policy failed:', ex.message);
            csel = this._selectWithAlgoBatchForArch(crossBatch, oj);
          }
          for (var ci = 0; ci < crossBatch.slotIds.length; ci++) {
            var csl = crossBatch.slotIds[ci];
            p2ActionsBySlot[csl] = csel.actionsBySlot[csl];
            p2ApplySlots.push(csl);
          }
          if (!minimal) this._recordSnapshot(-1, crossBatch, csel);
          entropySum += csel.entropySum;
          entropyCount += csel.entropyCount;
        }
      }

      if (ckptBatch.slotIds.length > 0 && (this._multiModel || (this.pool && this.pool.hasCheckpoints()))) {
        var ckptSel = this._selectWithCheckpoint(ckptBatch);
        for (var j = 0; j < ckptSel.slotIds.length; j++) {
          var cslot = ckptSel.slotIds[j];
          p2ActionsBySlot[cslot] = ckptSel.actionsBySlot[cslot];
          p2ApplySlots.push(cslot);
        }
      }
    }
    if (ins) {
      this._benchPolicyMsBatch += performance.now() - t0;
      t0 = performance.now();
    }

    if (p2ApplySlots.length > 0) {
      this.engine.applyActions(-1, p2ApplySlots, p2ActionsBySlot);
    }
    if (ins) {
      this._benchPhysicsMsBatch += performance.now() - t0;
      t0 = performance.now();
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

    if (ins) {
      this._benchPhysicsMsBatch += performance.now() - t0;
      this._benchTickSamplesBatch++;
    }

    if (entropyCount > 0) {
      this.lastEntropy = entropySum / entropyCount;
      if (this._multiModel && this.algos) {
        for (var ae = 0; ae < this.algos.length; ae++) {
          this.algos[ae].lastEntropy = this.lastEntropy;
        }
      } else if (this.algo) {
        this.algo.lastEntropy = this.lastEntropy;
      }
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
    var rb = this.engine ? this.engine.consumeReadbackFrame() : { calls: 0, bytes: 0 };
    var genVal = this.generation;
    var bufSz = this.algo && this.algo.getBufferSize ? this.algo.getBufferSize() : 0;
    var trainStepsVal = this.algo && this.algo.getTrainSteps ? this.algo.getTrainSteps() : 0;
    var eloVal = this.pool ? this.pool.getCurrentElo() : 0;
    var ckptWr = this.pool ? this.pool.getRecentWinRate() : 0;
    var modelStats = null;

    if (this._multiModel && this.algos && this.pools && this._modelTypeIds) {
      var maxGen = 0;
      var sumBuf = 0;
      var sumSteps = 0;
      modelStats = [];
      for (var ms = 0; ms < this._numModels; ms++) {
        maxGen = Math.max(maxGen, this.generation[ms]);
        var bsM = this.algos[ms].getBufferSize ? this.algos[ms].getBufferSize() : 0;
        sumBuf += bsM;
        var tsM = this.algos[ms].getTrainSteps ? this.algos[ms].getTrainSteps() : 0;
        sumSteps += tsM;
        modelStats.push({
          id: this._modelTypeIds[ms],
          elo: this.leagueElo ? this.leagueElo[ms] : this.pools[ms].getCurrentElo(),
          generation: this.generation[ms],
          loss: this.lastLossByModel[ms],
          bufferSize: bsM,
          checkpointWinRate: this.pools[ms].getRecentWinRate(),
          trainSteps: tsM
        });
      }
      genVal = maxGen;
      bufSz = sumBuf;
      trainStepsVal = sumSteps;
      eloVal = modelStats.length > 0 ? modelStats[0].elo : 1000;
      var ckSum = 0;
      for (var cx = 0; cx < modelStats.length; cx++) {
        ckSum += modelStats[cx].checkpointWinRate || 0;
      }
      ckptWr = modelStats.length > 0 ? ckSum / modelStats.length : 0;
    }

    var stats = {
      gamesCompleted: this.gamesCompleted,
      generation: genVal,
      loss: this.lastLoss,
      p1Wins: this.p1Wins,
      p2Wins: this.p2Wins,
      draws: this.draws,
      avgGameLength: avgLen,
      bufferSize: bufSz,
      trainSteps: trainStepsVal,
      elo: eloVal,
      checkpointWinRate: ckptWr,
      entropy: this.lastEntropy,
      trainInFlight: !!this._trainInFlight,
      gpuReadbackCalls: rb.calls,
      gpuReadbackBytes: rb.bytes,
      multiModel: !!this._multiModel,
      modelStats: modelStats
    };

    if (this._benchLoopMode !== 'off') {
      stats.benchLoopMode = this._benchLoopMode;
    }
    if (this._benchInstrument && this._benchTickSamplesBatch > 0) {
      stats.benchAvgPolicyMsPerSimTick = this._benchPolicyMsBatch / this._benchTickSamplesBatch;
      stats.benchAvgPhysicsMsPerSimTick = this._benchPhysicsMsBatch / this._benchTickSamplesBatch;
    }
    if (this._benchInstrument) {
      stats.benchInstrument = true;
    }
    this._benchPolicyMsBatch = 0;
    this._benchPhysicsMsBatch = 0;
    this._benchTickSamplesBatch = 0;

    if (this._benchTrainCallsPending > 0) {
      stats.benchTrainMs = this._benchTrainMsPending;
      stats.benchTrainCalls = this._benchTrainCallsPending;
    }
    this._benchTrainMsPending = 0;
    this._benchTrainCallsPending = 0;

    return stats;
  }

  _boardUiSlotIdsForSnapshot() {
    var N = this.numGames;
    var m = this._uiSnapshotMaxGames;
    if (m >= N) {
      var all = new Array(N);
      for (var i = 0; i < N; i++) all[i] = i;
      return all;
    }
    if (!this._hasDoneFullBoardSync) {
      var warm = new Array(N);
      for (var j = 0; j < N; j++) warm[j] = j;
      return warm;
    }
    var pageCount = Math.ceil(N / m);
    var page = this._boardSnapshotPage % pageCount;
    this._boardSnapshotPage++;
    var start = page * m;
    var out = [];
    for (var k = 0; k < m && start + k < N; k++) out.push(start + k);
    return out;
  }

  _emitTickResult() {
    var payload = { type: MSG.TICK_RESULT };
    var xfer = null;

    if (this._ticks % this._snapshotEveryTicks === 0) {
      var slotIds = this._boardUiSlotIdsForSnapshot();
      var boards = slotIds.length === this.numGames
        ? this.engine.getBoardsForRender()
        : this.engine.getBoardsForRenderGather(slotIds);
      if (slotIds.length === this.numGames) {
        this._hasDoneFullBoardSync = true;
      }
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
      xfer = [payload.boards.buffer, payload.done.buffer, payload.winners.buffer];
      if (slotIds.length < this.numGames) {
        var slotsU16 = new Uint16Array(slotIds.length);
        for (var si = 0; si < slotIds.length; si++) slotsU16[si] = slotIds[si];
        payload.boardSampleSlots = slotsU16;
        xfer.push(payload.boardSampleSlots.buffer);
      }
    }

    payload.stats = this._collectStats();

    if (xfer) {
      this._post(payload, xfer);
    } else {
      this._post(payload);
    }
  }

  async tick(steps) {
    if (!this._ready || this._disposed || !this.engine) return;
    if (this.pauseTicksWhenTraining && this._trainInFlight) {
      this._emitTickResult();
      return;
    }
    this.engine.beginReadbackFrame();
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

    if (this._multiModel && this.algos) {
      for (var dx = 0; dx < this.algos.length; dx++) {
        if (this.algos[dx] && typeof this.algos[dx].dispose === 'function') {
          try {
            this.algos[dx].dispose();
          } catch (e) { /* noop */ }
        }
      }
    } else if (this.algo && typeof this.algo.dispose === 'function') {
      try {
        this.algo.dispose();
      } catch (e) { /* noop */ }
    }
    this.algo = null;
    this.model = null;
    this.pool = null;
    this.models = null;
    this.algos = null;
    this.pools = null;
    this._multiModel = false;
    this._slotModelIndex = null;
    this.leagueElo = null;
    this._p2Kind = null;
    this._p2OppArchIdx = null;
    this._p2CheckpointPoolIdx = null;

    this._stepSnapshots = [];
    this._gameGenerations = null;
    this._vsCheckpoint = null;
    this._slotCheckpointId = null;
    this._trainInFlight = false;

    this._ready = false;
    this._disposed = true;
  }
}
