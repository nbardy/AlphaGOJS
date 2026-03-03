import * as tf from '@tensorflow/tfjs';
import { PPO } from './ppo';
import { flattenStates } from './action';

// Phasic Policy Gradient (PPG) on top of PPO.
//
// Phase 1: standard PPO updates (handled by superclass).
// Phase 2 (periodic): auxiliary value optimization with policy KL anchoring.
// This reduces policy drift while allowing stronger critic fitting.

export class PPG extends PPO {
  constructor(model) {
    super(model);

    // PPG-specific knobs
    this.auxFrequency = 4;   // run aux phase every N PPO train() calls
    this.auxEpochs = 2;
    this.auxBatchSize = 128;
    this.auxBcCoeff = 1.0;   // KL(old_policy || new_policy) weight
  }

  train(batchSize) {
    // Capture an auxiliary snapshot first (before PPO consumes buffer items).
    var auxBatch = this._sampleAuxBatch(Math.max(batchSize, this.auxBatchSize * 2));

    // Phase 1: PPO
    var ppoLoss = super.train(batchSize);

    // Phase 2: auxiliary critic training (periodic)
    var auxLoss = 0;
    if (this.trainSteps > 0 &&
        this.trainSteps % this.auxFrequency === 0 &&
        auxBatch.length >= Math.min(32, this.auxBatchSize)) {
      auxLoss = this._auxTrain(auxBatch);
    }

    return ppoLoss + auxLoss;
  }

  _sampleAuxBatch(targetSize) {
    var n = this.buffer.length;
    if (n === 0) return [];
    var size = Math.min(targetSize, n);
    var out = [];
    for (var i = 0; i < size; i++) {
      out.push(this.buffer[Math.floor(Math.random() * n)]);
    }
    return out;
  }

  _auxTrain(batch) {
    var n = batch.length;
    var boardSize = this.model.boardSize;
    var statesArr = [];
    var masksArr = [];
    var returnsArr = new Float32Array(n);

    for (var i = 0; i < n; i++) {
      statesArr.push(batch[i].state);
      masksArr.push(batch[i].mask);
      returnsArr[i] = batch[i].returnVal;
    }

    var statesT = tf.tensor2d(flattenStates(statesArr, boardSize), [n, boardSize]);
    var validMaskData = new Float32Array(n * boardSize);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < boardSize; j++) {
        validMaskData[i * boardSize + j] = masksArr[i][j];
      }
    }
    var validMaskT = tf.tensor2d(validMaskData, [n, boardSize]);
    var returnsT = tf.tensor1d(returnsArr);

    // Freeze the pre-aux policy distribution as KL anchor.
    var oldPolicyT = tf.tidy(function () {
      var combined = this.model.model.predict(statesT);
      var logits = combined.slice([0, 0], [-1, boardSize]);
      var masked = logits.add(validMaskT.sub(1).mul(1e9));
      return masked.softmax();
    }.bind(this));

    var totalLoss = 0;
    var updates = 0;
    var mbSize = Math.min(this.auxBatchSize, n);
    var varList = this.model.model.trainableWeights.map(function (w) { return w.val; });

    for (var epoch = 0; epoch < this.auxEpochs; epoch++) {
      var indices = [];
      for (var i = 0; i < n; i++) indices.push(i);
      for (var i = n - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
      }

      for (var start = 0; start + mbSize <= n; start += mbSize) {
        var mbIndices = indices.slice(start, start + mbSize);
        var gather = new Int32Array(mbSize);
        for (var gi = 0; gi < mbSize; gi++) gather[gi] = mbIndices[gi];

        var mbStates = tf.gather(statesT, Array.from(gather));
        var mbMask = tf.gather(validMaskT, Array.from(gather));
        var mbReturns = tf.gather(returnsT, Array.from(gather));
        var mbOldPolicy = tf.gather(oldPolicyT, Array.from(gather));

        var lossT = this.optimizer.minimize(function () {
          var combined = this.model.model.predict(mbStates);
          var logits = combined.slice([0, 0], [-1, boardSize]);
          var values = combined.slice([0, boardSize], [-1, 1]).squeeze([1]);
          var masked = logits.add(mbMask.sub(1).mul(1e9));
          var probs = masked.softmax();

          var valueLoss = mbReturns.sub(values).square().mean();
          var oldLog = mbOldPolicy.add(tf.scalar(1e-8)).log();
          var newLog = probs.add(tf.scalar(1e-8)).log();
          var kl = mbOldPolicy.mul(oldLog.sub(newLog)).sum(1).mean();

          return valueLoss.add(kl.mul(tf.scalar(this.auxBcCoeff)));
        }.bind(this), true, varList);

        if (lossT) {
          totalLoss += lossT.dataSync()[0];
          lossT.dispose();
          updates++;
        }

        mbStates.dispose();
        mbMask.dispose();
        mbReturns.dispose();
        mbOldPolicy.dispose();
      }
    }

    statesT.dispose();
    validMaskT.dispose();
    returnsT.dispose();
    oldPolicyT.dispose();

    return updates > 0 ? totalLoss / updates : 0;
  }
}
