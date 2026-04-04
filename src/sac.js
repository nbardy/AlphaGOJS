import * as tf from '@tensorflow/tfjs';
import { flattenStates, maskedSoftmax, sampleFromProbs } from './action';

// Discrete Soft Actor-Critic (SAC) for board-index action spaces.
// Actor: reuses the selected policy model (Dense/Spatial).
// Critics: twin Q networks + target networks (off-policy replay).
//
// Objective:
//   maximize E[Q(s,a)] + alpha * entropy(pi(.|s))
// with automatic alpha adaptation outside the gradient tape.

export class SAC {
  constructor(model) {
    if (model.expectsDiscreteInput) {
      throw new Error('SAC does not support discrete-observation models (patch3_discrete).');
    }
    this.model = model; // actor network (policy logits come from model.forward)
    this.boardSize = model.boardSize;

    // Optimizers
    this.actorOptimizer = tf.train.adam(0.0003);
    this.q1Optimizer = tf.train.adam(0.001);
    this.q2Optimizer = tf.train.adam(0.001);

    // Twin Q networks + slowly updated target networks
    this.q1 = this._buildQ();
    this.q2 = this._buildQ();
    this.targetQ1 = this._buildQ();
    this.targetQ2 = this._buildQ();
    this._hardSyncTargets();

    // Replay buffer (off-policy)
    this.buffer = [];
    this.maxBufferSize = 100000;
    this.minReplaySize = 1024;

    // Hyperparameters
    this.gamma = 0.99;
    this.tau = 0.01;
    this.minibatchSize = 128;
    this.updatesPerTrain = 2;

    // Entropy temperature alpha (log-space scalar)
    this.logAlpha = Math.log(0.03);
    this.alphaLr = 0.005;
    this.targetEntropy = 0.6 * Math.log(this.boardSize * 0.3);
    this.lastEntropy = 0;

    this.trainSteps = 0;
  }

  _buildQ() {
    var input = tf.input({ shape: [this.boardSize] });
    var x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input);
    x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(x);
    var out = tf.layers.dense({ units: this.boardSize }).apply(x);
    return tf.model({ inputs: input, outputs: out });
  }

  _hardSyncTargets() {
    this.targetQ1.setWeights(this.q1.getWeights());
    this.targetQ2.setWeights(this.q2.getWeights());
  }

  _softUpdateTarget(targetNet, sourceNet) {
    var tau = this.tau;
    var tWeights = targetNet.getWeights();
    var sWeights = sourceNet.getWeights();
    var mixed = [];
    for (var i = 0; i < tWeights.length; i++) {
      mixed.push(tf.tidy(function (tw, sw) {
        return tw.mul(1 - tau).add(sw.mul(tau));
      }.bind(null, tWeights[i], sWeights[i])));
    }
    targetNet.setWeights(mixed);
    for (var i = 0; i < tWeights.length; i++) tWeights[i].dispose();
    for (var i = 0; i < sWeights.length; i++) sWeights[i].dispose();
    for (var i = 0; i < mixed.length; i++) mixed[i].dispose();
  }

  selectActions(states, masks) {
    var n = states.length;
    var boardSize = this.boardSize;
    if (n === 0) return [];

    var statesTensor = tf.tensor2d(flattenStates(states, boardSize), [n, boardSize]);
    var out = this.model.forward(statesTensor);
    var logitsData = out.policy.dataSync();
    out.policy.dispose();
    out.value.dispose();
    statesTensor.dispose();

    var results = [];
    var entropySum = 0;
    for (var i = 0; i < n; i++) {
      var logits = new Float32Array(boardSize);
      for (var j = 0; j < boardSize; j++) logits[j] = logitsData[i * boardSize + j];
      var probs = maskedSoftmax(logits, masks[i]);
      var action = sampleFromProbs(probs);
      var ent = 0;
      for (var j = 0; j < boardSize; j++) {
        if (probs[j] > 1e-8) ent -= probs[j] * Math.log(probs[j]);
      }
      entropySum += ent;
      results.push({ action: action });
    }
    this.lastEntropy = n > 0 ? entropySum / n : 0;
    return results;
  }

  selectAction(state, mask) {
    return this.selectActions([state], [mask])[0].action;
  }

  onGameFinished(trajectory, winner) {
    var p1 = [];
    var p2 = [];
    for (var i = 0; i < trajectory.length; i++) {
      if (trajectory[i].player === 1) p1.push(trajectory[i]);
      else p2.push(trajectory[i]);
    }
    this._pushPlayerTransitions(p1, winner, 1);
    this._pushPlayerTransitions(p2, winner, -1);

    if (this.buffer.length > this.maxBufferSize) {
      this.buffer = this.buffer.slice(-this.maxBufferSize);
    }
  }

  _pushPlayerTransitions(steps, winner, player) {
    if (steps.length === 0) return;
    var terminalReward = (player === winner) ? 1 : (winner === 0 ? 0 : -1);

    for (var t = 0; t < steps.length; t++) {
      var isLast = t === steps.length - 1;
      var nextStep = isLast ? null : steps[t + 1];
      this.buffer.push({
        state: steps[t].state,
        mask: steps[t].mask,
        action: steps[t].action,
        reward: isLast ? terminalReward : 0,
        nextState: nextStep ? nextStep.state : steps[t].state,
        nextMask: nextStep ? nextStep.mask : steps[t].mask,
        done: isLast ? 1 : 0
      });
    }
  }

  shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize) {
    var minSize = Math.max(trainBatchSize, this.minReplaySize);
    return gamesSinceLastTrain >= trainInterval && this.buffer.length >= minSize;
  }

  _sampleBatch(batchSize) {
    var n = this.buffer.length;
    var size = Math.min(batchSize, n);
    var batch = [];
    for (var i = 0; i < size; i++) {
      batch.push(this.buffer[Math.floor(Math.random() * n)]);
    }
    return batch;
  }

  train(batchSize) {
    if (this.buffer.length < this.minReplaySize) return 0;

    var updates = Math.max(1, this.updatesPerTrain);
    var totalLoss = 0;
    var doneUpdates = 0;
    var mb = Math.min(this.minibatchSize, batchSize);

    for (var u = 0; u < updates; u++) {
      var batch = this._sampleBatch(mb);
      var loss = this._trainOnBatch(batch);
      totalLoss += loss;
      doneUpdates++;
    }

    // Adaptive alpha: raise alpha when entropy collapses.
    if (this.lastEntropy > 0) {
      var err = this.targetEntropy - this.lastEntropy;
      this.logAlpha = Math.max(Math.log(0.001), Math.min(Math.log(0.2), this.logAlpha + this.alphaLr * err));
    }

    this.trainSteps++;
    return doneUpdates > 0 ? totalLoss / doneUpdates : 0;
  }

  _trainOnBatch(batch) {
    var n = batch.length;
    var boardSize = this.boardSize;
    if (n === 0) return 0;

    var statesArr = [];
    var nextStatesArr = [];
    var actions = new Int32Array(n);
    var rewards = new Float32Array(n);
    var dones = new Float32Array(n);
    var masksArr = [];
    var nextMasksArr = [];

    for (var i = 0; i < n; i++) {
      statesArr.push(batch[i].state);
      nextStatesArr.push(batch[i].nextState);
      actions[i] = batch[i].action;
      rewards[i] = batch[i].reward;
      dones[i] = batch[i].done;
      masksArr.push(batch[i].mask);
      nextMasksArr.push(batch[i].nextMask);
    }

    var statesT = tf.tensor2d(flattenStates(statesArr, boardSize), [n, boardSize]);
    var nextStatesT = tf.tensor2d(flattenStates(nextStatesArr, boardSize), [n, boardSize]);
    var rewardsT = tf.tensor1d(rewards);
    var donesT = tf.tensor1d(dones);

    var actionMaskData = new Float32Array(n * boardSize);
    var validMaskData = new Float32Array(n * boardSize);
    var nextValidMaskData = new Float32Array(n * boardSize);
    for (var i = 0; i < n; i++) {
      actionMaskData[i * boardSize + actions[i]] = 1;
      for (var j = 0; j < boardSize; j++) {
        validMaskData[i * boardSize + j] = masksArr[i][j];
        nextValidMaskData[i * boardSize + j] = nextMasksArr[i][j];
      }
    }
    var actionMaskT = tf.tensor2d(actionMaskData, [n, boardSize]);
    var validMaskT = tf.tensor2d(validMaskData, [n, boardSize]);
    var nextValidMaskT = tf.tensor2d(nextValidMaskData, [n, boardSize]);

    var alpha = Math.exp(this.logAlpha);
    var alphaT = tf.scalar(alpha);
    var gammaT = tf.scalar(this.gamma);

    var targetQT = tf.tidy(function () {
      var outNext = this.model.model.predict(nextStatesT);
      var nextLogits = outNext.slice([0, 0], [-1, boardSize]);
      var maskedNextLogits = nextLogits.add(nextValidMaskT.sub(1).mul(1e9));
      var nextProbs = maskedNextLogits.softmax();
      var nextLogProbs = nextProbs.add(tf.scalar(1e-8)).log();

      var q1Next = this.targetQ1.predict(nextStatesT);
      var q2Next = this.targetQ2.predict(nextStatesT);
      var minQNext = tf.minimum(q1Next, q2Next);
      var vNext = nextProbs.mul(minQNext.sub(nextLogProbs.mul(alphaT))).sum(1);

      var notDone = tf.scalar(1).sub(donesT);
      return rewardsT.add(gammaT.mul(notDone).mul(vNext));
    }.bind(this));

    var q1VarList = this.q1.trainableWeights.map(function (w) { return w.val; });
    var q2VarList = this.q2.trainableWeights.map(function (w) { return w.val; });
    var actorVarList = this.model.model.trainableWeights.map(function (w) { return w.val; });

    var q1LossVal = 0;
    var q2LossVal = 0;
    var actorLossVal = 0;

    var q1LossT = this.q1Optimizer.minimize(function () {
      var q1All = this.q1.predict(statesT);
      var q1Sel = q1All.mul(actionMaskT).sum(1);
      return q1Sel.sub(targetQT).square().mean();
    }.bind(this), true, q1VarList);
    if (q1LossT) {
      q1LossVal = q1LossT.dataSync()[0];
      q1LossT.dispose();
    }

    var q2LossT = this.q2Optimizer.minimize(function () {
      var q2All = this.q2.predict(statesT);
      var q2Sel = q2All.mul(actionMaskT).sum(1);
      return q2Sel.sub(targetQT).square().mean();
    }.bind(this), true, q2VarList);
    if (q2LossT) {
      q2LossVal = q2LossT.dataSync()[0];
      q2LossT.dispose();
    }

    var actorLossT = this.actorOptimizer.minimize(function () {
      var out = this.model.model.predict(statesT);
      var logits = out.slice([0, 0], [-1, boardSize]);
      var maskedLogits = logits.add(validMaskT.sub(1).mul(1e9));
      var probs = maskedLogits.softmax();
      var logProbs = probs.add(tf.scalar(1e-8)).log();

      var q1All = this.q1.predict(statesT);
      var q2All = this.q2.predict(statesT);
      var minQ = tf.minimum(q1All, q2All);

      return probs.mul(logProbs.mul(alphaT).sub(minQ)).sum(1).mean();
    }.bind(this), true, actorVarList);
    if (actorLossT) {
      actorLossVal = actorLossT.dataSync()[0];
      actorLossT.dispose();
    }

    this._softUpdateTarget(this.targetQ1, this.q1);
    this._softUpdateTarget(this.targetQ2, this.q2);

    statesT.dispose();
    nextStatesT.dispose();
    rewardsT.dispose();
    donesT.dispose();
    actionMaskT.dispose();
    validMaskT.dispose();
    nextValidMaskT.dispose();
    alphaT.dispose();
    gammaT.dispose();
    targetQT.dispose();

    return q1LossVal + q2LossVal + actorLossVal;
  }

  getTrainSteps() {
    return this.trainSteps;
  }

  getBufferSize() {
    return this.buffer.length;
  }
}
