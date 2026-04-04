import * as tf from '@tensorflow/tfjs';
import { flattenStates, maskedSoftmax, sampleFromProbs } from './action';

// Stochastic MuZero-style planner (lightweight, JS/TF.js friendly).
//
// - Learns dynamics f(s, a) -> (s', r)
// - Uses learned one-step lookahead for action selection
// - Trains policy/value on replayed outcomes + entropy regularization
//
// This is a practical approximation for browser training, not a full UCT tree.

export class StochasticMuZero {
  constructor(model) {
    if (model.expectsDiscreteInput) {
      throw new Error(
        'StochasticMuZero does not support discrete-observation models (patch3_discrete).'
      );
    }
    this.model = model; // policy/value network
    this.boardSize = model.boardSize;

    this.optimizer = tf.train.adam(0.0003);
    this.dynamics = this._buildDynamics();
    this.dynamicsOptimizer = tf.train.adam(0.001);

    this.buffer = [];
    this.maxBufferSize = 100000;
    this.minReplaySize = 1024;
    this.minibatchSize = 128;
    this.updatesPerTrain = 2;

    this.gamma = 0.99;
    this.valueLossCoeff = 0.5;

    this.entropyCoeff = 0.03;
    this.targetEntropy = 0.5 * Math.log(this.boardSize * 0.3);
    this.entropyAlpha = 0.0005;
    this.lastEntropy = 0;

    // Planning knobs
    this.planCandidates = 6;      // top-k from policy prior
    this.planPriorCoeff = 0.25;   // prior log-prob regularization
    this.planTemperature = 0.8;   // action sampling temperature
    this.planNoise = 0.03;        // stochastic reward perturbation

    this.trainSteps = 0;
  }

  _buildDynamics() {
    var input = tf.input({ shape: [this.boardSize * 2] }); // [state, oneHot(action)]
    var x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input);
    x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(x);
    var nextState = tf.layers.dense({ units: this.boardSize, activation: 'tanh' }).apply(x);
    var reward = tf.layers.dense({ units: 1, activation: 'tanh' }).apply(x);
    var combined = tf.layers.concatenate().apply([nextState, reward]);
    return tf.model({ inputs: input, outputs: combined });
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

    var priors = [];
    var results = [];
    var entropySum = 0;

    for (var i = 0; i < n; i++) {
      var logits = new Float32Array(boardSize);
      for (var j = 0; j < boardSize; j++) logits[j] = logitsData[i * boardSize + j];
      var probs = maskedSoftmax(logits, masks[i]);
      priors.push(probs);
      results.push({ action: sampleFromProbs(probs) }); // fallback

      var ent = 0;
      for (var j = 0; j < boardSize; j++) {
        if (probs[j] > 1e-8) ent -= probs[j] * Math.log(probs[j]);
      }
      entropySum += ent;
    }
    this.lastEntropy = n > 0 ? entropySum / n : 0;

    // Build batched one-step lookahead for top-k candidates per state.
    var perState = [];
    for (var i = 0; i < n; i++) perState.push([]);
    var candState = [];
    var candAction = [];
    var candPrior = [];

    for (var i = 0; i < n; i++) {
      var top = this._topKValidActions(priors[i], masks[i], this.planCandidates);
      for (var t = 0; t < top.length; t++) {
        candState.push(i);
        candAction.push(top[t]);
        candPrior.push(priors[i][top[t]]);
      }
    }

    if (candAction.length === 0) return results;

    var m = candAction.length;
    var flatStates = new Float32Array(m * boardSize);
    var flatActions = new Float32Array(m * boardSize);
    for (var c = 0; c < m; c++) {
      var si = candState[c];
      flatStates.set(states[si], c * boardSize);
      flatActions[c * boardSize + candAction[c]] = 1;
    }

    var stateT = tf.tensor2d(flatStates, [m, boardSize]);
    var actT = tf.tensor2d(flatActions, [m, boardSize]);
    var dynIn = tf.concat([stateT, actT], 1);
    stateT.dispose();
    actT.dispose();

    var dynOut = this.dynamics.predict(dynIn);
    var nextStateT = dynOut.slice([0, 0], [-1, boardSize]);
    var rewardT = dynOut.slice([0, boardSize], [-1, 1]).squeeze([1]);

    var nextOut = this.model.forward(nextStateT);
    var nextValue = nextOut.value.dataSync();
    var rewardData = rewardT.dataSync();

    nextOut.policy.dispose();
    nextOut.value.dispose();
    dynIn.dispose();
    dynOut.dispose();
    nextStateT.dispose();
    rewardT.dispose();

    for (var c = 0; c < m; c++) {
      var si = candState[c];
      var prior = candPrior[c];
      var noisyReward = rewardData[c] + (Math.random() * 2 - 1) * this.planNoise;
      var score = noisyReward +
        this.gamma * nextValue[c] +
        this.planPriorCoeff * Math.log(prior + 1e-8);
      perState[si].push({ action: candAction[c], score: score });
    }

    // Sample from softmax(scores / temperature) per state.
    for (var i = 0; i < n; i++) {
      if (perState[i].length === 0) continue;
      var choice = this._sampleByScore(perState[i], this.planTemperature);
      results[i] = { action: choice.action };
    }

    return results;
  }

  selectAction(state, mask) {
    return this.selectActions([state], [mask])[0].action;
  }

  _topKValidActions(probs, mask, k) {
    var items = [];
    for (var i = 0; i < probs.length; i++) {
      if (mask[i] > 0) items.push({ idx: i, p: probs[i] });
    }
    items.sort(function (a, b) { return b.p - a.p; });
    var out = [];
    for (var i = 0; i < items.length && i < k; i++) out.push(items[i].idx);
    return out;
  }

  _sampleByScore(candidates, temperature) {
    if (candidates.length === 1) return candidates[0];
    var t = Math.max(0.05, temperature);
    var maxS = -Infinity;
    for (var i = 0; i < candidates.length; i++) {
      if (candidates[i].score > maxS) maxS = candidates[i].score;
    }
    var probs = new Float32Array(candidates.length);
    var sum = 0;
    for (var i = 0; i < candidates.length; i++) {
      var v = Math.exp((candidates[i].score - maxS) / t);
      probs[i] = v;
      sum += v;
    }
    if (sum <= 1e-8) return candidates[0];
    for (var i = 0; i < probs.length; i++) probs[i] /= sum;
    var picked = sampleFromProbs(probs);
    return candidates[picked];
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
    var n = steps.length;

    for (var t = 0; t < n; t++) {
      var isLast = t === n - 1;
      var nextStep = isLast ? null : steps[t + 1];
      var ret = terminalReward * Math.pow(this.gamma, n - 1 - t);
      this.buffer.push({
        state: steps[t].state,
        mask: steps[t].mask,
        action: steps[t].action,
        reward: isLast ? terminalReward : 0,
        returnVal: ret,
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
    var out = [];
    for (var i = 0; i < size; i++) {
      out.push(this.buffer[Math.floor(Math.random() * n)]);
    }
    return out;
  }

  train(batchSize) {
    if (this.buffer.length < this.minReplaySize) return 0;

    var mb = Math.min(this.minibatchSize, batchSize);
    var updates = Math.max(1, this.updatesPerTrain);
    var total = 0;
    var count = 0;

    for (var u = 0; u < updates; u++) {
      var b = this._sampleBatch(mb);
      total += this._trainOnBatch(b);
      count++;
    }

    // Adaptive entropy coeff for policy objective.
    if (this.lastEntropy > 0) {
      var err = this.targetEntropy - this.lastEntropy;
      this.entropyCoeff = Math.max(0.001, Math.min(0.1, this.entropyCoeff + this.entropyAlpha * err));
    }

    this.trainSteps++;
    return count > 0 ? total / count : 0;
  }

  _trainOnBatch(batch) {
    var n = batch.length;
    var boardSize = this.boardSize;
    if (n === 0) return 0;

    var statesArr = [];
    var nextStatesArr = [];
    var actions = new Int32Array(n);
    var rewards = new Float32Array(n);
    var returns = new Float32Array(n);
    var masksArr = [];

    for (var i = 0; i < n; i++) {
      statesArr.push(batch[i].state);
      nextStatesArr.push(batch[i].nextState);
      actions[i] = batch[i].action;
      rewards[i] = batch[i].reward;
      returns[i] = batch[i].returnVal;
      masksArr.push(batch[i].mask);
    }

    var statesT = tf.tensor2d(flattenStates(statesArr, boardSize), [n, boardSize]);
    var nextStatesT = tf.tensor2d(flattenStates(nextStatesArr, boardSize), [n, boardSize]);
    var rewardsT = tf.tensor1d(rewards);
    var returnsT = tf.tensor1d(returns);

    var actionMaskData = new Float32Array(n * boardSize);
    var validMaskData = new Float32Array(n * boardSize);
    for (var i = 0; i < n; i++) {
      actionMaskData[i * boardSize + actions[i]] = 1;
      for (var j = 0; j < boardSize; j++) {
        validMaskData[i * boardSize + j] = masksArr[i][j];
      }
    }
    var actionMaskT = tf.tensor2d(actionMaskData, [n, boardSize]);
    var validMaskT = tf.tensor2d(validMaskData, [n, boardSize]);

    // Dynamics update: supervise predicted (nextState, reward).
    var dynVarList = this.dynamics.trainableWeights.map(function (w) { return w.val; });
    var dynLossT = this.dynamicsOptimizer.minimize(function () {
      var dynIn = tf.concat([statesT, actionMaskT], 1);
      var pred = this.dynamics.predict(dynIn);
      var predNext = pred.slice([0, 0], [-1, boardSize]);
      var predReward = pred.slice([0, boardSize], [-1, 1]).squeeze([1]);
      var nextLoss = predNext.sub(nextStatesT).square().mean();
      var rewardLoss = predReward.sub(rewardsT).square().mean();
      return nextLoss.add(rewardLoss);
    }.bind(this), true, dynVarList);
    var dynLoss = 0;
    if (dynLossT) {
      dynLoss = dynLossT.dataSync()[0];
      dynLossT.dispose();
    }

    // Policy/value update with entropy regularization.
    var varList = this.model.model.trainableWeights.map(function (w) { return w.val; });
    var pvLossT = this.optimizer.minimize(function () {
      var combined = this.model.model.predict(statesT);
      var logits = combined.slice([0, 0], [-1, boardSize]);
      var values = combined.slice([0, boardSize], [-1, 1]).squeeze([1]);
      var maskedLogits = logits.add(validMaskT.sub(1).mul(1e9));
      var probs = maskedLogits.softmax();
      var selected = probs.mul(actionMaskT).sum(1);
      var logp = selected.add(tf.scalar(1e-8)).log();

      var policyLoss = logp.mul(returnsT).mean().neg();
      var valueLoss = returnsT.sub(values).square().mean();
      var entropy = probs.add(tf.scalar(1e-8)).log().mul(probs).sum(1).mean().neg();

      return policyLoss
        .add(valueLoss.mul(tf.scalar(this.valueLossCoeff)))
        .sub(entropy.mul(tf.scalar(this.entropyCoeff)));
    }.bind(this), true, varList);
    var pvLoss = 0;
    if (pvLossT) {
      pvLoss = pvLossT.dataSync()[0];
      pvLossT.dispose();
    }

    statesT.dispose();
    nextStatesT.dispose();
    rewardsT.dispose();
    returnsT.dispose();
    actionMaskT.dispose();
    validMaskT.dispose();

    return dynLoss + pvLoss;
  }

  getTrainSteps() {
    return this.trainSteps;
  }

  getBufferSize() {
    return this.buffer.length;
  }
}
