import * as tf from '@tensorflow/tfjs';
import { flattenStates, maskedSoftmax, sampleFromProbs, logProbOfAction } from './action';

// PPO (Proximal Policy Optimization) with GAE (Generalized Advantage Estimation).
// Owns optimizer, experience buffer, and training loop.
//
// Key differences from REINFORCE:
// - selectActions returns { action, logProb, value } for importance sampling
// - onGameFinished computes GAE advantages per player sub-trajectory
// - train() does K epochs of minibatch updates with clipped surrogate loss
//
// Loss: -clipped_surrogate + 0.5 * value_MSE - 0.01 * entropy
// Hyperparams: epsilon=0.2, gamma=0.99, lambda=0.95, lr=0.0003, K=2, minibatch=128
// Tuned for speed: 2 epochs x 2 minibatches = 4 gradient steps/gen (was 16).

export class PPO {
  constructor(model) {
    this.model = model;
    this.optimizer = tf.train.adam(0.0003);
    this.trainSteps = 0;
    this.buffer = [];
    this.maxBufferSize = 20000;

    // PPO hyperparameters
    this.epsilon = 0.2;
    this.gamma = 0.99;
    this.lambda = 0.95;
    this.epochs = 2;          // Tuned: 2 epochs (was 4) — halves training cost per generation
    this.minibatchSize = 128;  // Tuned: 128 (was 64) — fewer, larger gradient steps
    this.valueLossCoeff = 0.5;
    this.entropyCoeff = 0.01;
    this.lastEntropy = 0;
  }

  /**
   * Select actions for a batch of states.
   * Returns { action, logProb, value } per state for PPO importance sampling.
   */
  selectActions(states, masks) {
    var boardSize = this.model.boardSize;
    var n = states.length;
    if (n === 0) return [];

    var statesTensor = tf.tensor2d(flattenStates(states, boardSize), [n, boardSize]);
    var out = this.model.forward(statesTensor);
    var logitsData = out.policy.dataSync();
    var valuesData = out.value.dataSync();
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
      var lp = Math.log(probs[action] + 1e-8);
      // Compute entropy from inference probs (CPU, zero GPU cost)
      var ent = 0;
      for (var j = 0; j < boardSize; j++) {
        if (probs[j] > 1e-8) ent -= probs[j] * Math.log(probs[j]);
      }
      entropySum += ent;
      results.push({ action: action, logProb: lp, value: valuesData[i] });
    }
    this.lastEntropy = n > 0 ? entropySum / n : 0;
    return results;
  }

  /**
   * Select a single action (for human play UI).
   */
  selectAction(state, mask) {
    return this.selectActions([state], [mask])[0].action;
  }

  /**
   * Called when a game finishes.
   * Splits trajectory by player, assigns terminal reward, computes GAE.
   */
  onGameFinished(trajectory, winner) {
    // Split trajectory by player
    var p1Steps = [];
    var p2Steps = [];
    for (var i = 0; i < trajectory.length; i++) {
      var step = trajectory[i];
      if (step.player === 1) p1Steps.push(step);
      else p2Steps.push(step);
    }

    this._processPlayerTrajectory(p1Steps, winner, 1);
    this._processPlayerTrajectory(p2Steps, winner, -1);
  }

  _processPlayerTrajectory(steps, winner, player) {
    if (steps.length === 0) return;

    var n = steps.length;
    // Terminal reward for last step, 0 for intermediate steps
    var terminalReward = (player === winner) ? 1 : (winner === 0 ? 0 : -1);

    // Build rewards array: 0 for all intermediate, terminal for last
    var rewards = new Float32Array(n);
    rewards[n - 1] = terminalReward;

    // GAE computation
    // advantages[t] = sum_{l=0}^{T-t-1} (gamma*lambda)^l * delta_t+l
    // where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    var advantages = new Float32Array(n);
    var returns = new Float32Array(n);
    var lastGae = 0;

    for (var t = n - 1; t >= 0; t--) {
      var nextValue = (t < n - 1) ? steps[t + 1].value : 0;
      var delta = rewards[t] + this.gamma * nextValue - steps[t].value;
      lastGae = delta + this.gamma * this.lambda * lastGae;
      advantages[t] = lastGae;
      returns[t] = advantages[t] + steps[t].value;
    }

    // Push to buffer
    for (var t = 0; t < n; t++) {
      this.buffer.push({
        state: steps[t].state,
        action: steps[t].action,
        oldLogProb: steps[t].logProb,
        advantage: advantages[t],
        returnVal: returns[t],
        mask: steps[t].mask
      });
    }

    if (this.buffer.length > this.maxBufferSize) {
      this.buffer = this.buffer.slice(-this.maxBufferSize);
    }
  }

  shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize) {
    return gamesSinceLastTrain >= trainInterval && this.buffer.length >= trainBatchSize;
  }

  /**
   * Train with PPO: K epochs of minibatch updates.
   * @param {number} batchSize - total batch to pull from buffer
   * @returns {number} average loss
   */
  train(batchSize) {
    if (this.buffer.length === 0) return 0;

    var batch = this.buffer.splice(0, Math.min(batchSize, this.buffer.length));
    var n = batch.length;
    var boardSize = this.model.boardSize;

    // Extract arrays
    var statesArr = [];
    var actionsArr = new Int32Array(n);
    var oldLogProbsArr = new Float32Array(n);
    var advantagesArr = new Float32Array(n);
    var returnsArr = new Float32Array(n);
    var masksArr = [];

    for (var i = 0; i < n; i++) {
      statesArr.push(batch[i].state);
      actionsArr[i] = batch[i].action;
      oldLogProbsArr[i] = batch[i].oldLogProb;
      advantagesArr[i] = batch[i].advantage;
      returnsArr[i] = batch[i].returnVal;
      masksArr.push(batch[i].mask);
    }

    // Normalize advantages
    var advMean = 0;
    for (var i = 0; i < n; i++) advMean += advantagesArr[i];
    advMean /= n;
    var advVar = 0;
    for (var i = 0; i < n; i++) advVar += (advantagesArr[i] - advMean) * (advantagesArr[i] - advMean);
    advVar /= n;
    var advStd = Math.sqrt(advVar + 1e-8);
    for (var i = 0; i < n; i++) advantagesArr[i] = (advantagesArr[i] - advMean) / advStd;

    var totalLoss = 0;
    var numUpdates = 0;
    var self = this;
    var mbSize = this.minibatchSize;

    // Build full tensors once
    var statesFull = tf.tensor2d(flattenStates(statesArr, boardSize), [n, boardSize]);

    for (var epoch = 0; epoch < this.epochs; epoch++) {
      // Shuffle indices
      var indices = [];
      for (var i = 0; i < n; i++) indices.push(i);
      for (var i = n - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
      }

      // Process minibatches
      for (var start = 0; start + mbSize <= n; start += mbSize) {
        var mbIndices = indices.slice(start, start + mbSize);

        // Gather minibatch data
        var mbActionMask = new Float32Array(mbSize * boardSize);
        var mbOldLogProbs = new Float32Array(mbSize);
        var mbAdvantages = new Float32Array(mbSize);
        var mbReturns = new Float32Array(mbSize);
        var mbGatherIndices = new Int32Array(mbSize);

        var mbValidMask = new Float32Array(mbSize * boardSize);
        for (var mi = 0; mi < mbSize; mi++) {
          var idx = mbIndices[mi];
          mbActionMask[mi * boardSize + actionsArr[idx]] = 1;
          mbOldLogProbs[mi] = oldLogProbsArr[idx];
          mbAdvantages[mi] = advantagesArr[idx];
          mbReturns[mi] = returnsArr[idx];
          mbGatherIndices[mi] = idx;
          for (var mj = 0; mj < boardSize; mj++) {
            mbValidMask[mi * boardSize + mj] = masksArr[idx][mj];
          }
        }

        // Gather states for this minibatch
        var mbStates = tf.gather(statesFull, Array.from(mbGatherIndices));
        var actionMaskT = tf.tensor2d(mbActionMask, [mbSize, boardSize]);
        var oldLogProbsT = tf.tensor1d(mbOldLogProbs);
        var advantagesT = tf.tensor1d(mbAdvantages);
        var returnsT = tf.tensor1d(mbReturns);
        var validMaskT = tf.tensor2d(mbValidMask, [mbSize, boardSize]);

        try {
          var loss = self.optimizer.minimize(function () {
            // NOTE: do NOT dispose tensors inside minimize() — the gradient tape
            // needs all intermediates alive until backprop completes. TF.js tidy
            // will clean them up automatically after gradients are computed.
            var combined = self.model.model.predict(mbStates);
            var policyLogits = combined.slice([0, 0], [-1, boardSize]);
            var valuesPred = combined.slice([0, boardSize], [-1, 1]).squeeze([1]);
            // Mask invalid moves: set their logits to -1e9 before softmax
            // so training matches inference (which uses maskedSoftmax).
            var maskedLogits = policyLogits.add(validMaskT.sub(1).mul(1e9));
            var probs = maskedLogits.softmax();
            var selectedProbs = probs.mul(actionMaskT).sum(1);
            var newLogProbs = selectedProbs.add(tf.scalar(1e-8)).log();

            // PPO clipped surrogate objective
            var ratio = newLogProbs.sub(oldLogProbsT).exp();
            var surr1 = ratio.mul(advantagesT);
            var surr2 = ratio.clipByValue(1 - self.epsilon, 1 + self.epsilon).mul(advantagesT);
            var policyLoss = tf.minimum(surr1, surr2).mean().neg();

            // Value loss (MSE)
            var valueLoss = returnsT.sub(valuesPred).square().mean();

            // Entropy bonus (only over valid moves, already masked)
            var entropy = probs.add(tf.scalar(1e-8)).log().mul(probs).sum(1).mean().neg();

            return policyLoss
              .add(valueLoss.mul(tf.scalar(self.valueLossCoeff)))
              .sub(entropy.mul(tf.scalar(self.entropyCoeff)));
          }, true);

          if (loss) {
            totalLoss += loss.dataSync()[0];
            loss.dispose();
            numUpdates++;
          }
        } catch (e) {
          console.warn('PPO training error:', e.message);
        }

        mbStates.dispose();
        actionMaskT.dispose();
        oldLogProbsT.dispose();
        advantagesT.dispose();
        returnsT.dispose();
        validMaskT.dispose();
      }
    }

    statesFull.dispose();
    self.trainSteps++;
    return numUpdates > 0 ? totalLoss / numUpdates : 0;
  }

  getTrainSteps() {
    return this.trainSteps;
  }

  getBufferSize() {
    return this.buffer.length;
  }
}
