import * as tf from '@tensorflow/tfjs';
import { flattenStates, maskedSoftmax, sampleFromProbs } from './action';

// REINFORCE (policy gradient) algorithm.
// Owns optimizer, experience buffer, and training loop.
// Uses model.forward() for the neural network forward pass.
//
// Loss: -mean(log(pi(a|s)) * R) - 0.01 * entropy

export class Reinforce {
  constructor(model) {
    this.model = model;
    this.optimizer = tf.train.adam(0.001);
    this.trainSteps = 0;
    this.buffer = [];
    this.maxBufferSize = 20000;
    this.lastEntropy = 0;
  }

  /**
   * Select actions for a batch of states.
   * @param {Float32Array[]} states
   * @param {Float32Array[]} masks
   * @returns {{ action: number }[]}
   */
  selectActions(states, masks) {
    var boardSize = this.model.boardSize;
    var n = states.length;
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
      // Compute entropy from inference probs (CPU, zero GPU cost)
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

  /**
   * Select a single action (for human play UI).
   */
  selectAction(state, mask) {
    return this.selectActions([state], [mask])[0].action;
  }

  /**
   * Called when a game finishes. Assigns terminal reward to ALL steps (episodic return).
   * @param {{ state: Float32Array, action: number, player: number }[]} trajectory
   * @param {number} winner - 1, -1, or 0 (draw)
   */
  onGameFinished(trajectory, winner) {
    for (var k = 0; k < trajectory.length; k++) {
      var step = trajectory[k];
      var reward = (step.player === winner) ? 1 : (winner === 0 ? 0 : -1);
      this.buffer.push({ state: step.state, action: step.action, reward: reward, mask: step.mask });
    }
    if (this.buffer.length > this.maxBufferSize) {
      this.buffer = this.buffer.slice(-this.maxBufferSize);
    }
  }

  shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize) {
    return gamesSinceLastTrain >= trainInterval && this.buffer.length >= trainBatchSize;
  }

  /**
   * Train on a batch from the buffer.
   * @param {number} batchSize
   * @returns {number} loss value
   */
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

    // Build valid-move mask tensor: invalid moves get logit = -1e9
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
      // Call model.model.predict (the raw keras model) so gradients
      // flow through the weights inside the tape.
      var loss = self.optimizer.minimize(function () {
        // NOTE: do NOT dispose tensors inside minimize() — the gradient tape
        // needs all intermediates alive until backprop completes. TF.js tidy
        // will clean them up automatically after gradients are computed.
        var combined = self.model.model.predict(statesTensor);
        var policyLogits = combined.slice([0, 0], [-1, boardSize]);
        // Mask invalid moves: set their logits to -1e9 before softmax
        // so training matches inference (which uses maskedSoftmax).
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
      console.warn('REINFORCE training error:', e.message);
    }

    statesTensor.dispose();
    actionMaskTensor.dispose();
    rewardsTensor.dispose();
    validMaskTensor.dispose();
    self.trainSteps++;
    return lossVal;
  }

  getTrainSteps() {
    return this.trainSteps;
  }

  getBufferSize() {
    return this.buffer.length;
  }
}
