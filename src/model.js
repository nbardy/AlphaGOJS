import * as tf from '@tensorflow/tfjs';

export class PolicyNetwork {
  constructor(boardSize) {
    this.boardSize = boardSize;
    this.model = this._build();
    this.optimizer = tf.train.adam(0.001);
    this.trainSteps = 0;
  }

  _build() {
    var input = tf.input({ shape: [this.boardSize] });
    var x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input);
    x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);
    var output = tf.layers.dense({ units: this.boardSize, activation: 'softmax' }).apply(x);
    return tf.model({ inputs: input, outputs: output });
  }

  getActions(states, masks) {
    var boardSize = this.boardSize;
    var n = states.length;
    if (n === 0) return [];

    var statesTensor = tf.tensor2d(states);
    var preds = this.model.predict(statesTensor);
    var predsData = preds.dataSync();
    preds.dispose();
    statesTensor.dispose();

    var actions = [];
    for (var i = 0; i < n; i++) {
      var sum = 0;
      var probs = new Float32Array(boardSize);
      for (var j = 0; j < boardSize; j++) {
        probs[j] = predsData[i * boardSize + j] * masks[i][j];
        sum += probs[j];
      }
      if (sum > 1e-8) {
        for (var j = 0; j < boardSize; j++) probs[j] /= sum;
      } else {
        var validCount = 0;
        for (var j = 0; j < boardSize; j++) if (masks[i][j] > 0) validCount++;
        if (validCount > 0) {
          for (var j = 0; j < boardSize; j++) probs[j] = masks[i][j] > 0 ? 1.0 / validCount : 0;
        }
      }
      var r = Math.random();
      var cumSum = 0;
      var action = 0;
      for (var j = 0; j < boardSize; j++) {
        cumSum += probs[j];
        if (r < cumSum) { action = j; break; }
      }
      actions.push(action);
    }
    return actions;
  }

  getAction(state, mask) {
    return this.getActions([state], [mask])[0];
  }

  train(experiences) {
    if (experiences.length === 0) return 0;
    var self = this;
    var boardSize = this.boardSize;
    var batchSize = experiences.length;

    var statesData = [];
    var actionsArr = [];
    var rewardsArr = [];
    for (var i = 0; i < batchSize; i++) {
      statesData.push(experiences[i].state);
      actionsArr.push(experiences[i].action);
      rewardsArr.push(experiences[i].reward);
    }

    var statesTensor = tf.tensor2d(statesData);

    var actionMaskData = new Float32Array(batchSize * boardSize);
    for (var i = 0; i < batchSize; i++) {
      actionMaskData[i * boardSize + actionsArr[i]] = 1;
    }
    var actionMaskTensor = tf.tensor2d(actionMaskData, [batchSize, boardSize]);
    var rewardsTensor = tf.tensor1d(rewardsArr);

    var lossVal = 0;
    try {
      var loss = self.optimizer.minimize(function () {
        var preds = self.model.predict(statesTensor);
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
      console.warn('Training error:', e.message);
    }

    statesTensor.dispose();
    actionMaskTensor.dispose();
    rewardsTensor.dispose();
    self.trainSteps++;
    return lossVal;
  }
}
