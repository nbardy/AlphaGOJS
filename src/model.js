import * as tf from '@tensorflow/tfjs';

export class PolicyNetwork {
  constructor(boardSize, nnInputSize) {
    this.boardSize = boardSize;       // action space (number of cells)
    this.inputSize = nnInputSize || boardSize; // NN input dim (may be larger for multi-channel)
    this.policyModel = this._buildPolicy();
    this.valueModel = this._buildValue();
    this.policyOptimizer = tf.train.adam(0.0003);
    this.valueOptimizer = tf.train.adam(0.001);
    this.trainSteps = 0;
    this.snapshots = [];
  }

  _buildPolicy() {
    var input = tf.input({ shape: [this.inputSize] });
    var x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input);
    x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(x);
    var output = tf.layers.dense({ units: this.boardSize, activation: 'softmax' }).apply(x);
    return tf.model({ inputs: input, outputs: output });
  }

  _buildValue() {
    var input = tf.input({ shape: [this.inputSize] });
    var x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(input);
    x = tf.layers.dense({ units: 64, activation: 'relu' }).apply(x);
    var output = tf.layers.dense({ units: 1, activation: 'tanh' }).apply(x);
    return tf.model({ inputs: input, outputs: output });
  }

  _flattenStates(states) {
    var n = states.length;
    var is = this.inputSize;
    var flat = new Float32Array(n * is);
    for (var i = 0; i < n; i++) flat.set(states[i], i * is);
    return flat;
  }

  getActions(states, masks) {
    var boardSize = this.boardSize;
    var n = states.length;
    if (n === 0) return { actions: [], values: [] };

    var statesTensor = tf.tensor2d(this._flattenStates(states), [n, this.inputSize]);
    var preds = this.policyModel.predict(statesTensor);
    var predsData = preds.dataSync();
    var vals = this.valueModel.predict(statesTensor);
    var valsData = vals.dataSync();
    preds.dispose();
    vals.dispose();
    statesTensor.dispose();

    var actions = [];
    var values = [];
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
        var vc = 0;
        for (var j = 0; j < boardSize; j++) if (masks[i][j] > 0) vc++;
        if (vc > 0) for (var j = 0; j < boardSize; j++) probs[j] = masks[i][j] > 0 ? 1.0 / vc : 0;
      }
      var r = Math.random();
      var cumSum = 0;
      var action = 0;
      for (var j = 0; j < boardSize; j++) {
        cumSum += probs[j];
        if (r < cumSum) { action = j; break; }
      }
      actions.push(action);
      values.push(valsData[i]);
    }
    return { actions: actions, values: values };
  }

  getAction(state, mask) {
    var result = this.getActions([state], [mask]);
    return result.actions[0];
  }

  train(experiences) {
    if (experiences.length === 0) return 0;
    var self = this;
    var boardSize = this.boardSize;
    var batchSize = experiences.length;

    var statesFlat = this._flattenStates(experiences.map(function (e) { return e.state; }));
    var actionsArr = experiences.map(function (e) { return e.action; });
    var returnsArr = experiences.map(function (e) { return e.reward; });
    var oldValuesArr = experiences.map(function (e) { return e.value || 0; });

    var statesTensor = tf.tensor2d(statesFlat, [batchSize, this.inputSize]);
    var returnsTensor = tf.tensor1d(returnsArr);

    // Compute advantages: return - V(s) at time of play (detached)
    var advantagesArr = new Float32Array(batchSize);
    for (var i = 0; i < batchSize; i++) advantagesArr[i] = returnsArr[i] - oldValuesArr[i];
    var advantagesTensor = tf.tensor1d(advantagesArr);

    var actionMaskData = new Float32Array(batchSize * boardSize);
    for (var i = 0; i < batchSize; i++) actionMaskData[i * boardSize + actionsArr[i]] = 1;
    var actionMaskTensor = tf.tensor2d(actionMaskData, [batchSize, boardSize]);

    var policyLossVal = 0;
    try {
      var pLoss = self.policyOptimizer.minimize(function () {
        var preds = self.policyModel.predict(statesTensor);
        var selectedProbs = preds.mul(actionMaskTensor).sum(1);
        var logProbs = selectedProbs.add(tf.scalar(1e-8)).log();
        var policyLoss = logProbs.mul(advantagesTensor).mean().neg();
        var entropy = preds.add(tf.scalar(1e-8)).log().mul(preds).sum(1).mean().neg();
        return policyLoss.sub(entropy.mul(tf.scalar(0.01)));
      }, true);
      if (pLoss) { policyLossVal = pLoss.dataSync()[0]; pLoss.dispose(); }
    } catch (e) { console.warn('Policy train error:', e.message); }

    try {
      self.valueOptimizer.minimize(function () {
        var values = self.valueModel.predict(statesTensor).squeeze([1]);
        return returnsTensor.sub(values).square().mean();
      });
    } catch (e) { console.warn('Value train error:', e.message); }

    statesTensor.dispose();
    returnsTensor.dispose();
    advantagesTensor.dispose();
    actionMaskTensor.dispose();
    self.trainSteps++;
    return policyLossVal;
  }

  saveSnapshot(gen) {
    var weights = this.policyModel.getWeights();
    var snapshot = { gen: gen, weights: [] };
    for (var i = 0; i < weights.length; i++) {
      snapshot.weights.push({ data: Array.from(weights[i].dataSync()), shape: weights[i].shape });
    }
    this.snapshots.push(snapshot);
    if (this.snapshots.length > 20) this.snapshots.shift();
  }

  loadSnapshotInto(snapshot, targetModel) {
    var tensors = [];
    for (var i = 0; i < snapshot.weights.length; i++) {
      var w = snapshot.weights[i];
      tensors.push(tf.tensor(w.data, w.shape));
    }
    targetModel.setWeights(tensors);
    for (var i = 0; i < tensors.length; i++) tensors[i].dispose();
  }
}
