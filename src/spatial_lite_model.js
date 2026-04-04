import * as tf from '@tensorflow/tfjs';

// Faster spatial inductive bias than spatial_model.js:
//
// - Backbone: 2× separable 3×3 (depthwise + pointwise) instead of 7× full 3×3 convs.
//   Per layer MACs scale ~like 9·C + C² vs 9·C² for standard conv at the same H×W.
// - Policy: 1×1 conv with **1 filter** → one logit per cell; flatten matches row-major
//   board index (same as getBoardForNN). Avoids the old  (2·H·W)×boardSize dense head.
// - Value: globalAveragePooling2D → small MLP (no 400-wide flatten in the value path).
//
// Still use PPO + masked softmax (see 3e832bf) for stable convergence.

export class SpatialLiteModel {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.boardSize = rows * cols;
    this.model = this._build();
  }

  _sepBlock(x, filters) {
    return tf.layers
      .separableConv2d({
        filters: filters,
        kernelSize: 3,
        padding: 'same',
        depthMultiplier: 1,
        activation: 'relu',
        useBias: true,
        depthwiseInitializer: 'glorotUniform',
        pointwiseInitializer: 'glorotUniform'
      })
      .apply(x);
  }

  _build() {
    var input = tf.input({ shape: [this.boardSize] });
    var reshaped = tf.layers.reshape({
      targetShape: [this.rows, this.cols, 1]
    }).apply(input);

    var x = this._sepBlock(reshaped, 32);
    x = this._sepBlock(x, 32);

    // One scalar logit per board cell (action = cell index).
    var policy = tf.layers
      .conv2d({
        filters: 1,
        kernelSize: 1,
        padding: 'same',
        activation: 'linear',
        useBias: true
      })
      .apply(x);
    policy = tf.layers.flatten().apply(policy);

    var vGap = tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }).apply(x);
    var vDense = tf.layers.dense({ units: 64, activation: 'relu' }).apply(vGap);
    var value = tf.layers.dense({ units: 1, activation: 'tanh' }).apply(vDense);

    var combined = tf.layers.concatenate().apply([policy, value]);
    return tf.model({ inputs: input, outputs: combined });
  }

  /**
   * @param {tf.Tensor2D} statesTensor - shape [batch, boardSize]
   * @returns {{ policy: tf.Tensor2D, value: tf.Tensor1D }}
   */
  forward(statesTensor) {
    var combined = this.model.predict(statesTensor);
    var policy = combined.slice([0, 0], [-1, this.boardSize]);
    var value = combined.slice([0, this.boardSize], [-1, 1]).squeeze([1]);
    combined.dispose();
    return { policy: policy, value: value };
  }
}
