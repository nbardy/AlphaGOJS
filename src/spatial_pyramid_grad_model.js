import * as tf from '@tensorflow/tfjs';

// Compact spatial policy/value net with:
// - Learned neighborhood stem: depthwise 3×3 (depth multiplier 4) → 1×1 to 2 channels,
//   concatenated with the raw board (3 input channels total).
// - Low-width separable backbone with dilation 2 on full grid for multi-scale context.
// - Two-level average pyramid: pooled maps are processed lightly, then global-pooled;
//   summaries are fused into a channel gate (sigmoid) applied to full-resolution features.
//
// Output contract matches SpatialLiteModel: concat [policy logits boardSize, value 1].

export class SpatialPyramidGradModel {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.boardSize = rows * cols;
    this._channels = 20;
    this.model = this._build();
  }

  _neighborStem(reshaped) {
    var dw = tf.layers
      .depthwiseConv2d({
        kernelSize: 3,
        depthMultiplier: 4,
        padding: 'same',
        activation: 'linear',
        useBias: true,
        depthwiseInitializer: 'glorotUniform',
        biasInitializer: 'zeros'
      })
      .apply(reshaped);
    var edge = tf.layers
      .conv2d({
        filters: 2,
        kernelSize: 1,
        padding: 'same',
        activation: 'linear',
        useBias: true
      })
      .apply(dw);
    return tf.layers.concatenate().apply([reshaped, edge]);
  }

  _sepDilated(x, dilationRate) {
    var c = this._channels;
    return tf.layers
      .separableConv2d({
        filters: c,
        kernelSize: 3,
        padding: 'same',
        depthMultiplier: 1,
        dilationRate: dilationRate,
        activation: 'relu',
        useBias: true,
        depthwiseInitializer: 'glorotUniform',
        pointwiseInitializer: 'glorotUniform'
      })
      .apply(x);
  }

  _sepCoarse(x) {
    var c = this._channels;
    return tf.layers
      .separableConv2d({
        filters: c,
        kernelSize: 3,
        padding: 'same',
        depthMultiplier: 1,
        dilationRate: 1,
        activation: 'relu',
        useBias: true,
        depthwiseInitializer: 'glorotUniform',
        pointwiseInitializer: 'glorotUniform'
      })
      .apply(x);
  }

  _pyramidGate(x) {
    var c = this._channels;
    var pool = { poolSize: 2, strides: 2, padding: 'same' };
    var p1 = tf.layers.averagePooling2d(pool).apply(x);
    var p2 = tf.layers.averagePooling2d(pool).apply(p1);
    var e1 = this._sepCoarse(p1);
    var e2 = this._sepCoarse(p2);
    var gap = { dataFormat: 'channelsLast' };
    var g1 = tf.layers.globalAveragePooling2d(gap).apply(e1);
    var g2 = tf.layers.globalAveragePooling2d(gap).apply(e2);
    var gcat = tf.layers.concatenate().apply([g1, g2]);
    var h = tf.layers.dense({ units: c, activation: 'relu' }).apply(gcat);
    var gate = tf.layers.dense({ units: c, activation: 'sigmoid' }).apply(h);
    var gate4d = tf.layers.reshape({ targetShape: [1, 1, c] }).apply(gate);
    return tf.layers.multiply().apply([x, gate4d]);
  }

  _build() {
    var c = this._channels;
    var input = tf.input({ shape: [this.boardSize] });
    var reshaped = tf.layers
      .reshape({ targetShape: [this.rows, this.cols, 1] })
      .apply(input);

    var stem = this._neighborStem(reshaped);
    var x = this._sepDilated(stem, 1);
    x = this._sepDilated(x, 2);
    x = this._pyramidGate(x);

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
    var vDense = tf.layers.dense({ units: 48, activation: 'relu' }).apply(vGap);
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
