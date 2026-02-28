import * as tf from '@tensorflow/tfjs';

// Spatial (convolutional) model for the plague game.
// Pure forward pass — no optimizer, no training logic.
//
// Architecture:
//   Input [boardSize] → reshape [rows, cols, 1]
//   → Conv2D(32, 3x3, same, relu)
//   → 3x residual blocks (conv+conv+skip)
//   → Policy head: conv 1x1 → flatten → dense(boardSize)
//   → Value head: conv 1x1 → flatten → dense(64, relu) → dense(1, tanh)
//
// Output concatenated as [boardSize + 1] same as DenseModel.

export class SpatialModel {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.boardSize = rows * cols;
    this.model = this._build();
  }

  _resBlock(x, filters) {
    var conv1 = tf.layers.conv2d({
      filters: filters, kernelSize: 3, padding: 'same', activation: 'relu'
    }).apply(x);
    var conv2 = tf.layers.conv2d({
      filters: filters, kernelSize: 3, padding: 'same'
    }).apply(conv1);
    // Skip connection: element-wise add then relu
    var added = tf.layers.add().apply([x, conv2]);
    return tf.layers.activation({ activation: 'relu' }).apply(added);
  }

  _build() {
    var input = tf.input({ shape: [this.boardSize] });

    // Reshape flat input to spatial grid
    var reshaped = tf.layers.reshape({
      targetShape: [this.rows, this.cols, 1]
    }).apply(input);

    // Initial convolution
    var x = tf.layers.conv2d({
      filters: 32, kernelSize: 3, padding: 'same', activation: 'relu'
    }).apply(reshaped);

    // 3 residual blocks
    x = this._resBlock(x, 32);
    x = this._resBlock(x, 32);
    x = this._resBlock(x, 32);

    // Policy head: 1x1 conv → flatten → dense(boardSize) raw logits
    var pConv = tf.layers.conv2d({
      filters: 2, kernelSize: 1, padding: 'same', activation: 'relu'
    }).apply(x);
    var pFlat = tf.layers.flatten().apply(pConv);
    var policy = tf.layers.dense({ units: this.boardSize }).apply(pFlat);

    // Value head: 1x1 conv → flatten → dense(64, relu) → dense(1, tanh)
    var vConv = tf.layers.conv2d({
      filters: 1, kernelSize: 1, padding: 'same', activation: 'relu'
    }).apply(x);
    var vFlat = tf.layers.flatten().apply(vConv);
    var vDense = tf.layers.dense({ units: 64, activation: 'relu' }).apply(vFlat);
    var value = tf.layers.dense({ units: 1, activation: 'tanh' }).apply(vDense);

    // Concatenate into single output: [batch, boardSize + 1]
    var combined = tf.layers.concatenate().apply([policy, value]);

    return tf.model({ inputs: input, outputs: combined });
  }

  /**
   * Forward pass through the model.
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
