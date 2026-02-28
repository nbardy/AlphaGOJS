import * as tf from '@tensorflow/tfjs';

// Dense (MLP) model for the plague game.
// Pure forward pass — no optimizer, no training logic.
// Architecture: [boardSize] → Dense(256,relu) → Dense(128,relu) → policy(boardSize) + value(1)
//
// Policy + value are concatenated into a single output and split in forward().
// Output layout: [boardSize logits, 1 value] = boardSize + 1 total units.

export class DenseModel {
  constructor(rows, cols) {
    this.boardSize = rows * cols;
    this.model = this._build();
  }

  _build() {
    var input = tf.input({ shape: [this.boardSize] });
    var x = tf.layers.dense({ units: 256, activation: 'relu' }).apply(input);
    x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(x);

    // Policy head: raw logits (no activation) for maskedSoftmax in algorithm
    var policy = tf.layers.dense({ units: this.boardSize }).apply(x);

    // Value head: tanh activation, scalar per sample
    var value = tf.layers.dense({ units: 1, activation: 'tanh' }).apply(x);

    // Concatenate into single output: [batch, boardSize + 1]
    var combined = tf.layers.concatenate().apply([policy, value]);

    return tf.model({ inputs: input, outputs: combined });
  }

  /**
   * Forward pass through the model.
   * @param {tf.Tensor2D} statesTensor - shape [batch, boardSize]
   * @returns {{ policy: tf.Tensor2D, value: tf.Tensor1D }}
   *   policy: raw logits [batch, boardSize], value: scalar estimates [batch]
   */
  forward(statesTensor) {
    var combined = this.model.predict(statesTensor);
    // Split: first boardSize columns = policy logits, last column = value
    var policy = combined.slice([0, 0], [-1, this.boardSize]);
    var value = combined.slice([0, this.boardSize], [-1, 1]).squeeze([1]);
    combined.dispose();
    return { policy: policy, value: value };
  }
}
