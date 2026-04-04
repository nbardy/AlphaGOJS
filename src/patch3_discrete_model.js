import * as tf from '@tensorflow/tfjs';

// 3×3-periodic joint embedding **per cell** (overlapping / tiled phase): index =
// ((r%3)*3+(c%3))*4 + code, vocab 36. Full H×W conv trunk — not patch-compressed.
// For true 3×3 **non-overlapping patch tokens** + coarse conv, use Patch3TokenDiscreteModel (`patch3_token`).
//
// expectsDiscreteInput: trajectory / batches store Int32Array raw codes 0..3 per cell.

export class Patch3DiscreteModel {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.boardSize = rows * cols;
    this.patchSize = 3;
    this.jointVocab = this.patchSize * this.patchSize * 4;
    this.expectsDiscreteInput = true;
    this.embedDim = 32;
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
    var input = tf.input({ shape: [this.boardSize], dtype: 'int32' });
    var embedded = tf.layers
      .embedding({
        inputDim: this.jointVocab,
        outputDim: this.embedDim,
        embeddingsInitializer: 'glorotUniform'
      })
      .apply(input);

    var x = tf.layers
      .reshape({ targetShape: [this.rows, this.cols, this.embedDim] })
      .apply(embedded);

    x = this._sepBlock(x, 32);
    x = this._sepBlock(x, 32);

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
   * @param {tf.Tensor2D} statesTensor - int32 [batch, boardSize], joint indices 0..35
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
