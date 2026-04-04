import * as tf from '@tensorflow/tfjs';

// True non-overlapping 3×3 patches: each patch → 9 joint indices (k*4+code) → embed →
// average-pool the 9 vectors → one token per patch. Convs run on ⌈H/3⌉×⌈W/3⌉ grid (~1/9 cells).
// Policy: 1×1 logits on coarse map, upsample ×3, crop to original H×W, flatten.
//
// Input to TF graph: int32 [batch, hp*wp*9] packed patch-major (see flattenPatchTokenJointInput in action.js).
// expectsDiscreteInput + patchTokenInput: trajectories still store raw Int32Array codes per cell.

export class Patch3TokenDiscreteModel {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.boardSize = rows * cols;
    this.patchSize = 3;
    this.hp = Math.ceil(rows / this.patchSize);
    this.wp = Math.ceil(cols / this.patchSize);
    this.padRows = this.hp * this.patchSize - rows;
    this.padCols = this.wp * this.patchSize - cols;
    this.patchInputLen = this.hp * this.wp * 9;
    this.jointVocab = this.patchSize * this.patchSize * 4;
    this.expectsDiscreteInput = true;
    this.patchTokenInput = true;
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
    var hp = this.hp;
    var wp = this.wp;
    var pr = this.padRows;
    var pc = this.padCols;

    var input = tf.input({ shape: [this.patchInputLen], dtype: 'int32' });
    var seq = tf.layers.reshape({ targetShape: [hp * wp, 9] }).apply(input);
    var embedded = tf.layers
      .embedding({
        inputDim: this.jointVocab,
        outputDim: this.embedDim,
        embeddingsInitializer: 'glorotUniform'
      })
      .apply(seq);

    var pooled = tf.layers
      .averagePooling2d({ poolSize: [1, 9], strides: [1, 9], padding: 'valid' })
      .apply(embedded);
    var vecPerPatch = tf.layers.reshape({ targetShape: [hp * wp, this.embedDim] }).apply(pooled);
    var x = tf.layers.reshape({ targetShape: [hp, wp, this.embedDim] }).apply(vecPerPatch);

    x = this._sepBlock(x, 32);
    x = this._sepBlock(x, 32);

    var policyCoarse = tf.layers
      .conv2d({
        filters: 1,
        kernelSize: 1,
        padding: 'same',
        activation: 'linear',
        useBias: true
      })
      .apply(x);
    var policyUp = tf.layers.upSampling2d({ size: [3, 3] }).apply(policyCoarse);
    var policyCrop =
      pr > 0 || pc > 0
        ? tf.layers.cropping2D({ cropping: [[0, pr], [0, pc]] }).apply(policyUp)
        : policyUp;
    var policy = tf.layers.flatten().apply(policyCrop);

    var vGap = tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }).apply(x);
    var vDense = tf.layers.dense({ units: 64, activation: 'relu' }).apply(vGap);
    var value = tf.layers.dense({ units: 1, activation: 'tanh' }).apply(vDense);

    var combined = tf.layers.concatenate().apply([policy, value]);
    return tf.model({ inputs: input, outputs: combined });
  }

  /**
   * @param {tf.Tensor2D} statesTensor - int32 [batch, patchInputLen] joint indices per patch cell
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
