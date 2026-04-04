import * as tf from '@tensorflow/tfjs';
import { serialization } from '@tensorflow/tfjs-core';
import { Dense } from '@tensorflow/tfjs-layers/dist/layers/core.js';
import {
  registerPackedU8DenseReluKernels,
  packedU8DenseRelu,
  packVectorU8ToInt32Array,
  packWeightsU8RowMajorInt32,
} from './tfjs_packed_u8_dense_relu.js';

let kernelsEnsured = false;

function ensureKernels() {
  if (!kernelsEnsured) {
    registerPackedU8DenseReluKernels();
    kernelsEnsured = true;
  }
}

function quantizeRowToPackedU8(row, inDim) {
  var mn = Infinity;
  var mx = -Infinity;
  var i;
  for (i = 0; i < inDim; i++) {
    var v = row[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  var range = Math.max(mx - mn, 1e-6);
  var u8 = new Uint8Array(inDim);
  for (i = 0; i < inDim; i++) {
    u8[i] = Math.max(0, Math.min(255, Math.round(((row[i] - mn) / range) * 255)));
  }
  return {
    packed: packVectorU8ToInt32Array(u8),
    scale: range / 255,
    zero: mn,
  };
}

function packReluKernelToU8(kData, inDim, outDim) {
  var inPacked = Math.ceil(inDim / 4);
  var u8rows = new Uint8Array(outDim * inDim);
  var scalePerOut = new Float32Array(outDim);
  var j;
  var i;
  for (j = 0; j < outDim; j++) {
    var mn = Infinity;
    var mx = -Infinity;
    for (i = 0; i < inDim; i++) {
      var v = kData[i * outDim + j];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (mx < mn) mx = mn;
    var range = Math.max(mx - mn, 1e-6);
    scalePerOut[j] = range / 255;
    for (i = 0; i < inDim; i++) {
      var wv = kData[i * outDim + j];
      u8rows[j * inDim + i] = Math.max(0, Math.min(255, Math.round(((wv - mn) / range) * 255)));
    }
  }
  var int32Packed = packWeightsU8RowMajorInt32(u8rows, outDim, inDim);
  return { int32Packed: int32Packed, scalePerOut: scalePerOut, inPacked: inPacked };
}

/**
 * Forward: quantized packed u8 matmul + ReLU (approximate).
 * Gradients: straight-through estimator as dense ReLU w.r.t. float weights.
 */
function packedDenseReluFromFloatBatch(x, kern, bias, inDim, outDim) {
  var batch = x.shape[0];
  var xData = x.dataSync();
  var bData = bias.dataSync();
  var kRelu = tf.relu(kern);
  var kData = kRelu.dataSync();
  kRelu.dispose();

  var pw = packReluKernelToU8(kData, inDim, outDim);
  var wTensor = tf.tensor2d(Array.from(pw.int32Packed), [outDim, pw.inPacked], 'int32');

  var rows = [];
  var bi;
  for (bi = 0; bi < batch; bi++) {
    var row = new Float32Array(inDim);
    row.set(xData.subarray(bi * inDim, (bi + 1) * inDim));
    var qx = quantizeRowToPackedU8(row, inDim);
    var sx = qx.scale;
    var xp = tf.tensor1d(Array.from(qx.packed), 'int32');
    var scaleArr = new Float32Array(outDim);
    var j;
    for (j = 0; j < outDim; j++) {
      scaleArr[j] = sx * pw.scalePerOut[j];
    }
    var scales = tf.tensor1d(scaleArr, 'float32');
    var bt = tf.tensor1d(Array.from(bData), 'float32');
    var yi = packedU8DenseRelu(xp, wTensor, scales, bt, inDim, outDim);
    rows.push(yi.expandDims(0));
    tf.dispose([xp, scales, bt, yi]);
  }
  wTensor.dispose();
  return tf.concat(rows, 0);
}

/** Extends Dense (linear + bias) but replaces matmul+ReLU forward with packed u8 + STE grads. */
class PackedU8DenseReluSTE extends Dense {
  constructor(args) {
    super({
      units: args.units,
      activation: 'linear',
      useBias: true,
      kernelInitializer: args.kernelInitializer || 'glorotNormal',
      biasInitializer: args.biasInitializer || 'zeros',
    });
  }

  call(inputs, kwargs) {
    var self = this;
    return tf.tidy(function () {
      self.invokeCallHook(inputs, kwargs);
      var x = Array.isArray(inputs) ? inputs[0] : inputs;
      var k = self.kernel.read();
      var b = self.bias.read();
      var inDim = x.shape[1];
      var outDim = self.units;

      var op = tf.customGrad(function (inp, kern, bi, save) {
        var pre = tf.matMul(inp, kern).add(bi);
        save([inp, kern, bi, pre]);
        var yPacked = packedDenseReluFromFloatBatch(inp, kern, bi, inDim, outDim);
        return {
          value: yPacked,
          gradFunc: function (dy, saved) {
            var inpT = saved[0];
            var kernT = saved[1];
            var preT = saved[3];
            var mask = tf.cast(tf.greater(preT, 0), 'float32');
            var dyb = tf.mul(dy, mask);
            return [
              tf.matMul(dyb, kernT, false, true),
              tf.matMul(inpT, dyb, true, false),
              tf.sum(dyb, 0),
            ];
          },
        };
      });
      return op(x, k, b);
    });
  }
}

PackedU8DenseReluSTE.className = 'PackedU8DenseReluSTE';
serialization.registerClass(PackedU8DenseReluSTE);

/**
 * MLP with one packed-u8 + STE hidden block after a standard ReLU projection.
 * Forward actually runs PackedU8DenseRelu; training uses dense ReLU gradients (STE).
 */
export class PackedU8DenseModel {
  constructor(rows, cols) {
    ensureKernels();
    this.rows = rows;
    this.cols = cols;
    this.boardSize = rows * cols;
    this.model = this._build();
  }

  _build() {
    var input = tf.input({ shape: [this.boardSize] });
    var x = tf.layers.dense({ units: 128, activation: 'relu' }).apply(input);
    x = new PackedU8DenseReluSTE({ units: 128 }).apply(x);
    var policy = tf.layers.dense({ units: this.boardSize }).apply(x);
    var value = tf.layers.dense({ units: 1, activation: 'tanh' }).apply(x);
    var combined = tf.layers.concatenate().apply([policy, value]);
    return tf.model({ inputs: input, outputs: combined });
  }

  /**
   * @param {tf.Tensor2D} statesTensor
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
