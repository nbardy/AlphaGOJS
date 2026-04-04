import * as tf from '@tensorflow/tfjs';
import { BaseGPUAlgorithm } from './base_gpu_algorithm';
import { maskedSoftmax, sampleFromProbs, statesRowsToModelInputTensor } from '../action';

// Adapter: wraps an existing CPU-style algorithm (PPO/PPG/SAC/etc.)
// so it can be orchestrated by the GPU pipeline.
//
// The adapter delegates trajectory ingestion + training to the wrapped algo.
// Action selection for the live GPU loop is handled by the orchestrator.

export class GPUAlgoAdapter extends BaseGPUAlgorithm {
  constructor(algo, model, config) {
    super(model, config);
    this.algo = algo;
  }

  selectActions(states, masks) {
    if (typeof this.algo.selectActions === 'function') {
      return this.algo.selectActions(states, masks);
    }
    var out = [];
    for (var i = 0; i < states.length; i++) {
      out.push({ action: this.selectAction(states[i], masks[i]) });
    }
    return out;
  }

  onGameFinished(trajectory, winner) {
    this.algo.onGameFinished(trajectory, winner);
  }

  shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize) {
    return this.algo.shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize);
  }

  train(batchSize) {
    return this.algo.train(batchSize);
  }

  getBufferSize() {
    return this.algo.getBufferSize();
  }

  getTrainSteps() {
    return this.algo.getTrainSteps ? this.algo.getTrainSteps() : 0;
  }

  dispose() {
    if (this.algo && typeof this.algo.dispose === 'function') {
      this.algo.dispose();
    }
  }

  selectAction(state, mask) {
    if (typeof this.algo.selectAction === 'function') {
      return this.algo.selectAction(state, mask);
    }

    // Generic fallback using wrapped model if an algorithm doesn't expose selectAction.
    var boardSize = this.model.boardSize;
    var statesTensor = statesRowsToModelInputTensor(this.model, [state], 1);
    var out = this.model.forward(statesTensor);
    var logitsData = out.policy.dataSync();
    out.policy.dispose();
    out.value.dispose();
    statesTensor.dispose();
    var logits = new Float32Array(boardSize);
    for (var j = 0; j < boardSize; j++) logits[j] = logitsData[j];
    var probs = maskedSoftmax(logits, mask);
    return sampleFromProbs(probs);
  }
}
