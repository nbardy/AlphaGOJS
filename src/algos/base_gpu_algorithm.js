// Abstract GPU algorithm contract.
// Concrete implementations may wrap existing CPU algorithms while the game loop
// runs on GPU, or own full GPU-native train/infer logic.

export class BaseGPUAlgorithm {
  constructor(model, config) {
    this.model = model;
    this.config = config || {};
  }

  selectActions(states, masks) {
    throw new Error('BaseGPUAlgorithm.selectActions must be implemented');
  }

  onGameFinished(trajectory, winner) {
    throw new Error('BaseGPUAlgorithm.onGameFinished must be implemented');
  }

  shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize) {
    throw new Error('BaseGPUAlgorithm.shouldTrain must be implemented');
  }

  train(batchSize) {
    throw new Error('BaseGPUAlgorithm.train must be implemented');
  }

  getBufferSize() {
    throw new Error('BaseGPUAlgorithm.getBufferSize must be implemented');
  }

  getTrainSteps() {
    return 0;
  }

  selectAction(state, mask) {
    throw new Error('BaseGPUAlgorithm.selectAction must be implemented');
  }
}
