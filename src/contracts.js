// Lightweight runtime contracts for hot-swappable model/algorithm components.
// JS has no native abstract classes; these guards fail fast on bad wiring.

export function assertModelContract(model) {
  if (!model) throw new Error('Model is required');
  if (typeof model.forward !== 'function') {
    throw new Error('Model must implement forward(statesTensor)');
  }
  if (typeof model.boardSize !== 'number' || model.boardSize <= 0) {
    throw new Error('Model must expose numeric boardSize');
  }
  if (!model.model || typeof model.model.predict !== 'function') {
    throw new Error('Model must expose tfjs model at model.model');
  }
}

export function assertAlgorithmContract(algo) {
  if (!algo) throw new Error('Algorithm is required');
  if (typeof algo.selectActions !== 'function') {
    throw new Error('Algorithm must implement selectActions(states, masks)');
  }
  if (typeof algo.onGameFinished !== 'function') {
    throw new Error('Algorithm must implement onGameFinished(trajectory, winner)');
  }
  if (typeof algo.shouldTrain !== 'function') {
    throw new Error('Algorithm must implement shouldTrain(gamesSinceLastTrain, trainInterval, trainBatchSize)');
  }
  if (typeof algo.train !== 'function') {
    throw new Error('Algorithm must implement train(batchSize)');
  }
  if (typeof algo.getBufferSize !== 'function') {
    throw new Error('Algorithm must implement getBufferSize()');
  }
}
