import { GPUWorkerTrainerProxy } from './gpu_worker_trainer_proxy';

function numOpt(v, fallback) {
  return typeof v === 'number' ? v : fallback;
}

function boolOpt(v, fallback) {
  return typeof v === 'boolean' ? v : fallback;
}

function queueSoftCapFractionOpt(runtimeOptions) {
  if (typeof runtimeOptions.queueSoftCapFraction === 'number') return runtimeOptions.queueSoftCapFraction;
  return runtimeOptions.pipelineTypeOverride === 'full_gpu_resident' ? 0.75 : 1.0;
}

export function createGPUWorkerPipeline(modelType, algoType, rows, cols, numGames, gameType, checkpointPoolConfig, runtimeOptions) {
  runtimeOptions = runtimeOptions || {};
  var worker;
  try {
    worker = new Worker(new URL('./workers/gpu_owner.worker.js', import.meta.url), { type: 'module' });
  } catch (e) {
    throw new Error('Unable to start GPU worker pipeline: ' + e.message);
  }

  // Spread runtimeOptions first so bench flags (benchLoopMode, benchInstrument, …) reach the worker.
  var trainer = new GPUWorkerTrainerProxy(worker, Object.assign({}, runtimeOptions, {
    modelType: modelType,
    algoType: algoType,
    rows: rows,
    cols: cols,
    numGames: numGames,
    gameType: gameType,
    checkpointPool: checkpointPoolConfig || {},
    snapshotEveryTicks: numOpt(runtimeOptions.snapshotEveryTicks, 2),
    maxTickBatch: numOpt(runtimeOptions.maxTickBatch, 8),
    maxQueuedSteps: numOpt(runtimeOptions.maxQueuedSteps, 4096),
    queueSoftCapFraction: queueSoftCapFractionOpt(runtimeOptions),
    pauseTicksWhenTraining: boolOpt(runtimeOptions.pauseTicksWhenTraining, false),
    trainInFlightQueueCap: numOpt(runtimeOptions.trainInFlightQueueCap, 0),
    trainBatchSize: numOpt(runtimeOptions.trainBatchSize, 512),
    trainInterval: numOpt(runtimeOptions.trainInterval, 30),
    uiSnapshotMaxGames: numOpt(runtimeOptions.uiSnapshotMaxGames, 48)
  }));

  // For UI human-play path, expose async action selection via `algo || trainer`.
  return {
    trainer: trainer,
    algo: trainer,
    pool: null,
    pipelineType: runtimeOptions.pipelineTypeOverride || 'gpu_worker'
  };
}
