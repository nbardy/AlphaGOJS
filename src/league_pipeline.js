import { listModelTypes } from './model_registry';
import { createGPUWorkerPipeline } from './nextgen/create_gpu_worker_pipeline';
import { resolveRuntimeSpec } from './runtime/runtime_registry';

// League: slightly more checkpoint games than the main app — rollouts are cheap vs PPO, and past
// selves stabilize non-stationary self-play. trainInterval is higher so we collect more on-policy
// games before each (expensive) train step.
export var LEAGUE_CHECKPOINT_POOL_CONFIG = {
  maxCheckpoints: 50,
  recentWindow: 50,
  sampleMode: 'uniform_recent',
  checkpointFraction: 0.38,
  saveInterval: 15
};

/**
 * GPU worker only: train every registered model type in parallel with cross-arch league Elo.
 */
export function createLeaguePipeline(algoType, rows, cols, numGames, pipelineType, gameType, benchRuntimeExtras) {
  var resolvedGameType = gameType || 'plague_walls';
  var runtimeSpec = resolveRuntimeSpec(pipelineType || 'single_gpu_phased');
  var requestedRuntimeId = pipelineType || runtimeSpec.id;
  if (runtimeSpec.pipelineKind !== 'gpu_worker') {
    throw new Error('League mode requires a GPU worker runtime (got ' + runtimeSpec.pipelineKind + ')');
  }
  var modelTypes = listModelTypes().map(function (t) {
    return t.id;
  });
  if (modelTypes.length < 2) {
    throw new Error('League needs at least two model types in the registry');
  }
  var runtimeOptions = runtimeSpec.options || {};
  var benchX = benchRuntimeExtras || {};
  var workerOptions = Object.assign({}, runtimeOptions, benchX, {
    pipelineTypeOverride: requestedRuntimeId,
    multiModel: true,
    modelTypes: modelTypes,
    trainInterval: typeof runtimeOptions.trainInterval === 'number' ? runtimeOptions.trainInterval : 48
  });
  return createGPUWorkerPipeline(
    modelTypes[0],
    algoType || 'ppo',
    rows,
    cols,
    numGames,
    resolvedGameType,
    LEAGUE_CHECKPOINT_POOL_CONFIG,
    workerOptions
  );
}
