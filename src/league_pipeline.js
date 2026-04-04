import { listModelTypes } from './model_registry';
import { createGPUWorkerPipeline } from './nextgen/create_gpu_worker_pipeline';
import { resolveRuntimeSpec } from './runtime/runtime_registry';

/** Default games completed (per architecture) before a PPO train step in league mode. */
export var LEAGUE_DEFAULT_TRAIN_INTERVAL = 48;

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
 *
 * @param {object} [leagueOverrides] Optional tuning (e.g. from league URL). Undefined fields keep defaults.
 * @param {number} [leagueOverrides.trainInterval]
 * @param {number} [leagueOverrides.checkpointFraction] 0–1
 * @param {number} [leagueOverrides.trainBatchSize] forwarded to worker (per-model scaling still applies in multi)
 * @param {boolean} [leagueOverrides.multiTrainStagger] false = one train burst for all ready models (old behavior)
 */
export function createLeaguePipeline(
  algoType,
  rows,
  cols,
  numGames,
  pipelineType,
  gameType,
  benchRuntimeExtras,
  leagueOverrides
) {
  var l = leagueOverrides || {};
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

  var poolCfg = Object.assign({}, LEAGUE_CHECKPOINT_POOL_CONFIG);
  if (typeof l.checkpointFraction === 'number' && l.checkpointFraction >= 0 && l.checkpointFraction <= 1) {
    poolCfg.checkpointFraction = l.checkpointFraction;
  }

  var trainInt =
    typeof l.trainInterval === 'number' && l.trainInterval >= 1
      ? l.trainInterval
      : typeof runtimeOptions.trainInterval === 'number'
        ? runtimeOptions.trainInterval
        : LEAGUE_DEFAULT_TRAIN_INTERVAL;

  var workerOptions = Object.assign({}, runtimeOptions, benchX, {
    pipelineTypeOverride: requestedRuntimeId,
    multiModel: true,
    modelTypes: modelTypes,
    trainInterval: trainInt
  });
  if (typeof l.trainBatchSize === 'number' && l.trainBatchSize >= 32) {
    workerOptions.trainBatchSize = l.trainBatchSize;
  }
  if (l.multiTrainStagger === false) {
    workerOptions.multiTrainStagger = false;
  }

  return createGPUWorkerPipeline(
    modelTypes[0],
    algoType || 'ppo',
    rows,
    cols,
    numGames,
    resolvedGameType,
    poolCfg,
    workerOptions
  );
}
