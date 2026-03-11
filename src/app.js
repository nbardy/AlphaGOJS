import { createModel, listModelTypes } from './model_registry';
import { createAlgorithm, listAlgorithmTypes } from './algo_registry';
import { assertModelContract, assertAlgorithmContract } from './contracts';
import { SelfPlayTrainer } from './trainer';
import { GPUAlgoAdapter } from './algos/gpu_algo_adapter';
import { GPUOrchestrator } from './orchestration/gpu_orchestrator';
import { createGPUWorkerPipeline } from './nextgen/create_gpu_worker_pipeline';
import { probeCapabilities } from './nextgen/capability_probe';
import { chooseRuntimeTier } from './nextgen/runtime_planner';
import { CheckpointPool } from './checkpoint_pool';
import { resolveRuntimeSpec, listRuntimeTypes } from './runtime/runtime_registry';
import { UI } from './ui';

// --- Configuration ---
var ROWS = 20;
var COLS = 20;
var NUM_GAMES = 80;
var CHECKPOINT_POOL_CONFIG = {
  maxCheckpoints: 50,
  recentWindow: 50,
  sampleMode: 'uniform_recent',
  checkpointFraction: 0.25,
  saveInterval: 20
};

function pickNum(v, fallback) {
  return typeof v === 'number' ? v : fallback;
}

// Pipeline constructors (internal implementation targets). Runtime selection
// is handled separately by runtime_registry.
function createCPUPipeline(modelType, algoType, rows, cols, numGames, gameType, runtimeOptions, runtimeId) {
  runtimeOptions = runtimeOptions || {};
  var model = createModel(modelType, rows, cols);
  assertModelContract(model);
  var algo = createAlgorithm(algoType, model);
  assertAlgorithmContract(algo);
  var pool = new CheckpointPool(function () {
    return createModel(modelType, rows, cols);
  }, CHECKPOINT_POOL_CONFIG);
  var trainer = new SelfPlayTrainer(algo, {
    numGames: numGames,
    rows: rows,
    cols: cols,
    gameType: gameType,
    trainBatchSize: pickNum(runtimeOptions.trainBatchSize, 512),
    trainInterval: pickNum(runtimeOptions.trainInterval, 30),
    checkpointPool: pool
  });
  return { trainer: trainer, algo: algo, pool: pool, pipelineType: runtimeId || 'cpu' };
}

function createGPUPipeline(modelType, algoType, rows, cols, numGames, gameType, runtimeOptions, runtimeId) {
  runtimeOptions = runtimeOptions || {};
  var model = createModel(modelType, rows, cols);
  assertModelContract(model);
  var algo = createAlgorithm(algoType, model);
  assertAlgorithmContract(algo);
  var gpuAlgo = new GPUAlgoAdapter(algo, model);
  var pool = new CheckpointPool(function () {
    return createModel(modelType, rows, cols);
  }, CHECKPOINT_POOL_CONFIG);
  // GPUOrchestrator keeps game simulation on GPU while allowing pluggable
  // algorithms (PPO/PPG/SAC/MuZero/REINFORCE) through a unified adapter.
  var trainer = new GPUOrchestrator(model, gpuAlgo, {
    numGames: numGames,
    rows: rows,
    cols: cols,
    gameType: gameType,
    trainBatchSize: pickNum(runtimeOptions.trainBatchSize, 512),
    trainInterval: pickNum(runtimeOptions.trainInterval, 30),
    checkpointPool: pool
  });
  return { trainer: trainer, algo: algo, pool: pool, pipelineType: runtimeId || 'gpu' };
}

export function createPipeline(modelType, algoType, rows, cols, numGames, pipelineType, gameType) {
  var resolvedGameType = gameType || 'plague_walls';
  var runtimeSpec = resolveRuntimeSpec(pipelineType);
  var requestedRuntimeId = pipelineType || runtimeSpec.id;
  var runtimeOptions = runtimeSpec.options || {};

  if (runtimeSpec.pipelineKind === 'gpu_worker') {
    var workerOptions = Object.assign({}, runtimeOptions, {
      pipelineTypeOverride: requestedRuntimeId
    });
    return createGPUWorkerPipeline(
      modelType,
      algoType,
      rows,
      cols,
      numGames,
      resolvedGameType,
      CHECKPOINT_POOL_CONFIG,
      workerOptions
    );
  }
  if (runtimeSpec.pipelineKind === 'gpu') {
    return createGPUPipeline(
      modelType,
      algoType,
      rows,
      cols,
      numGames,
      resolvedGameType,
      runtimeOptions,
      requestedRuntimeId
    );
  }
  if (runtimeSpec.pipelineKind === 'cpu') {
    return createCPUPipeline(
      modelType,
      algoType,
      rows,
      cols,
      numGames,
      resolvedGameType,
      runtimeOptions,
      requestedRuntimeId
    );
  }
  throw new Error('Unknown runtime pipeline kind: ' + runtimeSpec.pipelineKind);
}

export { listModelTypes, listAlgorithmTypes, listRuntimeTypes };

// --- Wiring ---

function mapTierToPipelineType(tier) {
  if (tier === 'A') return 'single_gpu_phased';
  if (tier === 'B') return 'cpu_actors_gpu_learner';
  return 'cpu_actors_gpu_learner';
}

async function start() {
  var pipelineType = 'cpu_actors_gpu_learner';
  try {
    var cap = await probeCapabilities();
    var plan = chooseRuntimeTier(cap, {});
    pipelineType = mapTierToPipelineType(plan.tier);
  } catch (e) {
    pipelineType = 'cpu_actors_gpu_learner';
  }

  var pipeline;
  try {
    pipeline = createPipeline('dense', 'ppo', ROWS, COLS, NUM_GAMES, pipelineType);
  } catch (e) {
    pipeline = createPipeline('dense', 'ppo', ROWS, COLS, NUM_GAMES, 'cpu_actors_gpu_learner');
  }
  var ui = new UI(pipeline.trainer, pipeline.algo, {
    rows: ROWS,
    cols: COLS,
    numGames: NUM_GAMES,
    pipelineType: pipeline.pipelineType,
    createPipeline: createPipeline,
    listModelTypes: listModelTypes,
    listAlgorithmTypes: listAlgorithmTypes,
    listRuntimeTypes: listRuntimeTypes
  });
  window.__alphaPlague = ui;
}

if (module.hot) {
  if (window.__alphaPlague) {
    window.__alphaPlague.destroy();
  }
  module.hot.accept();
}

start().catch(function (e) {
  console.error('Bootstrap failed:', e);
});
