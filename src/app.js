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
// League / Elo: slightly more checkpoint games and earlier first save so ratings move sooner.
var CHECKPOINT_POOL_CONFIG = {
  maxCheckpoints: 50,
  recentWindow: 50,
  sampleMode: 'uniform_recent',
  checkpointFraction: 0.3,
  saveInterval: 15
};

var DEFAULT_MODEL = 'spatial_lite';
var DEFAULT_ALGO = 'ppo';

function pickNum(v, fallback) {
  return typeof v === 'number' ? v : fallback;
}

function parseClampedPositiveIntParam(params, key, defaultVal, min, max) {
  var raw = params.get(key);
  if (raw === null || raw === '') {
    return defaultVal;
  }
  var n = parseInt(String(raw).trim(), 10);
  if (!Number.isFinite(n) || n < 1) {
    return defaultVal;
  }
  return Math.max(min, Math.min(max, n));
}

// Query overrides: ?pipeline=, bench flags, optional preset (?preset=fast|interactive),
// and optional grid/game count (?rows=&cols=&numGames=).
// When absent, bench extras are empty — no runtime cost beyond one URL parse at startup.
function parseRuntimeQueryOverrides() {
  var emptyBench = {};
  if (typeof location === 'undefined') {
    return {
      pipelineType: null,
      benchRuntimeExtras: emptyBench,
      rows: ROWS,
      cols: COLS,
      numGames: NUM_GAMES
    };
  }
  var p = new URLSearchParams(location.search);
  var bench = {};
  var bl = p.get('benchLoop');
  if (bl === 'sim_random' || bl === 'sim_forward') {
    bench.benchLoopMode = bl;
  }
  if (p.get('benchInstrument') === '1') {
    bench.benchInstrument = true;
  }
  if (p.get('benchMinimalUi') === '1') {
    bench.benchMinimalUi = true;
  }
  var pl = p.get('pipeline');
  var preset = p.get('preset');
  var presetFastOrInteractive = preset === 'fast' || preset === 'interactive';
  var rowsDefault = presetFastOrInteractive && !p.has('rows') ? 10 : ROWS;
  var colsDefault = presetFastOrInteractive && !p.has('cols') ? 10 : COLS;
  var numGamesDefault = presetFastOrInteractive && !p.has('numGames') ? 40 : NUM_GAMES;
  return {
    pipelineType: pl || null,
    benchRuntimeExtras: bench,
    rows: parseClampedPositiveIntParam(p, 'rows', rowsDefault, 4, 32),
    cols: parseClampedPositiveIntParam(p, 'cols', colsDefault, 4, 32),
    numGames: parseClampedPositiveIntParam(p, 'numGames', numGamesDefault, 4, 128)
  };
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
    checkpointPool: pool,
    uiSnapshotMaxGames: pickNum(runtimeOptions.uiSnapshotMaxGames, 48),
    algoType: algoType
  });
  return { trainer: trainer, algo: algo, pool: pool, pipelineType: runtimeId || 'gpu' };
}

export function createPipeline(modelType, algoType, rows, cols, numGames, pipelineType, gameType, benchRuntimeExtras) {
  var resolvedGameType = gameType || 'plague_walls';
  var runtimeSpec = resolveRuntimeSpec(pipelineType);
  var requestedRuntimeId = pipelineType || runtimeSpec.id;
  var runtimeOptions = runtimeSpec.options || {};
  var benchX = benchRuntimeExtras || {};

  if (runtimeSpec.pipelineKind === 'gpu_worker') {
    var workerOptions = Object.assign({}, runtimeOptions, benchX, {
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

// Tier A/C: GPU worker — phased mode avoids full_gpu queue stalls and long non-interactive bursts.
// Use runtime dropdown "Full GPU Resident" when you want max throughput and can tolerate jank.
function mapTierToPipelineType(tier) {
  if (tier === 'A' || tier === 'C') return 'single_gpu_phased';
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

  var q = parseRuntimeQueryOverrides();
  if (q.pipelineType) {
    pipelineType = q.pipelineType;
  }
  var benchExtras = q.benchRuntimeExtras || {};
  var rows = q.rows;
  var cols = q.cols;
  var numGames = q.numGames;

  var pipeline;
  try {
    pipeline = createPipeline(DEFAULT_MODEL, DEFAULT_ALGO, rows, cols, numGames, pipelineType, undefined, benchExtras);
  } catch (e) {
    pipeline = createPipeline(DEFAULT_MODEL, DEFAULT_ALGO, rows, cols, numGames, 'cpu_actors_gpu_learner', undefined, benchExtras);
  }
  if (benchExtras.benchLoopMode && typeof pipeline.trainer._ready !== 'boolean') {
    console.warn('[bench] benchLoop* only affects the GPU worker runtime; current pipeline:', pipeline.pipelineType);
  }
  var ui = new UI(pipeline.trainer, pipeline.algo, {
    rows: rows,
    cols: cols,
    numGames: numGames,
    pipelineType: pipeline.pipelineType,
    createPipeline: createPipeline,
    listModelTypes: listModelTypes,
    listAlgorithmTypes: listAlgorithmTypes,
    listRuntimeTypes: listRuntimeTypes,
    benchRuntimeExtras: benchExtras
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
