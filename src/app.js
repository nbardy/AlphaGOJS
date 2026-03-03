import { createModel, listModelTypes } from './model_registry';
import { createAlgorithm, listAlgorithmTypes } from './algo_registry';
import { assertModelContract, assertAlgorithmContract } from './contracts';
import { SelfPlayTrainer } from './trainer';
import { GPUTrainer } from './gpu_trainer';
import { CheckpointPool } from './checkpoint_pool';
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

// Pipeline type dispatcher: 'cpu' uses SelfPlayTrainer (model+algo abstraction),
// 'gpu' uses GPUTrainer (all game state on GPU, ~320 bytes/tick transfer).
// GPU pipeline bypasses the algo abstraction — GPUTrainer owns its own training loop
// and exposes selectAction directly for human play.
// gameType is a registry key (e.g. 'plague_walls', 'plague_classic').
function createCPUPipeline(modelType, algoType, rows, cols, numGames, gameType) {
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
    trainBatchSize: 512,
    trainInterval: 30,
    checkpointPool: pool
  });
  return { trainer: trainer, algo: algo, pool: pool, pipelineType: 'cpu' };
}

function createGPUPipeline(modelType, rows, cols, numGames, gameType) {
  var model = createModel(modelType, rows, cols);
  assertModelContract(model);
  var pool = new CheckpointPool(function () {
    return createModel(modelType, rows, cols);
  }, CHECKPOINT_POOL_CONFIG);
  // GPUTrainer owns its game loop on GPU; checkpoint pool runs async CPU-side
  // Elo eval (1 game every 10 gens) so the Elo chart is no longer flat.
  var trainer = new GPUTrainer(model, {
    numGames: numGames,
    rows: rows,
    cols: cols,
    gameType: gameType,
    trainBatchSize: 512,
    trainInterval: 30,
    checkpointPool: pool
  });
  return { trainer: trainer, algo: null, pool: pool, pipelineType: 'gpu' };
}

export function createPipeline(modelType, algoType, rows, cols, numGames, pipelineType, gameType) {
  var resolvedGameType = gameType || 'plague_walls';
  // GPUTrainer currently implements a REINFORCE-like internal loop.
  // Advanced algorithms (SAC/PPG/MuZero) run on CPU pipeline.
  if (pipelineType === 'gpu' && (algoType === 'ppo' || algoType === 'reinforce')) {
    return createGPUPipeline(modelType, rows, cols, numGames, resolvedGameType);
  }
  if (pipelineType === 'gpu') {
    return createCPUPipeline(modelType, algoType, rows, cols, numGames, resolvedGameType);
  }
  if (pipelineType === 'cpu' || !pipelineType) return createCPUPipeline(modelType, algoType, rows, cols, numGames, resolvedGameType);
  throw new Error('Unknown pipeline type: ' + pipelineType);
}

export { listModelTypes, listAlgorithmTypes };

// --- Wiring ---

function start() {
  var pipeline = createPipeline('dense', 'ppo', ROWS, COLS, NUM_GAMES, 'cpu');
  var ui = new UI(pipeline.trainer, pipeline.algo, {
    rows: ROWS,
    cols: COLS,
    numGames: NUM_GAMES,
    pipelineType: pipeline.pipelineType,
    createPipeline: createPipeline,
    listModelTypes: listModelTypes,
    listAlgorithmTypes: listAlgorithmTypes
  });
  window.__alphaPlague = ui;
}

if (module.hot) {
  if (window.__alphaPlague) {
    window.__alphaPlague.destroy();
  }
  module.hot.accept();
}

start();
