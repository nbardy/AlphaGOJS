import { DenseModel } from './dense_model';
import { SpatialModel } from './spatial_model';
import { Reinforce } from './reinforce';
import { PPO } from './ppo';
import { SelfPlayTrainer } from './trainer';
import { GPUTrainer } from './gpu_trainer';
import { CheckpointPool } from './checkpoint_pool';
import { UI } from './ui';

// --- Configuration ---
var ROWS = 20;
var COLS = 20;
var NUM_GAMES = 80;

// --- Thin dispatchers (One Clean Path: one handler per type, exhaustive) ---

function createModel(type, rows, cols) {
  if (type === 'dense') return new DenseModel(rows, cols);
  if (type === 'spatial') return new SpatialModel(rows, cols);
  throw new Error('Unknown model type: ' + type);
}

function createAlgorithm(type, model) {
  if (type === 'reinforce') return new Reinforce(model);
  if (type === 'ppo') return new PPO(model);
  throw new Error('Unknown algorithm type: ' + type);
}

// Pipeline type dispatcher: 'cpu' uses SelfPlayTrainer (model+algo abstraction),
// 'gpu' uses GPUTrainer (all game state on GPU, ~320 bytes/tick transfer).
// GPU pipeline bypasses the algo abstraction — GPUTrainer owns its own training loop
// and exposes selectAction directly for human play.
function createCPUPipeline(modelType, algoType, rows, cols, numGames, walls) {
  var model = createModel(modelType, rows, cols);
  var algo = createAlgorithm(algoType, model);
  var pool = new CheckpointPool(function () {
    return createModel(modelType, rows, cols);
  });
  var trainer = new SelfPlayTrainer(algo, {
    numGames: numGames,
    rows: rows,
    cols: cols,
    walls: walls,
    trainBatchSize: 512,
    trainInterval: 30,
    checkpointPool: pool
  });
  return { trainer: trainer, algo: algo, pool: pool, pipelineType: 'cpu' };
}

function createGPUPipeline(modelType, rows, cols, numGames, walls) {
  var model = createModel(modelType, rows, cols);
  var pool = new CheckpointPool(function () {
    return createModel(modelType, rows, cols);
  });
  // GPUTrainer owns its game loop on GPU; checkpoint pool runs async CPU-side
  // Elo eval (1 game every 10 gens) so the Elo chart is no longer flat.
  var trainer = new GPUTrainer(model, {
    numGames: numGames,
    rows: rows,
    cols: cols,
    walls: walls,
    trainBatchSize: 512,
    trainInterval: 30,
    checkpointPool: pool
  });
  return { trainer: trainer, algo: null, pool: pool, pipelineType: 'gpu' };
}

export function createPipeline(modelType, algoType, rows, cols, numGames, pipelineType, gameType) {
  var walls = gameType !== 'classic'; // 'advanced' (default) = walls, 'classic' = no walls
  if (pipelineType === 'gpu') return createGPUPipeline(modelType, rows, cols, numGames, walls);
  if (pipelineType === 'cpu' || !pipelineType) return createCPUPipeline(modelType, algoType, rows, cols, numGames, walls);
  throw new Error('Unknown pipeline type: ' + pipelineType);
}

// --- Wiring ---

function start() {
  var pipeline = createPipeline('dense', 'ppo', ROWS, COLS, NUM_GAMES, 'cpu');
  var ui = new UI(pipeline.trainer, pipeline.algo, {
    rows: ROWS,
    cols: COLS,
    numGames: NUM_GAMES,
    pipelineType: pipeline.pipelineType,
    createPipeline: createPipeline
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
