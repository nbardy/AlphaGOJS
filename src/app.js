import { DenseModel } from './dense_model';
import { SpatialModel } from './spatial_model';
import { Reinforce } from './reinforce';
import { PPO } from './ppo';
import { SelfPlayTrainer } from './trainer';
import { CheckpointPool } from './checkpoint_pool';
import { UI } from './ui';

// --- Configuration ---
var ROWS = 10;
var COLS = 10;
var NUM_GAMES = 40;

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

export function createPipeline(modelType, algoType, rows, cols, numGames) {
  var model = createModel(modelType, rows, cols);
  var algo = createAlgorithm(algoType, model);
  var pool = new CheckpointPool(function () {
    return createModel(modelType, rows, cols);
  });
  var trainer = new SelfPlayTrainer(algo, {
    numGames: numGames,
    rows: rows,
    cols: cols,
    trainBatchSize: 256,
    trainInterval: 20,
    checkpointPool: pool
  });
  return { trainer: trainer, algo: algo, pool: pool };
}

// --- Wiring ---

function start() {
  var pipeline = createPipeline('dense', 'ppo', ROWS, COLS, NUM_GAMES);
  var ui = new UI(pipeline.trainer, pipeline.algo, {
    rows: ROWS,
    cols: COLS,
    numGames: NUM_GAMES,
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
