import { DenseModel } from './dense_model';
import { SpatialModel } from './spatial_model';
import { Reinforce } from './reinforce';
import { PPO } from './ppo';
import { SelfPlayTrainer } from './trainer';
import { GPUTrainer } from './gpu_trainer';
import { UI } from './ui';

// --- Configuration ---
var ROWS = 10;
var COLS = 10;
var NUM_GAMES = 40;
var MODEL_TYPE = 'dense';      // 'dense' | 'spatial'
var ALGO_TYPE = 'reinforce';   // 'reinforce' | 'ppo'
var PIPELINE_TYPE = 'cpu';     // 'cpu' | 'gpu'

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

function createPipeline(pipelineType, modelType, algoType, rows, cols, numGames) {
  var model = createModel(modelType, rows, cols);

  if (pipelineType === 'cpu') {
    var algo = createAlgorithm(algoType, model);
    var trainer = new SelfPlayTrainer(algo, {
      numGames: numGames,
      rows: rows,
      cols: cols,
      trainBatchSize: 256,
      trainInterval: 20
    });
    return { trainer: trainer, algo: algo, type: 'cpu' };
  }

  if (pipelineType === 'gpu') {
    var gpuTrainer = new GPUTrainer(model, {
      numGames: numGames,
      rows: rows,
      cols: cols,
      trainBatchSize: 256,
      trainInterval: 20
    });
    // GPUTrainer implements selectAction for human play
    return { trainer: gpuTrainer, algo: gpuTrainer, type: 'gpu' };
  }

  throw new Error('Unknown pipeline type: ' + pipelineType);
}

// --- Wiring ---

function start() {
  var pipeline = createPipeline(PIPELINE_TYPE, MODEL_TYPE, ALGO_TYPE, ROWS, COLS, NUM_GAMES);
  var ui = new UI(pipeline.trainer, pipeline.algo);
  window.__alphaPlague = ui;
}

if (module.hot) {
  if (window.__alphaPlague) {
    window.__alphaPlague.destroy();
  }
  module.hot.accept();
}

start();
