import { DenseModel } from './dense_model';
import { SpatialModel } from './spatial_model';
import { Reinforce } from './reinforce';
import { PPO } from './ppo';
import { SelfPlayTrainer } from './trainer';
import { UI } from './ui';

// --- Configuration ---
var ROWS = 10;
var COLS = 10;
var NUM_GAMES = 40;
var MODEL_TYPE = 'dense';    // 'dense' | 'spatial'
var ALGO_TYPE = 'reinforce'; // 'reinforce' | 'ppo'

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

// --- Wiring ---

function start() {
  var model = createModel(MODEL_TYPE, ROWS, COLS);
  var algo = createAlgorithm(ALGO_TYPE, model);

  var trainer = new SelfPlayTrainer(algo, {
    numGames: NUM_GAMES,
    rows: ROWS,
    cols: COLS,
    trainBatchSize: 256,
    trainInterval: 20
  });

  var ui = new UI(trainer, algo);
  window.__alphaPlague = ui;
}

if (module.hot) {
  if (window.__alphaPlague) {
    window.__alphaPlague.destroy();
  }
  module.hot.accept();
}

start();
