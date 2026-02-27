import { PolicyNetwork } from './model';
import { SelfPlayTrainer } from './trainer';
import { UI } from './ui';

var ROWS = 10;
var COLS = 10;
var NUM_GAMES = 40;

function start() {
  var model = new PolicyNetwork(ROWS * COLS);

  var trainer = new SelfPlayTrainer(model, {
    numGames: NUM_GAMES,
    rows: ROWS,
    cols: COLS,
    trainBatchSize: 256,
    trainInterval: 20
  });

  var ui = new UI(trainer, model);
  window.__alphaPlague = ui;
}

if (module.hot) {
  if (window.__alphaPlague) {
    window.__alphaPlague.destroy();
  }
  module.hot.accept();
}

start();
