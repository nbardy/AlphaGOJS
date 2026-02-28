import './games/plague_classic';
import './games/plague_advanced';
import { createGame } from './games/registry';
import { PolicyNetwork } from './model';
import { SelfPlayTrainer } from './trainer';
import { UI } from './ui';

var ROWS = 20;
var COLS = 20;
var NUM_GAMES = 80;
var GAME_ID = 'plague_advanced'; // 'plague_classic' or 'plague_advanced'

function start() {
  // Probe the game to get NN input size (may differ from board size for multi-channel games)
  var probe = createGame(GAME_ID, ROWS, COLS);
  var nnInputSize = probe.getBoardForNN(1).length;
  var boardSize = ROWS * COLS;

  var model = new PolicyNetwork(boardSize, nnInputSize);

  var trainer = new SelfPlayTrainer(model, {
    gameId: GAME_ID,
    numGames: NUM_GAMES,
    rows: ROWS,
    cols: COLS,
    trainBatchSize: 512,
    trainInterval: 30,
    snapshotInterval: 50,
    evalInterval: 100,
    evalGames: 20
  });

  var ui = new UI(trainer, model);
  window.__alphaPlague = ui;
}

if (module.hot) {
  if (window.__alphaPlague) window.__alphaPlague.destroy();
  module.hot.accept();
}

start();
