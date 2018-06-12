import {newGame,gameEnded,getMoveRandom,getMoveClosestToICenter,progressBoard,makeMove} from './data.js';
import {newCanvas,drawGame} from './draw.js';
import {runml} from './nn.js';
import * as tf from '@tensorflow/tfjs'
const ROW_COUNT  = 50,
      COL_COUNT  = 50,
      CELL_WIDTH = 10,
      TURN_SPEED = 0;

var gameThreads = [];

if(module.hot) {
  console.log(module.hot);
  clean()
  module.hot.accept();
  start()
}

function start() {
  const canvas = newCanvas(COL_COUNT, ROW_COUNT, CELL_WIDTH)
  var game = newGame(ROW_COUNT, COL_COUNT);

  const board = document.createElement("div");
  document.body.appendChild(board);
  document.body.appendChild(canvas)

  const opts = {
    ROW_COUNT: ROW_COUNT,
    COL_COUNT: COL_COUNT,
    CELL_WIDTH: CELL_WIDTH,
    canvas: canvas,
    board: board
  }

  const optimizer = tf.train.momentum(0.1, 0.01)

  drawGame(game, opts)
  // dl.ENV.set('DEBUG',true)
  //
  const t0 = performance.mark("startgame");

  var player;
  var move;
  var nextGame;
  const next = function(currentGame) {
    
    tf.tidy(() => {
      player = 1;
      move = getMoveRandom(currentGame,opts);

      nextGame = makeMove(currentGame, player, move)
      currentGame.dispose()


      player = -1

      // drawGame(nextGame, opts);
      //

      move = getMoveRandom(nextGame,opts);
      nextGame = makeMove(nextGame, player, move)

      nextGame = progressBoard(nextGame,opts);

      var advancedGame;

      if (!gameEnded(nextGame)) {
        advancedGame = tf.keep(nextGame);
        gameThreads.push(setTimeout(() => { 
          next(advancedGame)
        }, TURN_SPEED));
      } else {
        drawGame(nextGame, opts);
        const t1 = performance.mark("endgame");
        performance.measure("game", "startgame","endgame")
        advancedGame = tf.keep(newGame(ROW_COUNT, COL_COUNT));
        next(advancedGame)
        // TODO
      }
    })
  }

  next(game)
}

function clean() {
  document.body.innerHTML = ""
  gameThreads.forEach(v => clearTimeout(v))
  gameThreads = [];
}
