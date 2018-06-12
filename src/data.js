import * as tf from '@tensorflow/tfjs'
import * as nn from './nn'

function newGame(rows,cols)  {
  const size = rows * cols;
  // const cells = new Int8Array(size);

  // const buffer = tf.buffer([size],'int32',cells);
  // return buffer;
  return tf.zeros([size], 'int32');

}

const bufferCache = {};

function zeroBuffer(game) {
  const gameShape = game.shape;
  if(bufferCache[gameShape]) {
    return bufferCache[gameShape];
  }
  else {
    const buf = tf.buffer(game.shape, 'int32');
    bufferCache[gameShape] = buf;
    return buf;
  }
}

export function makeMove(game, player, move) {
  const buf = zeroBuffer(game);
  return tf.tidy( () => {
    buf.set(player, move);

    const newGame = game.add(buf.toTensor());

    // Unset from zero buffer
    buf.set(0, move);

    return newGame;
  })
}

function isEmpty(game, i) {
    return game.get(i) === 0;
}

function getMoveRandom(game) {
  var randSpot = Math.floor(game.size * Math.random())

  return randSpot;

  if(isEmpty(game,randSpot)) {
    return randSpot;
  } else {
    return getFirstEmptySquare(game);
  }
}

function getFirstEmptySquare(game) {
  for(let i = 0; i < game.size; i++) {

    if(game.get(i) === 0) {
      return i;
    }
  }
}

function getMoveClosestToICenter(game,opts) {
  const {COL_COUNT,ROW_COUNT} = opts;
  const middle = rowAndColToI([ROW_COUNT/2,COL_COUNT/2],opts)
  // const middle = 5500;
  var currentOffset = 0;

  while(!isEmpty(game,middle + currentOffset)) {
    // When negative flip the sign
    if(currentOffset / Math.abs(currentOffset) === 1) {
      currentOffset *= -1;
    }
    else {
      currentOffset = currentOffset * -1 + 1;
    }
    // When positive increment
  }

  return middle + currentOffset;
}


function rowAndColToI([row,col],{COL_COUNT}) {
  return (row * COL_COUNT) + col
}

function ItoRowAndCol(i,{ROW_COUNT, COL_COUNT}) {
  return [(i / COL_COUNT) >> 0, i % ROW_COUNT];
}


const randomCache = [];
function fillCache(n,shape) {
  console.log("Filling cache")
  for(var i = 0; i < n; i++) {
    randomCache.push(tf.keep(tf.randomUniform(shape)));
  }
}

// Spreads plague to empty squares
function progressBoard(game, opts) {
  const {COL_COUNT,ROW_COUNT} = opts;
  const shape = [ROW_COUNT,COL_COUNT];

  var total;
  var accumlator;
  var newVal;
  // const neighborFreqs = neighborFrequencies(game,opts);

  return tf.tidy("spread", () => {
    const init = game;
    const x = tf.tensor1d([1, 2, 3, 4]);
    const init2d = init.reshape(shape);
    // TODO: Replace slice and stack with gather
    // init2d.print();
    const padded    = init2d.pad([[1,1],[1,1]]);
    const up        = padded.slice([0,1],shape);
    const left      = padded.slice([2,1],shape);
    const down      = padded.slice([1,0],shape);
    const right     = padded.slice([1,2],shape);
    const neighbors = tf.stack([up,down,left,right],2);
    // const random    = tf.randomUniform(neighbors.shape);

    if(randomCache.length == 0) {
      fillCache(200,neighbors.shape);
    }
    const random    = randomCache[Math.floor(Math.random() * 200)];
    const weights   = random.mul(tf.cast(neighbors, 'float32'));
    const prob      = weights.sum(2);
    const rounded   = tf.cast(prob.mul(tf.scalar(2)), 'int32');
    // const probInc   = tf.mul(rounded,tf.scalar(50)); // To make Effect less random
    const nextVals  = rounded.clipByValue(-1,1);
    const isZero    = init2d.notEqual(tf.scalar(0,'int32'))
    const final     = tf.where(isZero,init2d,nextVals)

    return final.as1D();
  });

}

function gameEnded(game) {
  return tf.tidy(() => {
    const v = game.equal(tf.scalar(0,'int32')).sum().dataSync()[0];
    return v === 0;
  })
}


export {rowAndColToI, getMoveRandom, getMoveClosestToICenter, gameEnded, newGame, progressBoard}
