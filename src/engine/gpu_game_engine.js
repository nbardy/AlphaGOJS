import * as tf from '@tensorflow/tfjs';

// GPU simulation engine: owns only environment state and transitions.
// No RL math, no replay logic.

export class GPUGameEngine {
  constructor(config) {
    this.numGames = config.numGames || 40;
    this.rows = config.rows || 10;
    this.cols = config.cols || 10;
    this.boardSize = this.rows * this.cols;
    this.gameType = config.gameType || 'plague_walls';

    this.state = tf.zeros([this.numGames, this.boardSize]);
    this.neighborKernel = tf.tensor4d(
      [0, 1, 0, 1, 0, 1, 0, 1, 0],
      [3, 3, 1, 1]
    );

    this.turns = new Int32Array(this.numGames);
    this.done = new Uint8Array(this.numGames);
    this.winners = new Int8Array(this.numGames);
    this.doneTimes = new Float64Array(this.numGames);
  }

  getActiveSlots() {
    var out = [];
    for (var i = 0; i < this.numGames; i++) {
      if (!this.done[i]) out.push(i);
    }
    return out;
  }

  incrementTurnsForActive() {
    for (var i = 0; i < this.numGames; i++) {
      if (!this.done[i]) this.turns[i]++;
    }
  }

  applyActions(player, slotIds, actionsBySlot) {
    var N = this.numGames;
    var boardSize = this.boardSize;
    if (!slotIds || slotIds.length === 0) return;

    var actionsArr = new Int32Array(N);
    var activeArr = new Float32Array(N);
    for (var i = 0; i < slotIds.length; i++) {
      var slot = slotIds[i];
      if (slot < 0 || slot >= N || this.done[slot]) continue;
      var a = actionsBySlot[slot];
      actionsArr[slot] = Number.isFinite(a) ? a : 0;
      activeArr[slot] = 1;
    }

    var oldState = this.state;
    this.state = tf.tidy(function () {
      var actionsT = tf.tensor1d(actionsArr, 'int32');
      var activeT = tf.tensor1d(activeArr);
      var emptyMask = oldState.equal(0).cast('float32');
      var hasValid = emptyMask.sum(1).greater(0).cast('float32');
      var activeValid = activeT.mul(hasValid);
      var oneHot = tf.oneHot(actionsT, boardSize).mul(player);
      return oldState.add(oneHot.mul(activeValid.expandDims(1)));
    });
    oldState.dispose();
  }

  spread() {
    var N = this.numGames;
    var rows = this.rows;
    var cols = this.cols;
    var oldState = this.state;
    var kernel = this.neighborKernel;
    var boardSize = this.boardSize;
    this.state = tf.tidy(function () {
      var grid = oldState.reshape([N, rows, cols, 1]);
      var neighborSum = tf.conv2d(grid, kernel, 1, 'same');
      var rand = tf.randomUniform(neighborSum.shape);
      var weighted = neighborSum.mul(rand).mul(2);
      var emptyMask = grid.equal(0).cast('float32');
      var newVals = weighted.sign().mul(emptyMask);
      var occupied = grid.mul(tf.scalar(1).sub(emptyMask));
      return occupied.add(newVals).reshape([N, boardSize]);
    });
    oldState.dispose();
  }

  resolveTerminals() {
    var N = this.numGames;
    var boardSize = this.boardSize;

    var emptyCounts = tf.tidy(function () {
      return this.state.equal(0).cast('float32').sum(1);
    }.bind(this));
    var emptyData = emptyCounts.dataSync();
    emptyCounts.dispose();

    var stateData = this.state.dataSync();
    var doneSlots = [];
    var winners = [];
    for (var i = 0; i < N; i++) {
      if (this.done[i]) continue;
      if (emptyData[i] !== 0) continue;

      var p1 = 0, p2 = 0;
      var offset = i * boardSize;
      for (var j = 0; j < boardSize; j++) {
        var v = stateData[offset + j];
        if (v > 0) p1++;
        else if (v < 0) p2++;
      }
      var winner = p1 > p2 ? 1 : (p2 > p1 ? -1 : 0);
      this.done[i] = 1;
      this.winners[i] = winner;
      this.doneTimes[i] = Date.now();
      doneSlots.push(i);
      winners.push(winner);
    }
    return { doneSlots: doneSlots, winners: winners };
  }

  resetSlots(indices) {
    if (!indices || indices.length === 0) return;
    var stateData = this.state.dataSync();
    var newData = Float32Array.from(stateData);
    for (var k = 0; k < indices.length; k++) {
      var i = indices[k];
      var offset = i * this.boardSize;
      for (var j = 0; j < this.boardSize; j++) newData[offset + j] = 0;
      this.done[i] = 0;
      this.winners[i] = 0;
      this.turns[i] = 0;
      this.doneTimes[i] = 0;
    }
    var oldState = this.state;
    this.state = tf.tensor2d(newData, [this.numGames, this.boardSize]);
    oldState.dispose();
  }

  extractStatesMasksCPU(slotIds, player) {
    var outStates = [];
    var outMasks = [];
    if (!slotIds || slotIds.length === 0) return { states: outStates, masks: outMasks };

    var stateData = this.state.dataSync();
    for (var s = 0; s < slotIds.length; s++) {
      var slot = slotIds[s];
      var offset = slot * this.boardSize;
      var state = new Float32Array(this.boardSize);
      var mask = new Float32Array(this.boardSize);
      for (var j = 0; j < this.boardSize; j++) {
        var v = stateData[offset + j];
        state[j] = v * player;
        mask[j] = v === 0 ? 1 : 0;
      }
      outStates.push(state);
      outMasks.push(mask);
    }
    return { states: outStates, masks: outMasks };
  }

  getBoardsForRender() {
    return {
      boards: this.state.dataSync(),
      done: this.done,
      winners: this.winners
    };
  }

  dispose() {
    if (this.state) { this.state.dispose(); this.state = null; }
    if (this.neighborKernel) { this.neighborKernel.dispose(); this.neighborKernel = null; }
  }
}
