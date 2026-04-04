import * as tf from '@tensorflow/tfjs';
import { generatePlagueWallsInto, PLAGUE_WALL_CELL } from './plague_walls_layout';
import { floatEngineCellToNnCode } from '../nn_cell_codes';

// GPU simulation engine: owns only environment state and transitions.
// Hot path avoids full-board readback: gather rows for inference, small
// stats tensor for terminals, tensor reset for finished games.

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

    this.readbackCallsThisFrame = 0;
    this.readbackFloatsThisFrame = 0;
  }

  beginReadbackFrame() {
    this.readbackCallsThisFrame = 0;
    this.readbackFloatsThisFrame = 0;
  }

  _trackReadback(floatCount) {
    this.readbackCallsThisFrame++;
    this.readbackFloatsThisFrame += floatCount;
  }

  consumeReadbackFrame() {
    var f = this.readbackFloatsThisFrame;
    var c = this.readbackCallsThisFrame;
    this.readbackFloatsThisFrame = 0;
    this.readbackCallsThisFrame = 0;
    return {
      calls: c,
      floats: f,
      bytes: f * 4
    };
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
      // Only ±1 cells influence spread; walls (2) and empty (0) contribute 0 — matches CPU plague_walls.
      var playerOnly = grid.mul(grid.abs().equal(1).cast('float32'));
      var neighborSum = tf.conv2d(playerOnly, kernel, 1, 'same');
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
    var self = this;
    var packed = tf.tidy(function () {
      var s = self.state;
      var empty = s.equal(0).cast('float32').sum(1).expandDims(1);
      // Walls are PLAGUE_WALL_CELL (2); must not count as P1 territory.
      var pos = s.equal(1).cast('float32').sum(1).expandDims(1);
      var neg = s.equal(-1).cast('float32').sum(1).expandDims(1);
      return tf.concat([empty, pos, neg], 1);
    });
    this._trackReadback(N * 3);
    var data = packed.dataSync();
    packed.dispose();

    var doneSlots = [];
    var winners = [];
    for (var i = 0; i < N; i++) {
      if (this.done[i]) continue;
      var base = i * 3;
      if (data[base] !== 0) continue;

      var p1 = data[base + 1];
      var p2 = data[base + 2];
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
    var N = this.numGames;
    var B = this.boardSize;
    var rows = this.rows;
    var cols = this.cols;
    var oldState = this.state;

    if (this.gameType === 'plague_walls') {
      // Unique valid slots, sorted so we can tf.slice contiguous "keep" runs and
      // splice fresh wall layouts — avoids full-board dataSync when few games reset.
      var sorted = indices.slice();
      sorted.sort(function (a, b) { return a - b; });
      var unique = [];
      for (var ui = 0; ui < sorted.length; ui++) {
        var sl = sorted[ui];
        if (sl < 0 || sl >= N) continue;
        if (ui === 0 || sl !== sorted[ui - 1]) unique.push(sl);
      }

      if (unique.length === 0) {
        // Invalid indices only — leave GPU state untouched (same observable rows).
      } else if (unique.length === N) {
        var dataAll = new Float32Array(N * B);
        var rowBufAll = new Int8Array(B);
        for (var sa = 0; sa < N; sa++) {
          generatePlagueWallsInto(rowBufAll, rows, cols);
          var baseA = sa * B;
          for (var ja = 0; ja < B; ja++) dataAll[baseA + ja] = rowBufAll[ja];
        }
        oldState.dispose();
        this.state = tf.tensor2d(dataAll, [N, B]);
      } else {
        var rowBuf = new Int8Array(B);
        var rowFloat = new Float32Array(B);
        var newState = tf.tidy(function () {
          var pieces = [];
          var rowStart = 0;
          for (var t = 0; t <= unique.length; t++) {
            var resetAt = (t < unique.length) ? unique[t] : N;
            if (resetAt > rowStart) {
              pieces.push(tf.slice(oldState, [rowStart, 0], [resetAt - rowStart, B]));
            }
            if (t < unique.length) {
              generatePlagueWallsInto(rowBuf, rows, cols);
              for (var j = 0; j < B; j++) rowFloat[j] = rowBuf[j];
              pieces.push(tf.tensor2d(rowFloat, [1, B]));
              rowStart = unique[t] + 1;
            }
          }
          return pieces.length === 1 ? pieces[0] : tf.concat(pieces, 0);
        });
        oldState.dispose();
        this.state = newState;
      }
    } else {
      this.state = tf.tidy(function () {
        var idxT = tf.tensor1d(indices, 'int32');
        var oh = tf.oneHot(idxT, N);
        var rowReset = oh.max(0).expandDims(1);
        var keep = tf.sub(1, rowReset);
        return oldState.mul(keep);
      });
      oldState.dispose();
    }

    for (var k = 0; k < indices.length; k++) {
      var i = indices[k];
      this.done[i] = 0;
      this.winners[i] = 0;
      this.turns[i] = 0;
      this.doneTimes[i] = 0;
    }
  }

  /**
   * Gather state rows for listed slots. Returns NEW tensors (caller must dispose):
   * raw board rows shaped [k, boardSize]. On GPU, legal cells are empty iff
   * `raw.equal(0)` (mask for policy: `.cast('float32')`); player observation is
   * `raw.mul(player)` for player ∈ {1, -1} (walls: see extractStatesMasksCPU).
   */
  gatherSlotsTensor(slotIds) {
    if (!slotIds || slotIds.length === 0) return null;
    var indices = tf.tensor1d(slotIds, 'int32');
    var raw = tf.gather(this.state, indices, 0);
    indices.dispose();
    return raw;
  }

  extractStatesMasksCPU(slotIds, player) {
    var outStates = [];
    var outMasks = [];
    var outCodes = [];
    if (!slotIds || slotIds.length === 0) {
      return { states: outStates, masks: outMasks, codes: outCodes };
    }

    var boardSize = this.boardSize;
    var k = slotIds.length;
    var indices = tf.tensor1d(slotIds, 'int32');
    var gathered = tf.gather(this.state, indices, 0);
    indices.dispose();
    this._trackReadback(k * boardSize);
    var stateData = gathered.dataSync();
    gathered.dispose();

    for (var s = 0; s < k; s++) {
      var offset = s * boardSize;
      var state = new Float32Array(boardSize);
      var mask = new Float32Array(boardSize);
      var codes = new Int32Array(boardSize);
      for (var j = 0; j < boardSize; j++) {
        var v = stateData[offset + j];
        if (v === PLAGUE_WALL_CELL) {
          state[j] = 0.5;
        } else {
          state[j] = v * player;
        }
        mask[j] = v === 0 ? 1 : 0;
        codes[j] = floatEngineCellToNnCode(v, player);
      }
      outStates.push(state);
      outMasks.push(mask);
      outCodes.push(codes);
    }
    return { states: outStates, masks: outMasks, codes: outCodes };
  }

  /**
   * Full board readback (e.g. one-time UI warm-up or when maxSlots >= numGames).
   */
  getBoardsForRender() {
    var boards = this.state.dataSync();
    this._trackReadback(this.numGames * this.boardSize);
    return {
      boards: boards,
      done: this.done,
      winners: this.winners
    };
  }

  /**
   * Partial readback: only listed slot rows (hot path for UI / worker transfer).
   */
  getBoardsForRenderGather(slotIds) {
    if (!slotIds || slotIds.length === 0) {
      return {
        boards: new Float32Array(0),
        done: this.done,
        winners: this.winners
      };
    }
    var boardSize = this.boardSize;
    var k = slotIds.length;
    var indices = tf.tensor1d(slotIds, 'int32');
    var gathered = tf.gather(this.state, indices, 0);
    indices.dispose();
    this._trackReadback(k * boardSize);
    var boards = gathered.dataSync();
    gathered.dispose();
    return {
      boards: boards,
      done: this.done,
      winners: this.winners
    };
  }

  /**
   * plague_walls: random wall layout per row (matches CPU). plague_classic: no-op (zeros).
   */
  seedInitialBoardsIfNeeded() {
    if (this.gameType !== 'plague_walls') return;
    var N = this.numGames;
    var B = this.boardSize;
    var data = new Float32Array(N * B);
    for (var s = 0; s < N; s++) {
      var rowBuf = new Int8Array(B);
      generatePlagueWallsInto(rowBuf, this.rows, this.cols);
      var base = s * B;
      for (var j = 0; j < B; j++) data[base + j] = rowBuf[j];
    }
    if (this.state) this.state.dispose();
    this.state = tf.tensor2d(data, [N, B]);
  }

  dispose() {
    if (this.state) { this.state.dispose(); this.state = null; }
    if (this.neighborKernel) { this.neighborKernel.dispose(); this.neighborKernel = null; }
  }
}
