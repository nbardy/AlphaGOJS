/**
 * WebGPU batched plague_walls env — same responsibilities as GPUGameEngine (worker path).
 * Uses unpacked u32 per cell on GPU; policy/UI paths use a CPU cache + TF tensors.
 */

import * as tf from '@tensorflow/tfjs';
import { generatePlagueWallsInto, PLAGUE_WALL_CELL } from './plague_walls_layout';
import { copyGpuBufferToUint32 } from './webgpu_copy_read_u32.js';
import {
  CELL_EMPTY,
  CELL_P1,
  CELL_P2,
  CELL_WALL,
  encodePlagueWallsBoardToPacked,
  requestPlagueSpreadDevice
} from './webgpu_plague_spread_engine.js';

function packedRowToFloat32Row(packedRow, boardSize, out) {
  for (var j = 0; j < boardSize; j++) {
    var u = packedRow[j];
    if (u === CELL_WALL) out[j] = PLAGUE_WALL_CELL;
    else if (u === CELL_P1) out[j] = 1;
    else if (u === CELL_P2) out[j] = -1;
    else out[j] = 0;
  }
}

export class WebGPUGameEngine {
  /**
   * @param {GPUDevice} device
   * @param {{ numGames: number, rows: number, cols: number, gameType: string }} config
   * @param {string} wgslSource full plague_env.wgsl source
   */
  constructor(device, config, wgslSource) {
    this.device = device;
    this.numGames = config.numGames | 0;
    this.rows = config.rows | 0;
    this.cols = config.cols | 0;
    this.boardSize = this.rows * this.cols;
    this.gameType = config.gameType || 'plague_walls';

    this.turns = new Int32Array(this.numGames);
    this.done = new Uint8Array(this.numGames);
    this.winners = new Int8Array(this.numGames);
    this.doneTimes = new Float64Array(this.numGames);

    this.readbackCallsThisFrame = 0;
    this.readbackFloatsThisFrame = 0;

    this._tick = 0 >>> 0;
    this._readIsA = true;
    this._boardCacheDirty = true;
    this._packedCache = new Uint32Array(this.numGames * this.boardSize);

    var bytes = this.boardSize * this.numGames * 4;
    this.bufA = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    this.bufB = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    this.uniformSpread = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.uniformApply = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    this.uniformTerm = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    this.actionsBuf = device.createBuffer({
      size: Math.max(4, this.numGames * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    this.activeBuf = device.createBuffer({
      size: Math.max(4, this.numGames * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });

    var termBytes = Math.max(12, this.numGames * 3 * 4);
    this.termOut = device.createBuffer({
      size: termBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    var module = device.createShaderModule({ code: wgslSource });

    var layoutSpread = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });
    var layoutApply = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
      ]
    });
    var layoutTerm = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });

    this.pipelineSpread = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layoutSpread] }),
      compute: { module: module, entryPoint: 'spread_pass' }
    });
    this.pipelineApply = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layoutApply] }),
      compute: { module: module, entryPoint: 'apply_pass' }
    });
    this.pipelineTerm = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layoutTerm] }),
      compute: { module: module, entryPoint: 'terminal_pass' }
    });

    this._makeSpreadBinds(layoutSpread);
    this._makeApplyBinds(layoutApply);
    this._makeTermBinds(layoutTerm);

    this._wgSpread = Math.ceil((this.boardSize * this.numGames) / 64);
    this._wgApply = this._wgSpread;
    this._wgTerm = Math.ceil(this.numGames / 64);

    this.isWebGPUGameEngine = true;
  }

  _makeSpreadBinds(layoutSpread) {
    var d = this.device;
    this.bindSpreadA = d.createBindGroup({
      layout: layoutSpread,
      entries: [
        { binding: 0, resource: { buffer: this.uniformSpread } },
        { binding: 1, resource: { buffer: this.bufA } },
        { binding: 2, resource: { buffer: this.bufB } }
      ]
    });
    this.bindSpreadB = d.createBindGroup({
      layout: layoutSpread,
      entries: [
        { binding: 0, resource: { buffer: this.uniformSpread } },
        { binding: 1, resource: { buffer: this.bufB } },
        { binding: 2, resource: { buffer: this.bufA } }
      ]
    });
  }

  _makeApplyBinds(layoutApply) {
    var d = this.device;
    this.bindApplyA = d.createBindGroup({
      layout: layoutApply,
      entries: [
        { binding: 0, resource: { buffer: this.uniformApply } },
        { binding: 1, resource: { buffer: this.bufA } },
        { binding: 2, resource: { buffer: this.bufB } },
        { binding: 3, resource: { buffer: this.actionsBuf } },
        { binding: 4, resource: { buffer: this.activeBuf } }
      ]
    });
    this.bindApplyB = d.createBindGroup({
      layout: layoutApply,
      entries: [
        { binding: 0, resource: { buffer: this.uniformApply } },
        { binding: 1, resource: { buffer: this.bufB } },
        { binding: 2, resource: { buffer: this.bufA } },
        { binding: 3, resource: { buffer: this.actionsBuf } },
        { binding: 4, resource: { buffer: this.activeBuf } }
      ]
    });
  }

  _makeTermBinds(layoutTerm) {
    var d = this.device;
    this.bindTermA = d.createBindGroup({
      layout: layoutTerm,
      entries: [
        { binding: 0, resource: { buffer: this.uniformTerm } },
        { binding: 1, resource: { buffer: this.bufA } },
        { binding: 2, resource: { buffer: this.termOut } }
      ]
    });
    this.bindTermB = d.createBindGroup({
      layout: layoutTerm,
      entries: [
        { binding: 0, resource: { buffer: this.uniformTerm } },
        { binding: 1, resource: { buffer: this.bufB } },
        { binding: 2, resource: { buffer: this.termOut } }
      ]
    });
  }

  static async create(config, wgslSource) {
    var req = await requestPlagueSpreadDevice();
    return new WebGPUGameEngine(req.device, config, wgslSource);
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
    return { calls: c, floats: f, bytes: f * 4 };
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

  /**
   * Pull latest GPU board into `_packedCache` for policy gather + UI.
   */
  async syncBoardFromGpu() {
    var readBuf = this._readIsA ? this.bufA : this.bufB;
    var count = this.boardSize * this.numGames;
    var copy = await copyGpuBufferToUint32(this.device, readBuf, 0, count);
    this._packedCache.set(copy);
    this._boardCacheDirty = false;
    this._trackReadback(count);
  }

  async ensureBoardCacheForPolicy() {
    if (!this._boardCacheDirty) return;
    await this.syncBoardFromGpu();
  }

  async syncBoardFromGpuForUi() {
    await this.syncBoardFromGpu();
  }

  _writeSpreadUniform() {
    var u = new Uint32Array([this.rows >>> 0, this.cols >>> 0, this._tick >>> 0, this.numGames >>> 0]);
    this.device.queue.writeBuffer(this.uniformSpread, 0, u);
  }

  _writeTermUniform() {
    var u = new Uint32Array([this.rows >>> 0, this.cols >>> 0, 0, this.numGames >>> 0]);
    this.device.queue.writeBuffer(this.uniformTerm, 0, u);
  }

  _writeApplyUniform(playerCellU32) {
    var u = new Uint32Array([
      this.rows >>> 0,
      this.cols >>> 0,
      this.numGames >>> 0,
      playerCellU32 >>> 0
    ]);
    this.device.queue.writeBuffer(this.uniformApply, 0, u);
  }

  async applyActions(player, slotIds, actionsBySlot) {
    var N = this.numGames;
    if (!slotIds || slotIds.length === 0) return;

    var actionsArr = new Int32Array(N);
    var activeArr = new Uint32Array(N);
    for (var i = 0; i < slotIds.length; i++) {
      var slot = slotIds[i];
      if (slot < 0 || slot >= N || this.done[slot]) continue;
      var a = actionsBySlot[slot];
      actionsArr[slot] = Number.isFinite(a) ? a : 0;
      activeArr[slot] = 1;
    }

    var playerCell = player > 0 ? CELL_P1 : CELL_P2;
    this._writeApplyUniform(playerCell);
    this.device.queue.writeBuffer(this.actionsBuf, 0, actionsArr);
    this.device.queue.writeBuffer(this.activeBuf, 0, activeArr);

    var enc = this.device.createCommandEncoder();
    var pass = enc.beginComputePass();
    pass.setPipeline(this.pipelineApply);
    pass.setBindGroup(0, this._readIsA ? this.bindApplyA : this.bindApplyB);
    pass.dispatchWorkgroups(this._wgApply);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    this._readIsA = !this._readIsA;
    this._boardCacheDirty = true;
  }

  async spread() {
    this._writeSpreadUniform();
    var enc = this.device.createCommandEncoder();
    var pass = enc.beginComputePass();
    pass.setPipeline(this.pipelineSpread);
    pass.setBindGroup(0, this._readIsA ? this.bindSpreadA : this.bindSpreadB);
    pass.dispatchWorkgroups(this._wgSpread);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    this._readIsA = !this._readIsA;
    this._tick = (this._tick + 1) >>> 0;
    this._boardCacheDirty = true;
  }

  async resolveTerminals() {
    var N = this.numGames;
    this._writeTermUniform();
    var enc = this.device.createCommandEncoder();
    var pass = enc.beginComputePass();
    pass.setPipeline(this.pipelineTerm);
    pass.setBindGroup(0, this._readIsA ? this.bindTermA : this.bindTermB);
    pass.dispatchWorkgroups(this._wgTerm);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    var data = await copyGpuBufferToUint32(this.device, this.termOut, 0, N * 3);
    this._trackReadback(N * 3);

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
    if (!indices || indices.length === 0) return Promise.resolve();
    var self = this;
    if (this.gameType !== 'plague_walls') {
      for (var k = 0; k < indices.length; k++) {
        var si = indices[k];
        if (si < 0 || si >= this.numGames) continue;
        var row0 = new Uint32Array(this.boardSize);
        var base = si * this.boardSize;
        for (var z = 0; z < this.boardSize; z++) {
          self._packedCache[base + z] = CELL_EMPTY;
        }
        var readBuf = this._readIsA ? this.bufA : this.bufB;
        var offset = si * this.boardSize * 4;
        this.device.queue.writeBuffer(readBuf, offset, row0);
      }
      for (var k2 = 0; k2 < indices.length; k2++) {
        var i = indices[k2];
        this.done[i] = 0;
        this.winners[i] = 0;
        this.turns[i] = 0;
        this.doneTimes[i] = 0;
      }
      this._boardCacheDirty = true;
      return Promise.resolve();
    }

    var N = this.numGames;
    var B = this.boardSize;
    var rows = this.rows;
    var cols = this.cols;
    var readBuf = this._readIsA ? this.bufA : this.bufB;
    var rowBuf = new Int8Array(B);

    for (var r = 0; r < indices.length; r++) {
      var slot = indices[r];
      if (slot < 0 || slot >= N) continue;
      generatePlagueWallsInto(rowBuf, rows, cols);
      var packed = encodePlagueWallsBoardToPacked(rowBuf);
      var offset = slot * B * 4;
      this.device.queue.writeBuffer(readBuf, offset, packed);
    }

    for (var k3 = 0; k3 < indices.length; k3++) {
      var ix = indices[k3];
      this.done[ix] = 0;
      this.winners[ix] = 0;
      this.turns[ix] = 0;
      this.doneTimes[ix] = 0;
    }
    // Partial cache update for reset rows; other slots unchanged if cache was valid.
    this._boardCacheDirty = true;
    return Promise.resolve();
  }

  gatherSlotsTensor(slotIds) {
    if (!slotIds || slotIds.length === 0) return null;
    var boardSize = this.boardSize;
    var k = slotIds.length;
    var data = new Float32Array(k * boardSize);
    var rowTmp = new Float32Array(boardSize);
    for (var s = 0; s < k; s++) {
      var slot = slotIds[s];
      var offset = slot * boardSize;
      var slice = this._packedCache.subarray(offset, offset + boardSize);
      packedRowToFloat32Row(slice, boardSize, rowTmp);
      data.set(rowTmp, s * boardSize);
    }
    return tf.tensor2d(data, [k, boardSize]);
  }

  extractStatesMasksCPU(slotIds, player) {
    var outStates = [];
    var outMasks = [];
    if (!slotIds || slotIds.length === 0) return { states: outStates, masks: outMasks };
    var boardSize = this.boardSize;
    var k = slotIds.length;
    this._trackReadback(k * boardSize);
    for (var s = 0; s < k; s++) {
      var slot = slotIds[s];
      var offset = slot * boardSize;
      var state = new Float32Array(boardSize);
      var mask = new Float32Array(boardSize);
      for (var j = 0; j < boardSize; j++) {
        var u = this._packedCache[offset + j];
        var v;
        if (u === CELL_WALL) v = PLAGUE_WALL_CELL;
        else if (u === CELL_P1) v = 1;
        else if (u === CELL_P2) v = -1;
        else v = 0;
        if (v === PLAGUE_WALL_CELL) {
          state[j] = 0.5;
        } else {
          state[j] = v * player;
        }
        mask[j] = v === 0 ? 1 : 0;
      }
      outStates.push(state);
      outMasks.push(mask);
    }
    return { states: outStates, masks: outMasks };
  }

  getBoardsForRender() {
    var boardSize = this.boardSize;
    var N = this.numGames;
    var boards = new Float32Array(N * boardSize);
    var rowTmp = new Float32Array(boardSize);
    for (var s = 0; s < N; s++) {
      var offset = s * boardSize;
      packedRowToFloat32Row(this._packedCache.subarray(offset, offset + boardSize), boardSize, rowTmp);
      boards.set(rowTmp, offset);
    }
    this._trackReadback(N * boardSize);
    return {
      boards: boards,
      done: this.done,
      winners: this.winners
    };
  }

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
    var boards = new Float32Array(k * boardSize);
    var rowTmp = new Float32Array(boardSize);
    for (var s = 0; s < k; s++) {
      var slot = slotIds[s];
      var offset = slot * boardSize;
      packedRowToFloat32Row(this._packedCache.subarray(offset, offset + boardSize), boardSize, rowTmp);
      boards.set(rowTmp, s * boardSize);
    }
    this._trackReadback(k * boardSize);
    return {
      boards: boards,
      done: this.done,
      winners: this.winners
    };
  }

  seedInitialBoardsIfNeeded() {
    if (this.gameType !== 'plague_walls') return;
    var N = this.numGames;
    var B = this.boardSize;
    var rows = this.rows;
    var cols = this.cols;
    var rowBuf = new Int8Array(B);
    var full = new Uint32Array(N * B);
    for (var s = 0; s < N; s++) {
      generatePlagueWallsInto(rowBuf, rows, cols);
      var packed = encodePlagueWallsBoardToPacked(rowBuf);
      full.set(packed, s * B);
    }
    this._readIsA = true;
    this.device.queue.writeBuffer(this.bufA, 0, full);
    this._packedCache.set(full);
    this._tick = 0;
    this._boardCacheDirty = false;
  }

  dispose() {
    try { this.bufA.destroy(); } catch (e) {}
    try { this.bufB.destroy(); } catch (e) {}
    try { this.uniformSpread.destroy(); } catch (e) {}
    try { this.uniformApply.destroy(); } catch (e) {}
    try { this.uniformTerm.destroy(); } catch (e) {}
    try { this.actionsBuf.destroy(); } catch (e) {}
    try { this.activeBuf.destroy(); } catch (e) {}
    try { this.termOut.destroy(); } catch (e) {}
    this.bufA = null;
    this.bufB = null;
  }
}
