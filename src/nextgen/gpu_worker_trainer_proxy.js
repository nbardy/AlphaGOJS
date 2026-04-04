import { MSG, makeInit, makeTick, makeInferAction } from './protocol/messages';

function makeDefaultStats() {
  return {
    gamesCompleted: 0,
    generation: 0,
    loss: 0,
    p1Wins: 0,
    p2Wins: 0,
    draws: 0,
    avgGameLength: 0,
    bufferSize: 0,
    trainSteps: 0,
    elo: 1000,
    checkpointWinRate: 0,
    entropy: 0,
    trainInFlight: false,
    gpuReadbackCalls: 0,
    gpuReadbackBytes: 0,
    multiModel: false,
    modelStats: null
  };
}

export class GPUWorkerTrainerProxy {
  constructor(worker, config) {
    this.worker = worker;
    this.rows = config.rows || 20;
    this.cols = config.cols || 20;
    this.numGames = config.numGames || 80;
    this.boardSize = this.rows * this.cols;

    this._ready = false;
    this._disposed = false;
    this._stats = makeDefaultStats();
    this._boards = new Int8Array(this.numGames * this.boardSize);
    this._done = new Uint8Array(this.numGames);
    this._winners = new Int8Array(this.numGames);

    this._queuedSteps = 0;
    this._tickInFlight = false;
    this._maxTickBatch = Math.max(1, config.maxTickBatch || 8);
    this._maxQueuedSteps = Math.max(1, config.maxQueuedSteps || 4096);
    var qscf = config.queueSoftCapFraction;
    if (typeof qscf !== 'number' || !isFinite(qscf)) {
      qscf = config.pipelineTypeOverride === 'full_gpu_resident' ? 0.75 : 1.0;
    }
    this._queueSoftCapFraction = Math.min(1, Math.max(0, qscf));
    this._pauseTicksWhenTraining = !!config.pauseTicksWhenTraining;
    this._trainInFlightQueueCap = Math.max(0, config.trainInFlightQueueCap || 0);
    this._nextRequestId = 1;
    this._pending = {};

    this._onMessage = this._onMessage.bind(this);
    this.worker.addEventListener('message', this._onMessage);
    this.worker.postMessage(makeInit(config));
  }

  _queueSoftCapSteps() {
    return Math.max(1, Math.floor(this._maxQueuedSteps * this._queueSoftCapFraction));
  }

  _onMessage(ev) {
    var data = ev && ev.data ? ev.data : {};
    if (!data || !data.type) return;

    if (data.type === MSG.READY) {
      this._ready = true;
      this._flushTicks();
      return;
    }

    if (data.type === MSG.TICK_RESULT) {
      if (data.stats) this._stats = data.stats;
      if (data.boards) {
        var B = this.boardSize;
        if (data.boardSampleSlots && data.boardSampleSlots.length > 0) {
          var slots = data.boardSampleSlots;
          for (var si = 0; si < slots.length; si++) {
            var slot = slots[si];
            var dst = slot * B;
            var src = si * B;
            for (var j = 0; j < B; j++) {
              this._boards[dst + j] = data.boards[src + j];
            }
          }
        } else if (data.boards.length === this._boards.length) {
          this._boards.set(data.boards);
        } else {
          this._boards = new Int8Array(data.boards);
        }
      }
      if (data.done && data.done.length === this._done.length) {
        this._done.set(data.done);
      }
      if (data.winners && data.winners.length === this._winners.length) {
        this._winners.set(data.winners);
      }
      this._tickInFlight = false;
      this._flushTicks();
      return;
    }

    if (data.type === MSG.INFER_ACTION_RESULT) {
      var reqId = data.requestId;
      var pending = this._pending[reqId];
      if (pending) {
        delete this._pending[reqId];
        pending.resolve(data.action);
      }
      return;
    }

    if (data.type === MSG.ERROR) {
      console.warn('GPU worker error:', data.message);
    }
  }

  _flushTicks() {
    if (this._disposed || !this._ready || this._tickInFlight) return;
    if (this._queuedSteps <= 0) return;
    // Bound each worker tick request so UI-driven enqueue bursts do not turn
    // into one giant long-running job (which harms latency and observability).
    var steps = Math.min(this._queuedSteps, this._maxTickBatch);
    this._queuedSteps -= steps;
    this._tickInFlight = true;
    this.worker.postMessage(makeTick(steps));
  }

  tick() {
    if (this._disposed) return;
    if (this._pauseTicksWhenTraining && this._stats && this._stats.trainInFlight) {
      if (this._queuedSteps > this._trainInFlightQueueCap) return;
    }
    var softCap = this._queueSoftCapSteps();
    if (this._queuedSteps >= softCap) return;
    this._queuedSteps++;
    this._flushTicks();
  }

  selectActionAsync(state, mask) {
    if (this._disposed) return Promise.resolve(0);
    var requestId = this._nextRequestId++;
    var self = this;
    return new Promise(function (resolve, reject) {
      self._pending[requestId] = { resolve: resolve, reject: reject };
      self.worker.postMessage(makeInferAction(requestId, state, mask));
    });
  }

  getBoardsForRender() {
    return {
      boards: this._boards,
      done: this._done,
      winners: this._winners
    };
  }

  getStats() {
    var softCap = this._queueSoftCapSteps();
    return Object.assign({}, this._stats, {
      queueDepth: this._queuedSteps,
      queueSoftCap: softCap,
      queueSoftCapFraction: this._queueSoftCapFraction,
      tickInFlight: this._tickInFlight
    });
  }

  getBufferSize() {
    return this._stats.bufferSize || 0;
  }

  getTrainSteps() {
    return this._stats.trainSteps || 0;
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    try {
      this.worker.postMessage({ type: MSG.DISPOSE });
    } catch (e) {}
    try {
      this.worker.removeEventListener('message', this._onMessage);
    } catch (e) {}
    try {
      this.worker.terminate();
    } catch (e) {}
    var ids = Object.keys(this._pending);
    for (var i = 0; i < ids.length; i++) {
      var p = this._pending[ids[i]];
      if (p && p.resolve) p.resolve(0);
    }
    this._pending = {};
  }
}
