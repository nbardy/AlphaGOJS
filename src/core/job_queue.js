// Lightweight async queue for non-blocking orchestration jobs.
// Runs tasks on setTimeout(0) with optional key-based de-duplication.

export class AsyncJobQueue {
  constructor(maxConcurrent) {
    this.maxConcurrent = maxConcurrent || 1;
    this.pending = [];
    this.running = 0;
    this.activeKeys = {};
    this._scheduled = false;
    this._closed = false;
  }

  enqueue(key, fn) {
    if (this._closed) return false;
    var dedupeKey = key || null;
    if (dedupeKey && this.activeKeys[dedupeKey]) return false;
    if (dedupeKey) this.activeKeys[dedupeKey] = 1;
    this.pending.push({ key: dedupeKey, fn: fn });
    this._schedule();
    return true;
  }

  hasKey(key) {
    return !!this.activeKeys[key];
  }

  size() {
    return this.pending.length + this.running;
  }

  close() {
    this._closed = true;
    this.pending = [];
    this.activeKeys = {};
  }

  _schedule() {
    if (this._scheduled || this._closed) return;
    this._scheduled = true;
    var self = this;
    setTimeout(function () {
      if (self._closed) return;
      self._scheduled = false;
      self._drain();
    }, 0);
  }

  _drain() {
    if (this._closed) return;
    var self = this;
    while (this.running < this.maxConcurrent && this.pending.length > 0) {
      var job = this.pending.shift();
      this.running++;
      setTimeout(function (j) {
        if (self._closed) {
          self._complete(j);
          return;
        }
        try {
          var out = j.fn();
          if (out && typeof out.then === 'function') {
            out.then(function () { self._complete(j); })
              .catch(function () { self._complete(j); });
            return;
          }
        } catch (e) {
          // Job failures are non-fatal at queue layer.
        }
        self._complete(j);
      }.bind(null, job), 0);
    }
  }

  _complete(job) {
    this.running = Math.max(0, this.running - 1);
    if (job.key) delete this.activeKeys[job.key];
    if (!this._closed && this.pending.length > 0) this._schedule();
  }
}
