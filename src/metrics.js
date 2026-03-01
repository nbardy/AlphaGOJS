// Per-generation metrics log.
// Stores rolling window of snapshots for chart rendering.

export class MetricsLog {
  constructor(maxEntries) {
    this.maxEntries = maxEntries || 500;
    this.entries = [];
  }

  push(snapshot) {
    this.entries.push(snapshot);
    if (this.entries.length > this.maxEntries) {
      this.entries.shift();
    }
  }

  getSeries(key) {
    var values = [];
    for (var i = 0; i < this.entries.length; i++) {
      values.push(this.entries[i][key]);
    }
    return values;
  }

  getGenerations() {
    return this.getSeries('generation');
  }

  last() {
    return this.entries.length > 0 ? this.entries[this.entries.length - 1] : null;
  }

  get length() {
    return this.entries.length;
  }
}
