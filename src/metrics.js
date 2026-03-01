// Per-generation metrics log.
// Stores ALL entries (full run history). getSeries() downsamples via
// largest-triangle-three-buckets (LTTB) when the series exceeds maxPoints,
// preserving visual shape while keeping chart rendering fast.

export class MetricsLog {
  constructor(maxPoints) {
    this.maxPoints = maxPoints || 500;
    this.entries = [];
  }

  push(snapshot) {
    this.entries.push(snapshot);
  }

  getSeries(key) {
    var n = this.entries.length;
    if (n <= this.maxPoints) {
      var values = [];
      for (var i = 0; i < n; i++) values.push(this.entries[i][key]);
      return values;
    }
    // LTTB downsampling: keeps first/last, picks representative points
    return this._lttb(key, this.maxPoints);
  }

  // Largest-Triangle-Three-Buckets downsampling.
  // Preserves visual shape of time series better than uniform sampling.
  _lttb(key, threshold) {
    var data = this.entries;
    var n = data.length;
    if (n <= threshold) {
      var out = [];
      for (var i = 0; i < n; i++) out.push(data[i][key]);
      return out;
    }

    var sampled = [];
    // Always include first point
    sampled.push(data[0][key]);

    var bucketSize = (n - 2) / (threshold - 2);
    var a = 0; // index of previously selected point

    for (var i = 1; i < threshold - 1; i++) {
      // Calculate bucket boundaries
      var bucketStart = Math.floor((i - 1) * bucketSize) + 1;
      var bucketEnd = Math.floor(i * bucketSize) + 1;
      if (bucketEnd > n - 1) bucketEnd = n - 1;

      // Calculate average of next bucket for triangle area
      var nextStart = Math.floor(i * bucketSize) + 1;
      var nextEnd = Math.floor((i + 1) * bucketSize) + 1;
      if (nextEnd > n - 1) nextEnd = n - 1;
      var avgX = 0, avgY = 0, nextCount = 0;
      for (var j = nextStart; j <= nextEnd; j++) {
        avgX += j;
        avgY += data[j][key];
        nextCount++;
      }
      if (nextCount > 0) { avgX /= nextCount; avgY /= nextCount; }

      // Find point in current bucket with max triangle area
      var maxArea = -1;
      var maxIdx = bucketStart;
      var aX = a, aY = data[a][key];
      for (var j = bucketStart; j <= bucketEnd; j++) {
        var area = Math.abs((aX - avgX) * (data[j][key] - aY) - (aX - j) * (avgY - aY));
        if (area > maxArea) {
          maxArea = area;
          maxIdx = j;
        }
      }

      sampled.push(data[maxIdx][key]);
      a = maxIdx;
    }

    // Always include last point
    sampled.push(data[n - 1][key]);
    return sampled;
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
