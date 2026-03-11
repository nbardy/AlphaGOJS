// Runtime capability probing for tiered pipeline selection.
// Keep probes defensive: failures should degrade to false, never throw.

function hasOffscreenCanvas() {
  return typeof OffscreenCanvas !== 'undefined';
}

function canOffscreenContext(kind) {
  if (!hasOffscreenCanvas()) return false;
  try {
    var c = new OffscreenCanvas(1, 1);
    return !!c.getContext(kind);
  } catch (e) {
    return false;
  }
}

export async function probeCapabilities() {
  var nav = typeof navigator !== 'undefined' ? navigator : {};
  var hasScheduler = typeof scheduler !== 'undefined';
  var hasSAB = typeof SharedArrayBuffer !== 'undefined';
  var isolated = typeof crossOriginIsolated !== 'undefined' ? !!crossOriginIsolated : false;

  var out = {
    webgpu: !!(nav && nav.gpu),
    workerWebGPU: false,
    offscreenCanvas: hasOffscreenCanvas(),
    offscreenWebGPU: canOffscreenContext('webgpu'),
    offscreenWebGL2: canOffscreenContext('webgl2'),
    sab: hasSAB && isolated,
    atomicsWaitAsync: typeof Atomics !== 'undefined' && typeof Atomics.waitAsync === 'function',
    schedulerPostTask: hasScheduler && typeof scheduler.postTask === 'function',
    schedulerYield: hasScheduler && typeof scheduler.yield === 'function',
    compatibilityMode: false
  };

  // Probe worker WebGPU without requiring app-wide worker architecture.
  // Worker script is tiny and terminates immediately.
  if (typeof Worker !== 'undefined') {
    try {
      var blob = new Blob([
        'self.postMessage({ ok: !!(self.navigator && self.navigator.gpu) });'
      ], { type: 'application/javascript' });
      var url = URL.createObjectURL(blob);
      out.workerWebGPU = await new Promise(function (resolve) {
        var done = false;
        var w = new Worker(url);
        var finish = function (v) {
          if (done) return;
          done = true;
          resolve(!!v);
          try { w.terminate(); } catch (e) {}
          try { URL.revokeObjectURL(url); } catch (e) {}
        };
        var timer = setTimeout(function () { finish(false); }, 250);
        w.onmessage = function (ev) {
          clearTimeout(timer);
          finish(ev && ev.data && ev.data.ok);
        };
        w.onerror = function () {
          clearTimeout(timer);
          finish(false);
        };
      });
    } catch (e) {
      out.workerWebGPU = false;
    }
  }

  // Feature-level probe for compatibility mode is backend-specific and costly;
  // leave false by default and let GPU runtime set this after adapter request.
  return out;
}

