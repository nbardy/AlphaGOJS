import { MSG } from '../protocol/messages';
import { GPUOwnerRuntime } from '../runtime/gpu_owner_runtime';

var runtime = new GPUOwnerRuntime(function (msg, transfer) {
  self.postMessage(msg, transfer || []);
});

var queue = Promise.resolve();

function runSerial(fn) {
  queue = queue.then(function () {
    return fn();
  }).catch(function (e) {
    self.postMessage({
      type: MSG.ERROR,
      message: e && e.message ? e.message : 'Unknown worker error'
    });
  });
}

self.onmessage = function (ev) {
  var data = ev && ev.data ? ev.data : {};
  var type = data.type;

  if (type === MSG.INIT) {
    runSerial(function () { return runtime.init(data.config || {}); });
    return;
  }
  if (type === MSG.RESTART) {
    runSerial(function () { return runtime.restart(data.config || {}); });
    return;
  }
  if (type === MSG.TICK) {
    runSerial(function () { return runtime.tick(data.steps || 1); });
    return;
  }
  if (type === MSG.INFER_ACTION) {
    runSerial(function () {
      return runtime.inferAction(data.requestId, data.state, data.mask);
    });
    return;
  }
  if (type === MSG.DISPOSE) {
    runSerial(function () { return runtime.dispose(); });
    return;
  }
};

