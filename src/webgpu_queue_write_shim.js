/**
 * Some browsers cap a single GPUQueue.writeBuffer size. TensorFlow.js and other code may
 * issue large writes; chunk here so all call sites are covered (main thread + GPU worker).
 * Load before @tensorflow/tfjs-backend-webgpu (see tf_backend_bootstrap.js).
 */
(function installWebGpuQueueWriteBufferChunkShim() {
  if (typeof GPUQueue === 'undefined' || !GPUQueue.prototype) return;
  var proto = GPUQueue.prototype;
  if (proto.writeBuffer.__alphaPlagueChunkShim) return;

  /** Multiple of 256 for dest alignment; keep small for strict per-call caps (WebKit). */
  var CHUNK = 256;
  var orig = proto.writeBuffer;

  proto.writeBuffer = function (destination, destinationOffset, data, dataOffset, size) {
    var argc = arguments.length;
    if (argc < 3) {
      return orig.apply(this, arguments);
    }

    var destOff = Number(destinationOffset) || 0;
    var dOff = argc >= 4 ? Number(dataOffset) : 0;
    if (!Number.isFinite(dOff)) dOff = 0;

    var total;
    if (argc >= 5 && arguments[4] !== undefined) {
      total = Number(size);
    } else if (data instanceof ArrayBuffer) {
      total = data.byteLength - dOff;
    } else if (ArrayBuffer.isView(data)) {
      total = data.byteLength - dOff;
    } else {
      return orig.apply(this, arguments);
    }

    if (!Number.isFinite(total) || total <= 0) {
      return orig.apply(this, arguments);
    }
    if (total <= CHUNK) {
      return orig.apply(this, arguments);
    }

    var backing;
    var base;
    if (data instanceof ArrayBuffer) {
      backing = data;
      base = dOff;
    } else if (ArrayBuffer.isView(data)) {
      backing = data.buffer;
      base = data.byteOffset + dOff;
    } else {
      return orig.apply(this, arguments);
    }

    var u8 = new Uint8Array(backing, base, total);
    var pos = 0;
    while (pos < total) {
      var n = Math.min(CHUNK, total - pos);
      orig.call(this, destination, destOff + pos, u8, pos, n);
      pos += n;
    }
  };

  proto.writeBuffer.__alphaPlagueChunkShim = true;
})();
