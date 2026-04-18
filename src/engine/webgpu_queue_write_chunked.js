/**
 * Some WebGPU stacks cap the size of a single GPUQueue.writeBuffer (e.g. strict WebKit /
 * "Number of bytes to write is too large"). Chunk uploads stay under that cap.
 */
export var WEBGPU_QUEUE_WRITE_CHUNK_BYTES = 64 * 1024;

export function queueWriteBufferChunked(queue, gpuBuffer, dstByteOffset, srcView, srcByteOffset, totalBytes) {
  var maxChunk = WEBGPU_QUEUE_WRITE_CHUNK_BYTES;
  var written = 0;
  while (written < totalBytes) {
    var chunk = Math.min(maxChunk, totalBytes - written);
    queue.writeBuffer(gpuBuffer, dstByteOffset + written, srcView, srcByteOffset + written, chunk);
    written += chunk;
  }
}
