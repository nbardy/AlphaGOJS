/**
 * Shared MAP_READ staging copy for WebGPU benchmarks + game engine (avoid duplicating
 * copyBufferToBuffer / mapAsync / Uint32Array view gotchas).
 */
export async function copyGpuBufferToUint32(device, srcBuffer, srcByteOffset, numU32) {
  var bytes = numU32 * 4;
  var staging = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  var enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(srcBuffer, srcByteOffset, staging, 0, bytes);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  var mapped = staging.getMappedRange();
  var out = new Uint32Array(numU32);
  out.set(new Uint32Array(mapped, 0, numU32));
  staging.unmap();
  staging.destroy();
  return out;
}
