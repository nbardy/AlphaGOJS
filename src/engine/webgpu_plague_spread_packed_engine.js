/**
 * WebGPU plague spread on **2-bit packed** boards (16 cells / u32).
 * WGSL: `plague_spread_packed.wgsl` — same RNG + rules as unpacked `spread_pass`.
 */

import { copyGpuBufferToUint32 } from './webgpu_copy_read_u32.js';
import { wordsPerGameBoard } from './plague_spread_pack_2bit.js';

export class WebGPUPlagueSpreadPackedEngine {
  /**
   * @param {GPUDevice} device
   * @param {{ rows: number, cols: number, numGames: number }} config
   * @param {string} wgslSource full plague_spread_packed.wgsl
   */
  constructor(device, config, wgslSource) {
    if (!wgslSource || typeof wgslSource !== 'string') {
      throw new Error('WebGPUPlagueSpreadPackedEngine: wgslSource string required');
    }
    this.device = device;
    this.rows = config.rows | 0;
    this.cols = config.cols | 0;
    this.numGames = config.numGames | 0;
    this.boardSize = this.rows * this.cols;
    this.wordsPerGame = wordsPerGameBoard(this.rows, this.cols);
    this.packedLength = this.wordsPerGame * this.numGames;
    this._tick = 0 >>> 0;

    var bytes = this.packedLength * 4;
    this.bufA = device.createBuffer({
      size: Math.max(4, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });
    this.bufB = device.createBuffer({
      size: Math.max(4, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    });

    this.uniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    var shaderModule = device.createShaderModule({ code: wgslSource });
    var bindLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
      ]
    });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindLayout] }),
      compute: { module: shaderModule, entryPoint: 'spread_packed_pass' }
    });

    this._bindReadA = device.createBindGroup({
      layout: bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bufA } },
        { binding: 2, resource: { buffer: this.bufB } }
      ]
    });
    this._bindReadB = device.createBindGroup({
      layout: bindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.bufB } },
        { binding: 2, resource: { buffer: this.bufA } }
      ]
    });

    this._readIsA = true;
    this._workgroups = Math.ceil(this.packedLength / 64);
    this._writeUniform();
  }

  _writeUniform() {
    var u = new Uint32Array([
      this.rows >>> 0,
      this.cols >>> 0,
      this._tick >>> 0,
      this.numGames >>> 0
    ]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, u);
  }

  resetSimulationState() {
    this._tick = 0 >>> 0;
    this._readIsA = true;
    this._writeUniform();
  }

  uploadPackedWords(packedWords) {
    if (packedWords.length !== this.packedLength) {
      throw new Error(
        'uploadPackedWords: expected ' + this.packedLength + ' words, got ' + packedWords.length
      );
    }
    var readBuf = this._readIsA ? this.bufA : this.bufB;
    this.device.queue.writeBuffer(readBuf, 0, packedWords);
  }

  spread() {
    this._writeUniform();
    var enc = this.device.createCommandEncoder();
    var pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this._readIsA ? this._bindReadA : this._bindReadB);
    pass.dispatchWorkgroups(this._workgroups);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this._readIsA = !this._readIsA;
    this._tick = (this._tick + 1) >>> 0;
  }

  async spreadAndSync() {
    this.spread();
    await this.device.queue.onSubmittedWorkDone();
  }

  async downloadPackedWords() {
    var readBuf = this._readIsA ? this.bufA : this.bufB;
    return copyGpuBufferToUint32(this.device, readBuf, 0, this.packedLength);
  }

  dispose() {
    try { this.bufA.destroy(); } catch (e) {}
    try { this.bufB.destroy(); } catch (e) {}
    try { this.uniformBuffer.destroy(); } catch (e) {}
    this.bufA = null;
    this.bufB = null;
    this.uniformBuffer = null;
  }
}
