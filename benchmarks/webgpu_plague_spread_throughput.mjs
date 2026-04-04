#!/usr/bin/env node
/**
 * Throughput of native WGSL plague spread vs CPU reference (same math/RNG).
 * Compare to end-to-end RL via `npm run bench:loop` (games/s includes policy/train).
 *
 * Prereq: WebGPU adapter (browser with WebGPU, or `webgpu` Dawn in Node).
 * Run: npm run bench:webgpu:spread
 *      node benchmarks/webgpu_plague_spread_throughput.mjs --rows=20 --cols=20 --numGames=80
 *
 * If no adapter: prints skipped JSON and exits 0 (CI-friendly).
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { initWebGPUDevice } from './webgpu_node_helpers.mjs';
import { WebGPUPlagueSpreadEngine } from '../src/engine/webgpu_plague_spread_engine.js';
import { spreadPackedAllGamesInOut } from '../src/engine/plague_spread_cpu.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WGSL_PATH = join(__dirname, '../src/engine/wgsl/plague_spread.wgsl');

function clampInt(v, fallback, min, max) {
  const n = Number.parseInt(v, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function parseArgs(argv) {
  const out = {
    rows: 20,
    cols: 20,
    numGames: 80,
    warmup: 5,
    runs: 10,
    spreadsPerSample: 256
  };
  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const [key, value = ''] = arg.slice(2).split('=', 2);
    if (key === 'rows') out.rows = clampInt(value, out.rows, 4, 32);
    else if (key === 'cols') out.cols = clampInt(value, out.cols, 4, 32);
    else if (key === 'numGames') out.numGames = clampInt(value, out.numGames, 1, 4096);
    else if (key === 'warmup') out.warmup = clampInt(value, out.warmup, 0, 100);
    else if (key === 'runs') out.runs = clampInt(value, out.runs, 1, 50);
    else if (key === 'spreads') out.spreadsPerSample = clampInt(value, out.spreadsPerSample, 16, 100000);
  }
  return out;
}

function percentile(sorted, p) {
  if (!sorted.length) return NaN;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const w = idx - lo;
  return sorted[lo] * (1 - w) + sorted[hi] * w;
}

function summarize(samples) {
  if (!samples.length) {
    return { count: 0, min: NaN, max: NaN, mean: NaN, median: NaN, p95: NaN };
  }
  const sorted = samples.slice().sort((a, b) => a - b);
  const sum = samples.reduce((a, b) => a + b, 0);
  return {
    count: samples.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / samples.length,
    median: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95)
  };
}

function makeInitialPacked(rows, cols, numGames) {
  const boardSize = rows * cols;
  const packed = new Uint32Array(boardSize * numGames);
  for (let g = 0; g < numGames; g++) {
    const base = g * boardSize;
    for (let i = 0; i < boardSize; i++) {
      packed[base + i] = (i + g * 7) % 4;
    }
  }
  return packed;
}

async function runGpuSample(engine, cfg, spreadsPerSample) {
  const { rows, cols, numGames } = cfg;
  const packed = makeInitialPacked(rows, cols, numGames);
  engine.resetSimulationState();
  engine.uploadPacked(packed);

  const t0 = performance.now();
  for (let i = 0; i < spreadsPerSample; i++) {
    engine.spread();
  }
  await engine.device.queue.onSubmittedWorkDone();
  const t1 = performance.now();
  return (t1 - t0) / 1000;
}

function runCpuSample(cfg, spreadsPerSample) {
  const { rows, cols, numGames } = cfg;
  const boardSize = rows * cols;
  const len = boardSize * numGames;
  let a = makeInitialPacked(rows, cols, numGames);
  let b = new Uint32Array(len);
  let tick = 0;

  const t0 = performance.now();
  for (let i = 0; i < spreadsPerSample; i++) {
    spreadPackedAllGamesInOut(a, b, rows, cols, numGames, tick);
    tick = (tick + 1) >>> 0;
    const tmp = a;
    a = b;
    b = tmp;
  }
  const t1 = performance.now();
  return (t1 - t0) / 1000;
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2));
  const boardSize = cfg.rows * cfg.cols;
  const cellsPerSpread = boardSize * cfg.numGames;

  const { ok, device, provider } = await initWebGPUDevice();

  const result = {
    benchmark: 'webgpu_plague_spread_throughput',
    config: cfg,
    boardSize,
    cellsPerSpread,
    skipped: !ok,
    provider: ok ? provider : null,
    note: 'WGSL spread pass only (not TF.js, not policy). Compare to bench:loop games/s for full RL.'
  };

  if (!ok || !device) {
    console.log('WebGPU plague spread throughput — SKIPPED (no adapter). Install `webgpu` or use a GPU-capable machine.');
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  const wgsl = readFileSync(WGSL_PATH, 'utf8');
  const engine = new WebGPUPlagueSpreadEngine(device, { rows: cfg.rows, cols: cfg.cols, numGames: cfg.numGames }, wgsl);

  try {
    const warmN = Math.min(32, cfg.spreadsPerSample);
    for (let w = 0; w < cfg.warmup; w++) {
      await runGpuSample(engine, cfg, warmN);
    }

    const gpuSecs = [];
    const cpuSecs = [];
    for (let r = 0; r < cfg.runs; r++) {
      gpuSecs.push(await runGpuSample(engine, cfg, cfg.spreadsPerSample));
      cpuSecs.push(runCpuSample(cfg, cfg.spreadsPerSample));
    }

  const gpuMed = summarize(gpuSecs).median;
  const cpuMed = summarize(cpuSecs).median;
  const spreadsPerSecGpu = cfg.spreadsPerSample / Math.max(1e-9, gpuMed);
  const spreadsPerSecCpu = cfg.spreadsPerSample / Math.max(1e-9, cpuMed);
  const cellUpdatesPerSecGpu = spreadsPerSecGpu * cellsPerSpread;

  result.gpu = {
    spreadsPerSample: cfg.spreadsPerSample,
    wallSecSamples: gpuSecs,
    wallSec: summarize(gpuSecs),
    spreadsPerSecMedian: spreadsPerSecGpu,
    cellUpdatesPerSecMedian: cellUpdatesPerSecGpu
  };
  result.cpu = {
    spreadsPerSample: cfg.spreadsPerSample,
    wallSecSamples: cpuSecs,
    wallSec: summarize(cpuSecs),
    spreadsPerSecMedian: spreadsPerSecCpu
  };
  result.ratioGpuVsCpuSpreadsPerSec = spreadsPerSecCpu > 0 ? spreadsPerSecGpu / spreadsPerSecCpu : null;

  console.log('WebGPU plague spread throughput (WGSL spread only vs CPU reference)');
  console.log(
    'rows=' + cfg.rows + ' cols=' + cfg.cols + ' numGames=' + cfg.numGames
    + ' cells/spread=' + cellsPerSpread
    + ' spreads/sample=' + cfg.spreadsPerSample
    + ' runs=' + cfg.runs
    + ' provider=' + provider
  );
  console.log(
    'GPU  spreads/s ~' + spreadsPerSecGpu.toFixed(0)
    + '  (cell-updates/s ~' + cellUpdatesPerSecGpu.toExponential(2) + ')'
    + '  wall_ms/sample p50=' + (gpuMed * 1000).toFixed(2)
  );
  console.log(
    'CPU  spreads/s ~' + spreadsPerSecCpu.toFixed(0)
    + '  wall_ms/sample p50=' + (cpuMed * 1000).toFixed(2)
  );
  if (result.ratioGpuVsCpuSpreadsPerSec != null) {
    console.log('GPU/CPU spread throughput ratio ~' + result.ratioGpuVsCpuSpreadsPerSec.toFixed(1) + 'x');
  }
  console.log('');
  console.log('Compare: npm run bench:loop  → games/s includes sim_random|sim_forward|full RL (policy + physics + train).');
  console.log(JSON.stringify(result, null, 2));
  } finally {
    engine.dispose();
  }
}

main().catch((e) => {
  console.error(e.message || e);
  process.exitCode = 1;
});
