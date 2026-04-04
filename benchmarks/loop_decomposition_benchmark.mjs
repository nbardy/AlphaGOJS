#!/usr/bin/env node
/**
 * Compares GPU worker loop modes (query-driven, no production impact when unused):
 *   - sim_random: legal random moves, no replay/train
 *   - sim_forward: real policy forward, no train
 *   - full: default RL (omit benchLoop)
 *
 * Optional benchInstrument=1 adds worker-reported ms/tick (policy vs physics) and train wall time.
 * benchMinimalUi=1 throttles board snapshots to reduce postMessage overhead.
 * --webgpuEnv=1 adds webgpuEnv=1 to the page URL (WebGPU sim in worker vs TF tensor env).
 *
 * Prereq: npm run build, npx puppeteer browsers install chrome
 * Run: npm run bench:loop
 */
import {
  getPuppeteerLaunchOptions,
  loadPuppeteer,
  resolveBuiltAppFileUrl,
  waitForAppReady
} from './puppeteer_bench_common.mjs';
import {
  computePercentDelta,
  emitBenchmarkReport,
  formatNumber,
  formatSignedPercent,
  prepareBenchmarkOutput
} from './benchmark_output.mjs';

function clampInt(v, fallback, min, max) {
  const n = Number.parseInt(v, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function clampFloat(v, fallback, min, max) {
  const n = Number.parseFloat(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function parseArgs(argv) {
  const out = {
    warmupSec: 2,
    durationSec: 8,
    runs: 2,
    ticksPerFrame: 20,
    pipeline: 'single_gpu_phased',
    timeoutMs: 180000,
    headless: true,
    instrument: true,
    minimalUi: true,
    webgpuEnv: false
  };
  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const [key, value = ''] = arg.slice(2).split('=', 2);
    if (key === 'warmup') out.warmupSec = clampFloat(value, out.warmupSec, 0, 120);
    else if (key === 'duration') out.durationSec = clampFloat(value, out.durationSec, 1, 600);
    else if (key === 'runs') out.runs = clampInt(value, out.runs, 1, 10);
    else if (key === 'ticks') out.ticksPerFrame = clampInt(value, out.ticksPerFrame, 1, 200);
    else if (key === 'pipeline') out.pipeline = value || out.pipeline;
    else if (key === 'timeoutMs') out.timeoutMs = clampInt(value, out.timeoutMs, 10000, 900000);
    else if (key === 'headless') out.headless = value !== 'false';
    else if (key === 'instrument') out.instrument = value !== 'false';
    else if (key === 'minimalUi') out.minimalUi = value !== 'false';
    else if (key === 'webgpuEnv') out.webgpuEnv = value !== 'false' && value !== '0';
  }
  return out;
}

async function waitForWorkerReady(page, timeoutMs) {
  await page.waitForFunction(() => {
    const ui = window.__alphaPlague;
    if (!ui || !ui.trainer) return false;
    if (typeof ui.trainer._ready === 'boolean') return !!ui.trainer._ready;
    return true;
  }, { timeout: timeoutMs });
}

function buildBenchUrl(baseUrl, cfg, mode) {
  const u = new URL(baseUrl);
  u.searchParams.set('pipeline', cfg.pipeline);
  if (cfg.minimalUi) u.searchParams.set('benchMinimalUi', '1');
  if (cfg.instrument) u.searchParams.set('benchInstrument', '1');
  if (mode === 'sim_random') u.searchParams.set('benchLoop', 'sim_random');
  else if (mode === 'sim_forward') u.searchParams.set('benchLoop', 'sim_forward');
  if (cfg.webgpuEnv) u.searchParams.set('webgpuEnv', '1');
  return u.toString();
}

async function runThroughputSample(page, durationMs, ticksPerFrame) {
  return page.evaluate(async ({ durationMs: d, ticksPerFrame: tf }) => {
    const ui = window.__alphaPlague;
    if (!ui || !ui.trainer) throw new Error('UI trainer not ready');

    const stats0 = ui.trainer.getStats();
    const g0 = Number(stats0.gamesCompleted) || 0;
    const gen0 = Number(stats0.generation) || 0;
    const ts0 = Number(stats0.trainSteps) || 0;

    const paused0 = !!ui.paused;
    const ticks0 = Number(ui.ticksPerFrame) || 1;
    ui.paused = false;
    ui.ticksPerFrame = tf;

    const t0 = performance.now();
    await new Promise((r) => setTimeout(r, d));
    const t1 = performance.now();

    // Allow one more worker round-trip so getStats() includes bench averages from a tick batch.
    await new Promise((r) => setTimeout(r, 300));
    const stats1 = ui.trainer.getStats();
    ui.paused = paused0;
    ui.ticksPerFrame = ticks0;

    const sec = Math.max(0.001, (t1 - t0) / 1000);
    const g1 = Number(stats1.gamesCompleted) || 0;
    return {
      durationSec: sec,
      gamesPerSec: (g1 - g0) / sec,
      generationsPerMin: ((Number(stats1.generation) || 0) - gen0) / sec * 60,
      trainStepsPerSec: ((Number(stats1.trainSteps) || 0) - ts0) / sec,
      endStats: stats1
    };
  }, { durationMs, ticksPerFrame });
}

async function main() {
  const argv = process.argv.slice(2);
  const cfg = parseArgs(argv);
  const output = prepareBenchmarkOutput('loop_decomposition_benchmark', argv);
  const { fileUrl: baseUrl } = resolveBuiltAppFileUrl(process.cwd());

  const puppeteer = await loadPuppeteer();
  const browser = await puppeteer.launch(
    getPuppeteerLaunchOptions({
      headless: cfg.headless,
      protocolTimeout: 600000
    })
  );

  const modes = [
    { id: 'sim_random', label: 'sim_random (no forward/train)' },
    { id: 'sim_forward', label: 'sim_forward (forward, no train)' },
    { id: 'full', label: 'full RL (default)' }
  ];

  const rows = [];

  try {
    const page = await browser.newPage();

    for (const m of modes) {
      const url = buildBenchUrl(baseUrl, cfg, m.id);
      await page.goto(url, { waitUntil: 'load' });
      await waitForAppReady(page, cfg.timeoutMs);
      await waitForWorkerReady(page, cfg.timeoutMs);

      if (cfg.warmupSec > 0) {
        await runThroughputSample(page, Math.round(cfg.warmupSec * 1000), cfg.ticksPerFrame);
      }

      const samples = [];
      for (let i = 0; i < cfg.runs; i++) {
        samples.push(await runThroughputSample(page, Math.round(cfg.durationSec * 1000), cfg.ticksPerFrame));
      }

      const med = (arr) => {
        const s = arr.slice().sort((a, b) => a - b);
        return s[Math.floor(s.length / 2)];
      };

      const gps = samples.map((s) => s.gamesPerSec);
      const last = samples[samples.length - 1];
      const es = last.endStats || {};

      rows.push({
        mode: m.id,
        label: m.label,
        gamesPerSecMedian: med(gps),
        gamesPerSecMin: Math.min(...gps),
        gamesPerSecMax: Math.max(...gps),
        trainStepsPerSecMedian: med(samples.map((s) => s.trainStepsPerSec)),
        benchAvgPolicyMsPerSimTick: es.benchAvgPolicyMsPerSimTick,
        benchAvgPhysicsMsPerSimTick: es.benchAvgPhysicsMsPerSimTick,
        benchTrainMs: es.benchTrainMs,
        benchTrainCalls: es.benchTrainCalls,
        gpuReadbackBytes: es.gpuReadbackBytes
      });
    }

    const byMode = Object.fromEntries(rows.map((row) => [row.mode, row]));
    const randomGps = byMode.sim_random ? byMode.sim_random.gamesPerSecMedian : NaN;
    const forwardGps = byMode.sim_forward ? byMode.sim_forward.gamesPerSecMedian : NaN;
    const fullGps = byMode.full ? byMode.full.gamesPerSecMedian : NaN;
    const result = {
      benchmark: 'loop_decomposition_benchmark',
      config: cfg,
      rows,
      comparisons: {
        simForwardVsRandomPercent: computePercentDelta(forwardGps, randomGps),
        fullVsForwardPercent: computePercentDelta(fullGps, forwardGps),
        fullVsRandomPercent: computePercentDelta(fullGps, randomGps)
      }
    };

    const summaryLines = [
      'Loop decomposition benchmark (GPU worker query modes)',
      'pipeline=' + cfg.pipeline
        + ' duration=' + cfg.durationSec + 's runs=' + cfg.runs
        + ' ticks/frame=' + cfg.ticksPerFrame
        + ' instrument=' + cfg.instrument
        + ' minimalUi=' + cfg.minimalUi
        + ' webgpuEnv=' + cfg.webgpuEnv
    ];

    for (const r of rows) {
      let deltaText = '';
      if (r.mode === 'sim_forward') {
        deltaText = ' delta_vs_sim_random=' + formatSignedPercent(result.comparisons.simForwardVsRandomPercent);
      } else if (r.mode === 'full') {
        deltaText = ' delta_vs_sim_forward=' + formatSignedPercent(result.comparisons.fullVsForwardPercent)
          + ' delta_vs_sim_random=' + formatSignedPercent(result.comparisons.fullVsRandomPercent);
      }

      let line = r.mode
        + ' games/s=' + formatNumber(r.gamesPerSecMedian, 1)
        + ' range=' + formatNumber(r.gamesPerSecMin, 1) + '..' + formatNumber(r.gamesPerSecMax, 1)
        + ' trainSteps/s=' + formatNumber(r.trainStepsPerSecMedian, 3)
        + deltaText;
      if (typeof r.benchAvgPolicyMsPerSimTick === 'number') {
        const pol = r.benchAvgPolicyMsPerSimTick;
        const phy = r.benchAvgPhysicsMsPerSimTick;
        const tot = pol + phy;
        const pct = tot > 0 ? (100 * pol / tot) : NaN;
        line += ' policy_ms/tick=' + formatNumber(pol, 3)
          + ' physics_ms/tick=' + formatNumber(phy, 3)
          + ' policy_share=' + formatNumber(pct, 1) + '%';
      }
      if (r.benchTrainCalls > 0) {
        line += ' train_wall_ms=' + formatNumber(Number(r.benchTrainMs), 1)
          + ' train_calls=' + r.benchTrainCalls;
      }
      summaryLines.push(line);
    }

    emitBenchmarkReport(output, result, summaryLines);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error('Loop decomposition benchmark failed:', err && err.message ? err.message : err);
  process.exitCode = 1;
});
