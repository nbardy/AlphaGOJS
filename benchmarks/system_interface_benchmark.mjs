#!/usr/bin/env node
/**
 * End-to-end UI + worker throughput (Puppeteer opens `docs/index.html` after `npm run build`).
 *
 * **Wall time:** Defaults use 3 pipelines × (warmup + timed runs + **120× `selectActionAsync`**
 * round-trips per pipeline). That can exceed **30+ minutes** and look “hung.” Prefer
 * `--pipelines=single_gpu_phased`, `--inferenceRuns=10`, shorter `--duration` for a quick check.
 * The **`bench:all`** aggregator passes fewer pipelines + `--inferenceRuns=24` unless `--full-system`.
 *
 * **Chrome:** `npx puppeteer browsers install chrome` if Puppeteer cannot find the binary.
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

function parseNumberList(csv, min, max) {
  if (!csv) return [];
  return csv
    .split(',')
    .map((s) => Number.parseFloat(s.trim()))
    .filter((n) => Number.isFinite(n) && n >= min && n <= max);
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
  const sum = samples.reduce((acc, v) => acc + v, 0);
  return {
    count: samples.length,
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / samples.length,
    median: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95)
  };
}

function parseArgs(argv) {
  const out = {
    warmupSec: 3,
    durationSec: 10,
    runs: 2,
    ticksPerFrame: 20,
    inferenceRuns: 60,
    inferenceTimeoutMs: 5000,
    protocolTimeoutMs: 600000,
    model: 'spatial_lite',
    algo: 'ppo',
    algos: [],
    qualityWindowsSec: [],
    game: 'plague_walls',
    pipelines: ['single_gpu_phased', 'cpu_actors_gpu_learner', 'full_gpu_resident'],
    timeoutMs: 180000,
    headless: true
  };

  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const [key, value = ''] = arg.slice(2).split('=', 2);
    if (key === 'warmup') out.warmupSec = clampFloat(value, out.warmupSec, 0, 120);
    else if (key === 'duration') out.durationSec = clampFloat(value, out.durationSec, 1, 600);
    else if (key === 'runs') out.runs = clampInt(value, out.runs, 1, 20);
    else if (key === 'ticks') out.ticksPerFrame = clampInt(value, out.ticksPerFrame, 1, 200);
    else if (key === 'inferenceRuns') out.inferenceRuns = clampInt(value, out.inferenceRuns, 1, 5000);
    else if (key === 'inferenceTimeoutMs') out.inferenceTimeoutMs = clampInt(value, out.inferenceTimeoutMs, 10, 120000);
    else if (key === 'protocolTimeoutMs') out.protocolTimeoutMs = clampInt(value, out.protocolTimeoutMs, 10000, 1800000);
    else if (key === 'model') out.model = value || out.model;
    else if (key === 'algo') out.algo = value || out.algo;
    else if (key === 'algos') out.algos = value.split(',').map((s) => s.trim()).filter(Boolean);
    else if (key === 'qualityWindows') out.qualityWindowsSec = parseNumberList(value, 1, 86400);
    else if (key === 'game') out.game = value || out.game;
    else if (key === 'pipelines') out.pipelines = value.split(',').map((s) => s.trim()).filter(Boolean);
    else if (key === 'timeoutMs') out.timeoutMs = clampInt(value, out.timeoutMs, 10000, 900000);
    else if (key === 'headless') out.headless = value !== 'false';
  }

  return out;
}

async function configureAndRestart(page, cfg, pipeline, algoOverride) {
  await page.evaluate(({ model, algo, game, pipeline }) => {
    const setSel = (id, value) => {
      const el = document.getElementById(id);
      if (!el) throw new Error('Missing select: ' + id);
      el.value = value;
      el.dispatchEvent(new Event('change', { bubbles: true }));
    };

    setSel('model-sel', model);
    setSel('algo-sel', algo);
    setSel('game-sel', game);
    setSel('pipe-sel', pipeline);

    const btn = document.querySelector('button.restart-btn');
    if (!btn) throw new Error('Missing restart button');
    btn.click();
  }, { model: cfg.model, algo: algoOverride || cfg.algo, game: cfg.game, pipeline });

  await page.waitForFunction((pipelineName) => {
    const ui = window.__alphaPlague;
    if (!ui || !ui.trainer) return false;
    if (ui.pipelineType !== pipelineName) return false;
    if (typeof ui.trainer._ready === 'boolean') {
      return !!ui.trainer._ready;
    }
    return true;
  }, { timeout: cfg.timeoutMs }, pipeline);
}

async function runThroughputSample(page, opts) {
  return page.evaluate(async ({ durationMs, ticksPerFrame }) => {
    const ui = window.__alphaPlague;
    if (!ui || !ui.trainer) throw new Error('UI trainer not ready');

    const toStats = (s) => ({
      gamesCompleted: Number(s && s.gamesCompleted) || 0,
      generation: Number(s && s.generation) || 0,
      bufferSize: Number(s && s.bufferSize) || 0,
      trainSteps: Number(s && s.trainSteps) || 0
    });

    const getTrainSteps = () => {
      try {
        if (ui.trainer && typeof ui.trainer.getTrainSteps === 'function') {
          return Number(ui.trainer.getTrainSteps()) || 0;
        }
        if (ui.algo && typeof ui.algo.getTrainSteps === 'function') {
          return Number(ui.algo.getTrainSteps()) || 0;
        }
      } catch (e) {}
      return 0;
    };

    const stats0 = toStats(ui.trainer.getStats());
    stats0.trainSteps = getTrainSteps() || stats0.trainSteps;

    const frame0 = Number(ui._frameCount) || 0;
    const paused0 = !!ui.paused;
    const ticks0 = Number(ui.ticksPerFrame) || 1;

    ui.paused = false;
    ui.ticksPerFrame = ticksPerFrame;

    const rafIntervals = [];
    let running = true;
    let last = performance.now();

    function onFrame(ts) {
      if (!running) return;
      rafIntervals.push(ts - last);
      last = ts;
      requestAnimationFrame(onFrame);
    }
    requestAnimationFrame(onFrame);

    const t0 = performance.now();
    await new Promise((resolve) => setTimeout(resolve, durationMs));
    const t1 = performance.now();

    running = false;

    const stats1 = toStats(ui.trainer.getStats());
    stats1.trainSteps = getTrainSteps() || stats1.trainSteps;
    const frame1 = Number(ui._frameCount) || 0;

    ui.paused = paused0;
    ui.ticksPerFrame = ticks0;

    const seconds = Math.max(0.001, (t1 - t0) / 1000);
    const deltaGames = stats1.gamesCompleted - stats0.gamesCompleted;
    const deltaGen = stats1.generation - stats0.generation;
    const deltaTrainSteps = stats1.trainSteps - stats0.trainSteps;
    const deltaFrames = frame1 - frame0;

    const sorted = rafIntervals.slice().sort((a, b) => a - b);
    const idx95 = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * 0.95)));
    const rafP95 = sorted.length ? sorted[idx95] : NaN;
    const rafMedian = sorted.length ? sorted[Math.floor((sorted.length - 1) * 0.5)] : NaN;

    const queueDepth = (ui.trainer && Number(ui.trainer._queuedSteps)) || 0;

    return {
      durationSec: seconds,
      deltaGames,
      deltaGenerations: deltaGen,
      deltaTrainSteps,
      deltaFrames,
      gamesPerSec: deltaGames / seconds,
      generationsPerMin: (deltaGen / seconds) * 60,
      trainStepsPerSec: deltaTrainSteps / seconds,
      framesPerSec: deltaFrames / seconds,
      estimatedTicksPerSec: (deltaFrames * ticksPerFrame) / seconds,
      rafMedianMs: rafMedian,
      rafP95Ms: rafP95,
      startStats: stats0,
      endStats: stats1,
      queueDepth
    };
  }, opts);
}

async function runInferenceSample(page, opts) {
  return page.evaluate(async ({ runs, inferenceTimeoutMs }) => {
    const ui = window.__alphaPlague;
    if (!ui || !ui.trainer) throw new Error('UI trainer not ready');

    const source = ui.algo || ui.trainer;
    if (!source) throw new Error('No action source');

    const boardSize = ui.rows * ui.cols;
    const mask = new Float32Array(boardSize);
    mask.fill(1);

    let seed = 1337;
    const latencies = [];
    let timeoutCount = 0;

    const paused0 = !!ui.paused;
    ui.paused = true;

    for (let i = 0; i < runs; i++) {
      const state = new Float32Array(boardSize);
      for (let j = 0; j < boardSize; j++) {
        seed = (1664525 * seed + 1013904223) >>> 0;
        const u = seed / 4294967296;
        state[j] = u * 2 - 1;
      }

      const t0 = performance.now();
      var timedOut = false;
      if (source && typeof source.selectActionAsync === 'function') {
        const actionPromise = source.selectActionAsync(state, mask);
        await Promise.race([
          actionPromise,
          new Promise((resolve) => {
            setTimeout(() => {
              timedOut = true;
              resolve();
            }, inferenceTimeoutMs);
          })
        ]);
      } else if (source && typeof source.selectAction === 'function') {
        source.selectAction(state, mask);
      } else {
        throw new Error('Action source has no selectAction method');
      }
      const t1 = performance.now();
      if (timedOut) timeoutCount++;
      latencies.push(t1 - t0);
    }

    ui.paused = paused0;

    return { latencies, timeoutCount };
  }, opts);
}

async function runQualitySample(page, opts) {
  return page.evaluate(async ({ durationMs, ticksPerFrame }) => {
    const ui = window.__alphaPlague;
    if (!ui || !ui.trainer) throw new Error('UI trainer not ready');

    const toStats = (s) => ({
      gamesCompleted: Number(s && s.gamesCompleted) || 0,
      generation: Number(s && s.generation) || 0,
      bufferSize: Number(s && s.bufferSize) || 0,
      trainSteps: Number(s && s.trainSteps) || 0,
      elo: Number(s && s.elo) || 0,
      checkpointWinRate: Number(s && s.checkpointWinRate) || 0
    });

    const getTrainSteps = () => {
      try {
        if (ui.trainer && typeof ui.trainer.getTrainSteps === 'function') {
          return Number(ui.trainer.getTrainSteps()) || 0;
        }
        if (ui.algo && typeof ui.algo.getTrainSteps === 'function') {
          return Number(ui.algo.getTrainSteps()) || 0;
        }
      } catch (e) {}
      return 0;
    };

    const stats0 = toStats(ui.trainer.getStats());
    stats0.trainSteps = getTrainSteps() || stats0.trainSteps;

    const paused0 = !!ui.paused;
    const ticks0 = Number(ui.ticksPerFrame) || 1;
    ui.paused = false;
    ui.ticksPerFrame = ticksPerFrame;

    const t0 = performance.now();
    await new Promise((resolve) => setTimeout(resolve, durationMs));
    const t1 = performance.now();

    const stats1 = toStats(ui.trainer.getStats());
    stats1.trainSteps = getTrainSteps() || stats1.trainSteps;

    ui.paused = paused0;
    ui.ticksPerFrame = ticks0;

    const seconds = Math.max(0.001, (t1 - t0) / 1000);
    const deltaElo = stats1.elo - stats0.elo;
    const deltaCheckpointWinRate = stats1.checkpointWinRate - stats0.checkpointWinRate;
    const deltaTrainSteps = stats1.trainSteps - stats0.trainSteps;
    const deltaGames = stats1.gamesCompleted - stats0.gamesCompleted;
    const deltaGenerations = stats1.generation - stats0.generation;

    return {
      durationSec: seconds,
      startStats: stats0,
      endStats: stats1,
      deltaElo,
      deltaCheckpointWinRate,
      deltaTrainSteps,
      deltaGames,
      deltaGenerations,
      eloPerSec: deltaElo / seconds,
      qualityPerGpuSecond: deltaElo / seconds
    };
  }, opts);
}

function summarizeRunSet(runSet) {
  const pick = (key) => runSet.map((r) => r[key]).filter((v) => Number.isFinite(v));
  return {
    gamesPerSec: summarize(pick('gamesPerSec')),
    generationsPerMin: summarize(pick('generationsPerMin')),
    trainStepsPerSec: summarize(pick('trainStepsPerSec')),
    framesPerSec: summarize(pick('framesPerSec')),
    estimatedTicksPerSec: summarize(pick('estimatedTicksPerSec')),
    rafMedianMs: summarize(pick('rafMedianMs')),
    rafP95Ms: summarize(pick('rafP95Ms')),
    queueDepth: summarize(pick('queueDepth'))
  };
}

function systemBenchLog(msg) {
  const t = new Date().toISOString().replace('T', ' ').replace('Z', '');
  console.log('[bench:system ' + t + '] ' + msg);
}

async function main() {
  const argv = process.argv.slice(2);
  const cfg = parseArgs(argv);
  const output = prepareBenchmarkOutput('system_interface_benchmark', argv);
  const algoList = cfg.algos.length > 0 ? cfg.algos : [cfg.algo];
  const puppeteer = await loadPuppeteer();

  const { fileUrl: url } = resolveBuiltAppFileUrl(process.cwd());

  const browser = await puppeteer.launch(
    getPuppeteerLaunchOptions({
      headless: cfg.headless,
      protocolTimeout: cfg.protocolTimeoutMs
    })
  );

  const out = {
    timestamp: new Date().toISOString(),
    config: cfg,
    environment: {},
    pipelines: {},
    algorithms: {}
  };

  try {
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'load' });
    await waitForAppReady(page, cfg.timeoutMs);

    out.environment = await page.evaluate(() => {
      const ui = window.__alphaPlague;
      return {
        userAgent: navigator.userAgent,
        initialPipeline: ui ? ui.pipelineType : 'unknown',
        hasNavigatorGPU: !!navigator.gpu,
        rows: ui ? ui.rows : 0,
        cols: ui ? ui.cols : 0,
        numGames: ui && ui.trainer ? ui.trainer.numGames : 0
      };
    });

    for (const algo of algoList) {
      out.algorithms[algo] = { pipelines: {} };
      for (const pipeline of cfg.pipelines) {
        systemBenchLog('start pipeline=' + pipeline + ' algo=' + algo + ' (restart + warmup ' + cfg.warmupSec + 's)');
        await configureAndRestart(page, cfg, pipeline, algo);

        // Warmup run (discarded)
        if (cfg.warmupSec > 0) {
          await runThroughputSample(page, {
            durationMs: Math.round(cfg.warmupSec * 1000),
            ticksPerFrame: cfg.ticksPerFrame
          });
        }

        const runs = [];
        systemBenchLog('throughput ' + cfg.runs + ' x ' + cfg.durationSec + 's …');
        for (let i = 0; i < cfg.runs; i++) {
          const sample = await runThroughputSample(page, {
            durationMs: Math.round(cfg.durationSec * 1000),
            ticksPerFrame: cfg.ticksPerFrame
          });
          runs.push(sample);
        }

        systemBenchLog('inference busy ' + cfg.inferenceRuns + ' RPCs (may take minutes on gpu_worker) …');
        const infBusyRaw = await runInferenceSample(page, {
          runs: cfg.inferenceRuns,
          inferenceTimeoutMs: cfg.inferenceTimeoutMs
        });

        // Restart to get a clean queue state for idle inference latency.
        systemBenchLog('restart + inference idle ' + cfg.inferenceRuns + ' RPCs …');
        await configureAndRestart(page, cfg, pipeline, algo);
        const infIdleRaw = await runInferenceSample(page, {
          runs: cfg.inferenceRuns,
          inferenceTimeoutMs: cfg.inferenceTimeoutMs
        });
        systemBenchLog('done pipeline=' + pipeline + ' algo=' + algo);

        const inference = {
          busyLatencyMs: summarize(infBusyRaw.latencies || []),
          idleLatencyMs: summarize(infIdleRaw.latencies || []),
          busyTimeoutCount: Number(infBusyRaw.timeoutCount) || 0,
          idleTimeoutCount: Number(infIdleRaw.timeoutCount) || 0
        };

        let quality = null;
        if (cfg.qualityWindowsSec.length > 0) {
          quality = {};
          for (const windowSec of cfg.qualityWindowsSec) {
            await configureAndRestart(page, cfg, pipeline, algo);
            const q = await runQualitySample(page, {
              durationMs: Math.round(windowSec * 1000),
              ticksPerFrame: cfg.ticksPerFrame
            });
            quality[String(windowSec)] = q;
          }
        }

        out.algorithms[algo].pipelines[pipeline] = {
          runs,
          summary: summarizeRunSet(runs),
          inference,
          quality
        };
      }
    }

    // Backward compatibility: keep top-level pipelines when only one algo runs.
    if (algoList.length === 1) {
      out.pipelines = out.algorithms[algoList[0]].pipelines;
    }
  } finally {
    await browser.close();
  }

  const summaries = [];
  for (const algo of algoList) {
    const algoPipes = out.algorithms[algo] ? out.algorithms[algo].pipelines : {};
    for (const pipeline of cfg.pipelines) {
      const s = algoPipes[pipeline] && algoPipes[pipeline].summary;
      const inf = algoPipes[pipeline] && algoPipes[pipeline].inference;
      const q = algoPipes[pipeline] && algoPipes[pipeline].quality;
      if (!s || !inf) continue;
      const qualityKeys = q ? Object.keys(q) : [];
      let lastQuality = null;
      if (qualityKeys.length > 0) {
        qualityKeys.sort((a, b) => Number.parseFloat(a) - Number.parseFloat(b));
        lastQuality = q[qualityKeys[qualityKeys.length - 1]];
      }
      summaries.push({
        algo,
        pipeline,
        gamesPerSec: s.gamesPerSec.median,
        trainStepsPerSec: s.trainStepsPerSec.median,
        generationsPerMin: s.generationsPerMin.median,
        framesPerSec: s.framesPerSec.median,
        rafP95Ms: s.rafP95Ms.median,
        inferBusyP95Ms: inf.busyLatencyMs.p95,
        inferIdleP95Ms: inf.idleLatencyMs.p95,
        inferBusyTimeouts: inf.busyTimeoutCount,
        inferIdleTimeouts: inf.idleTimeoutCount,
        qualityWindowSec: lastQuality ? lastQuality.durationSec : NaN,
        qualityEloDelta: lastQuality ? lastQuality.deltaElo : NaN,
        qualityPerGpuSecond: lastQuality ? lastQuality.qualityPerGpuSecond : NaN
      });
    }
  }

  const comparisons = {};
  const summaryLines = [
    'System interface benchmark complete',
    'config duration=' + cfg.durationSec + 's runs=' + cfg.runs
      + ' ticks=' + cfg.ticksPerFrame
      + ' model=' + cfg.model
      + ' algo=' + cfg.algo
      + ' algos=' + (cfg.algos.length > 0 ? cfg.algos.join(',') : '(single)')
      + ' game=' + cfg.game
      + ' qualityWindows=' + (cfg.qualityWindowsSec.length > 0 ? cfg.qualityWindowsSec.join(',') : '(none)')
  ];

  for (const algo of algoList) {
    const algoRows = summaries.filter((row) => row.algo === algo);
    if (!algoRows.length) continue;

    const baselineRow = algoRows.find((row) => row.pipeline === 'single_gpu_phased') || algoRows[0];
    const bestRow = algoRows
      .slice()
      .sort((a, b) => (Number.isFinite(b.gamesPerSec) ? b.gamesPerSec : -Infinity) - (Number.isFinite(a.gamesPerSec) ? a.gamesPerSec : -Infinity))[0];

    comparisons[algo] = {
      baselinePipeline: baselineRow.pipeline,
      bestPipelineByGamesPerSec: bestRow ? bestRow.pipeline : null,
      rows: algoRows.map((row) => ({
        pipeline: row.pipeline,
        gamesPerSecDeltaVsBaselinePercent: computePercentDelta(row.gamesPerSec, baselineRow.gamesPerSec),
        inferBusyP95DeltaVsBaselinePercent: computePercentDelta(row.inferBusyP95Ms, baselineRow.inferBusyP95Ms),
        inferIdleP95DeltaVsBaselinePercent: computePercentDelta(row.inferIdleP95Ms, baselineRow.inferIdleP95Ms)
      }))
    };

    summaryLines.push(
      'algo=' + algo
      + ' baseline=' + baselineRow.pipeline
      + ' best_games/s=' + (bestRow ? bestRow.pipeline : 'n/a')
      + ' best_delta=' + formatSignedPercent(bestRow ? computePercentDelta(bestRow.gamesPerSec, baselineRow.gamesPerSec) : null)
    );

    for (const row of algoRows) {
      const rowCmp = comparisons[algo].rows.find((entry) => entry.pipeline === row.pipeline) || {};
      let line = row.pipeline
        + ' games/s=' + formatNumber(row.gamesPerSec, 2)
        + ' trainSteps/s=' + formatNumber(row.trainStepsPerSec, 2)
        + ' gen/min=' + formatNumber(row.generationsPerMin, 2)
        + ' busy_p95_ms=' + formatNumber(row.inferBusyP95Ms, 3)
        + ' idle_p95_ms=' + formatNumber(row.inferIdleP95Ms, 3)
        + ' timeouts=' + row.inferBusyTimeouts + '/' + row.inferIdleTimeouts;
      if (row.pipeline !== baselineRow.pipeline) {
        line += ' games_delta=' + formatSignedPercent(rowCmp.gamesPerSecDeltaVsBaselinePercent)
          + ' busy_delta=' + formatSignedPercent(rowCmp.inferBusyP95DeltaVsBaselinePercent)
          + ' idle_delta=' + formatSignedPercent(rowCmp.inferIdleP95DeltaVsBaselinePercent);
      }
      if (Number.isFinite(row.qualityEloDelta)) {
        line += ' quality_elo_delta=' + formatNumber(row.qualityEloDelta, 3)
          + ' quality_per_gpu_sec=' + formatNumber(row.qualityPerGpuSecond, 6);
      }
      summaryLines.push(line);
    }
  }

  out.summaries = summaries;
  out.comparisons = comparisons;
  emitBenchmarkReport(output, out, summaryLines);
}

main().catch((err) => {
  const msg = err && err.message ? err.message : String(err);
  console.error('System interface benchmark failed:', msg);
  process.exitCode = 1;
});
