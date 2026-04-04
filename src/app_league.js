import { listModelTypes } from './model_registry';
import { listAlgorithmTypes } from './algo_registry';
import { listRuntimeTypes } from './runtime/runtime_registry';
import { createLeaguePipeline } from './league_pipeline';
import { probeCapabilities } from './nextgen/capability_probe';
import { chooseRuntimeTier } from './nextgen/runtime_planner';
import { UI } from './ui';

var DEFAULT_ALGO = 'ppo';
var ROWS = 20;
var COLS = 20;
var NUM_GAMES = 80;

function pickNum(v, fallback) {
  return typeof v === 'number' ? v : fallback;
}

function parseClampedPositiveIntParam(params, key, defaultVal, min, max) {
  var raw = params.get(key);
  if (raw === null || raw === '') {
    return defaultVal;
  }
  var n = parseInt(String(raw).trim(), 10);
  if (!Number.isFinite(n) || n < 1) {
    return defaultVal;
  }
  return Math.max(min, Math.min(max, n));
}

function parseLeagueQuery() {
  if (typeof location === 'undefined') {
    return { pipelineType: null, benchRuntimeExtras: {}, rows: ROWS, cols: COLS, numGames: NUM_GAMES };
  }
  var p = new URLSearchParams(location.search);
  var bench = {};
  if (p.get('benchInstrument') === '1') bench.benchInstrument = true;
  if (p.get('benchMinimalUi') === '1') bench.benchMinimalUi = true;
  if (p.get('webgpuEnv') === '1') bench.useWebGPUGameEngine = true;
  var preset = p.get('preset');
  var presetFast = preset === 'fast' || preset === 'interactive';
  return {
    pipelineType: p.get('pipeline') || null,
    benchRuntimeExtras: bench,
    rows: parseClampedPositiveIntParam(p, 'rows', presetFast && !p.has('rows') ? 10 : ROWS, 4, 32),
    cols: parseClampedPositiveIntParam(p, 'cols', presetFast && !p.has('cols') ? 10 : COLS, 4, 32),
    numGames: parseClampedPositiveIntParam(p, 'numGames', presetFast && !p.has('numGames') ? 40 : NUM_GAMES, 4, 128)
  };
}

function mapTierToPipelineType(tier) {
  if (tier === 'A' || tier === 'C') return 'single_gpu_phased';
  if (tier === 'B') return 'cpu_actors_gpu_learner';
  return 'cpu_actors_gpu_learner';
}

async function startLeague() {
  var pipelineType = 'single_gpu_phased';
  try {
    var cap = await probeCapabilities();
    var plan = chooseRuntimeTier(cap, {});
    pipelineType = mapTierToPipelineType(plan.tier);
  } catch (e) {
    pipelineType = 'single_gpu_phased';
  }

  var q = parseLeagueQuery();
  if (q.pipelineType) {
    pipelineType = q.pipelineType;
  }
  var rows = q.rows;
  var cols = q.cols;
  var numGames = q.numGames;

  var pipeline;
  try {
    pipeline = createLeaguePipeline(DEFAULT_ALGO, rows, cols, numGames, pipelineType, undefined, q.benchRuntimeExtras);
  } catch (e) {
    console.error('League pipeline failed (GPU worker + 2+ models required):', e.message);
    document.body.innerHTML =
      '<div style="padding:24px;font-family:monospace;background:#0a0a1a;color:#c0c0e0;max-width:640px;margin:40px auto;">'
      + '<h1 style="color:#ff8866;">League mode unavailable</h1>'
      + '<p>' + (e.message || String(e)) + '</p>'
      + '<p>Open <a href="index.html" style="color:#66ccff;">index.html</a> for standard single-model training.</p>'
      + '</div>';
    return;
  }

  var ui = new UI(pipeline.trainer, pipeline.algo, {
    rows: rows,
    cols: cols,
    numGames: numGames,
    pipelineType: pipeline.pipelineType,
    createPipeline: function (_modelType, algoType, r, c, n, pipelineT, gameType, benchExtras) {
      return createLeaguePipeline(
        algoType,
        r,
        c,
        n,
        pipelineT,
        gameType,
        benchExtras || {}
      );
    },
    listModelTypes: listModelTypes,
    listAlgorithmTypes: listAlgorithmTypes,
    listRuntimeTypes: listRuntimeTypes,
    benchRuntimeExtras: q.benchRuntimeExtras,
    leagueMode: true,
    homeHref: 'index.html'
  });
  window.__alphaPlagueLeague = ui;
}

if (module.hot) {
  if (window.__alphaPlagueLeague) {
    window.__alphaPlagueLeague.destroy();
  }
  module.hot.accept();
}

startLeague().catch(function (e) {
  console.error('League bootstrap failed:', e);
});
