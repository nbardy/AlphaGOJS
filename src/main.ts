import { createChartCanvas, drawLineChart } from './charts';
import type { ChartOptions, ChartSeries } from './charts';
import { LEAGUE_ARCHS } from './arch_config';

// --- System diagnostics ---
console.log("--- SYSTEM DIAGNOSTICS ---");
console.log("WebGPU available (navigator.gpu):", !!navigator.gpu);
console.log("Float16Array available:", typeof Float16Array !== "undefined");
console.log("--------------------------");

// --- DOM refs ---
const canvas = document.getElementById('board') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const statLoss       = document.getElementById('stat-loss')!;
const statElo        = document.getElementById('stat-elo')!;
const statTime       = document.getElementById('stat-time')!;
const statRollout    = document.getElementById('stat-rollout')!;
const statGames      = document.getElementById('stat-games')!;
const statGamesPerSec = document.getElementById('stat-games-per-sec')!;
const statAvgSteps   = document.getElementById('stat-avg-steps')!;
const errorOverlay   = document.getElementById('error-overlay')!;
const btnPause       = document.getElementById('btn-pause') as HTMLButtonElement;
const statusIndicator = document.getElementById('status-indicator')!;
const chartGrid      = document.getElementById('chart-grid')!;

// --- League mode DOM refs ---
const btnSingle      = document.getElementById('btn-single') as HTMLButtonElement;
const btnLeague      = document.getElementById('btn-league') as HTMLButtonElement;
const leaguePanel    = document.getElementById('league-panel')!;
const leagueTbody    = document.getElementById('league-tbody')!;
const leagueTrainingStatus = document.getElementById('league-training-status')!;
const evalLog        = document.getElementById('eval-log')!;
const boardLabel     = document.getElementById('board-label')!;
const statsBar       = document.querySelector('.stats') as HTMLElement;

// --- Board rendering ---
const K = 24;
const CELL_SIZE = canvas.width / K;

function drawBoard(packed: Uint32Array): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < K; y++) {
    for (let x = 0; x < K; x++) {
      const n = y * K + x;
      const word = packed[n >> 4];
      const state = (word >> ((n & 15) << 1)) & 3;

      let color = '#1e1e1e';
      if (state === 1) color = '#4488ff';
      else if (state === 2) color = '#ff4444';
      else if (state === 3) color = '#666666';

      ctx.fillStyle = color;
      ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1);
    }
  }
}

// --- Metric history ---
const history = {
  elo:         [] as number[],
  loss:        [] as number[],
  gamesPerSec: [] as number[],
  avgSteps:    [] as number[],
  winRate:     [] as number[],
  entropy:     [] as number[],
};

// --- Chart setup: one canvas per metric, inserted into the 2x3 grid ---

interface ChartDef {
  key: keyof typeof history;
  options: ChartOptions;
}

const chartDefs: ChartDef[] = [
  {
    key: 'elo',
    options: { title: 'Elo Rating', color: '#4488ff', refLine: 1000, refColor: '#ffcc00' },
  },
  {
    key: 'loss',
    options: { title: 'Training Loss', color: '#ff4444' },
  },
  {
    key: 'gamesPerSec',
    options: { title: 'Games/sec', color: '#00ff88' },
  },
  {
    key: 'avgSteps',
    options: { title: 'Avg Steps/Game', color: '#ff8844' },
  },
  {
    key: 'winRate',
    options: { title: 'Win Rate vs Checkpoints', color: '#44dddd', refLine: 0.5, refColor: '#ffcc00', minY: 0, maxY: 1 },
  },
  {
    key: 'entropy',
    options: { title: 'Policy Entropy', color: '#aa44ff' },
  },
];

// Create chart canvases and mount them
const chartCanvases = chartDefs.map(def => {
  const c = createChartCanvas(300, 150);
  chartGrid.appendChild(c);
  return { canvas: c, def };
});

/** Redraw all 6 charts from current history. */
function redrawCharts(): void {
  for (const { canvas: c, def } of chartCanvases) {
    drawLineChart(c, history[def.key], def.options);
  }
}

// Draw initial empty state
redrawCharts();

// --- Pause state (declared early so league toggle can reference it) ---
let paused = false;

// --- League mode state ---
let mode: 'single' | 'league' = 'single';
let leagueWorker: Worker | null = null;

// Per-architecture metric history for league mode charts
const leagueHistory: Record<string, { elo: number[], loss: number[], gamesPerSec: number[] }> = {};

/** Update the league standings table from LEAGUE_STATS data. */
function updateLeagueTable(archs: Array<{
  config: { name: string, D: number, color: string },
  elo: number,
  generation: number,
  totalGames: number,
  latestLoss: number,
  latestGamesPerSec: number,
}>): void {
  // Sort by Elo descending for ranking
  const sorted = [...archs].sort((a, b) => b.elo - a.elo);
  leagueTbody.innerHTML = sorted.map((arch, i) => `
    <tr>
      <td>${i + 1}</td>
      <td class="arch-name-cell" style="border-left-color:${arch.config.color};color:${arch.config.color}">${arch.config.name}</td>
      <td>${arch.config.D}</td>
      <td style="font-weight:bold">${arch.elo.toFixed(0)}</td>
      <td>${arch.generation}</td>
      <td>${arch.totalGames}</td>
      <td>${arch.latestLoss.toFixed(4)}</td>
      <td>${arch.latestGamesPerSec.toFixed(1)}</td>
    </tr>
  `).join('');
}

/** Redraw the Elo chart in multi-series league mode. */
function redrawLeagueEloChart(): void {
  const eloChart = chartCanvases[0]; // Elo is the first chart
  const eloSeries: ChartSeries[] = LEAGUE_ARCHS.map(a => ({
    data: leagueHistory[a.name]?.elo ?? [],
    color: a.color,
    label: a.name,
  }));
  // Multi-series: pass empty data array, use options.series
  drawLineChart(eloChart.canvas, [], { title: 'League Elo', series: eloSeries, refLine: 1000, refColor: '#ffcc00' });
}

/** Append an eval result entry to the log (newest at top). */
function appendEvalEntry(entry: {
  archA: string, archB: string,
  winsA: number, winsB: number, draws: number,
  rollout: number,
}): void {
  const winner = entry.winsA > entry.winsB ? entry.archA
    : entry.winsB > entry.winsA ? entry.archB
    : 'draw';
  const outcome = winner === 'draw'
    ? `${entry.winsA}-${entry.winsB} (draw)`
    : `${entry.winsA}-${entry.winsB} (${winner} wins)`;
  const div = document.createElement('div');
  div.className = 'eval-entry';
  div.textContent = `Rollout ${entry.rollout}: ${entry.archA} vs ${entry.archB} — ${outcome}`;
  // Insert at top so newest entries appear first
  evalLog.insertBefore(div, evalLog.firstChild);
}

/** Handle messages from the league worker. */
function handleLeagueMessage(e: MessageEvent): void {
  const msg = e.data;

  if (msg.type === 'LOG') {
    console.log("[LEAGUE LOG]", msg.msg);
    return;
  }
  if (msg.type === 'ERROR') {
    errorOverlay.innerText = "League Error: " + msg.message;
    return;
  }
  if (msg.type === 'BOARD') {
    drawBoard(msg.board);
    // Show which architecture's board we're viewing
    const archName = msg.arch ?? 'unknown';
    const archConfig = LEAGUE_ARCHS.find(a => a.name === archName);
    boardLabel.textContent = archConfig
      ? `Board: ${archName} (D=${archConfig.D})`
      : `Board: ${archName}`;
    boardLabel.style.color = archConfig?.color ?? '#888';
    return;
  }
  if (msg.type === 'LEAGUE_STATS') {
    const { archs, evalHistory, currentTraining } = msg.stats;

    // Update standings table
    updateLeagueTable(archs);

    // Push to per-arch history arrays
    for (const arch of archs) {
      const name = arch.config.name;
      if (!leagueHistory[name]) {
        leagueHistory[name] = { elo: [], loss: [], gamesPerSec: [] };
      }
      leagueHistory[name].elo.push(arch.elo);
      leagueHistory[name].loss.push(arch.latestLoss);
      leagueHistory[name].gamesPerSec.push(arch.latestGamesPerSec);
    }

    // Redraw the Elo chart with multi-series lines
    redrawLeagueEloChart();

    // Show training status
    leagueTrainingStatus.textContent = `Currently training: ${currentTraining}`;

    // Append new eval results
    for (const entry of evalHistory) {
      appendEvalEntry(entry);
    }
    return;
  }
}

/** Start the league worker and wire up message handling. */
function startLeagueWorker(): void {
  if (leagueWorker) return;
  leagueWorker = new Worker(new URL('./league_worker.ts', import.meta.url), { type: 'module' });
  leagueWorker.onmessage = handleLeagueMessage;
  leagueWorker.onerror = (err: ErrorEvent) => {
    console.error("League worker failed:", err.message);
    errorOverlay.innerText = "League Worker Error: " + err.message;
  };
  leagueWorker.postMessage({ type: 'START_LEAGUE' });
}

/** Stop the league worker. */
function stopLeagueWorker(): void {
  if (!leagueWorker) return;
  leagueWorker.terminate();
  leagueWorker = null;
}

// --- Single-mode Worker ---
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

// --- Mode toggle ---
function setMode(newMode: 'single' | 'league'): void {
  if (newMode === mode) return;
  mode = newMode;

  // Toggle active button
  btnSingle.classList.toggle('active', mode === 'single');
  btnLeague.classList.toggle('active', mode === 'league');

  // Show/hide league panel and single-mode stats
  leaguePanel.style.display = mode === 'league' ? 'block' : 'none';
  statsBar.style.display = mode === 'single' ? 'flex' : 'none';
  boardLabel.style.display = mode === 'league' ? 'block' : 'none';

  if (mode === 'league') {
    // Pause single worker, start league worker
    if (!paused) {
      worker.postMessage({ type: 'PAUSE' });
    }
    startLeagueWorker();
    // Show multi-series Elo chart immediately (may be empty)
    redrawLeagueEloChart();
  } else {
    // Stop league worker, resume single worker
    stopLeagueWorker();
    if (!paused) {
      worker.postMessage({ type: 'RESUME' });
    }
    // Clear league board label
    boardLabel.textContent = '';
    // Redraw single-mode charts
    redrawCharts();
  }
}

// --- Mode toggle event listeners ---
btnSingle.addEventListener('click', () => setMode('single'));
btnLeague.addEventListener('click', () => setMode('league'));

// --- Pause / Resume ---

btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.textContent = paused ? 'Resume' : 'Pause';
  statusIndicator.textContent = paused ? 'Paused' : 'Training';
  statusIndicator.style.color = paused ? '#ffcc00' : '#00ff88';
  // Notify the active worker based on current mode
  const activeWorker = mode === 'league' ? leagueWorker : worker;
  activeWorker?.postMessage({ type: paused ? 'PAUSE' : 'RESUME' });
});

// --- Single-mode Worker message handler ---

worker.onmessage = (e: MessageEvent) => {
  const msg = e.data;

  if (msg.type === 'LOG') {
    console.log("[WORKER LOG]", msg.msg);
    return;
  }
  if (msg.type === 'ERROR') {
    errorOverlay.innerText = "Error: " + msg.message;
    return;
  }
  if (msg.type === 'BOARD') {
    if (msg.board[0]) console.log("Main received board:", msg.board[0], msg.board[1]);
    drawBoard(msg.board);
    return;
  }
  if (msg.type === 'STATS') {
    const s = msg.stats;

    // Update stat bar
    statLoss.innerText       = s.loss.toFixed(4);
    statElo.innerText        = s.elo.toFixed(0);
    statTime.innerText       = s.timeMs.toFixed(0) + 'ms';
    statRollout.innerText    = s.rollout.toString();
    statGames.innerText      = s.games.toString();
    statGamesPerSec.innerText = s.trainedGamesPerSec.toFixed(1);
    statAvgSteps.innerText   = s.avgStepsPerGame.toFixed(1);

    // Push to history arrays
    history.elo.push(s.elo);
    history.loss.push(s.loss);
    history.gamesPerSec.push(s.trainedGamesPerSec);
    history.avgSteps.push(s.avgStepsPerGame);
    // winRate and entropy may not be present yet -- the worker-side protocol
    // addition is handled by a separate agent. Default to 0 until available.
    history.winRate.push(s.winRate ?? 0);
    history.entropy.push(s.entropy ?? 0);

    redrawCharts();
    return;
  }
};

worker.onerror = (err: ErrorEvent) => {
  console.error("Worker failed to load:", err.message, err.filename, err.lineno);
  errorOverlay.innerText = "Worker Error: " + err.message;
};

worker.postMessage({ type: 'START' });
