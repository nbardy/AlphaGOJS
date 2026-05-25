import { createChartCanvas, drawLineChart } from './charts';
import type { ChartOptions } from './charts';

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

// --- Pause / Resume ---
let paused = false;

btnPause.addEventListener('click', () => {
  paused = !paused;
  btnPause.textContent = paused ? 'Resume' : 'Pause';
  statusIndicator.textContent = paused ? 'Paused' : 'Training';
  statusIndicator.style.color = paused ? '#ffcc00' : '#00ff88';
  // Notify worker. The worker-side PAUSE/RESUME handler is wired up by
  // another agent -- for now we send the message and the worker ignores it
  // until that protocol lands.
  worker.postMessage({ type: paused ? 'PAUSE' : 'RESUME' });
});

// --- Worker ---

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

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
