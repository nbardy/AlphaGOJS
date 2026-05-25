console.log("--- SYSTEM DIAGNOSTICS ---");
console.log("WebGPU available (navigator.gpu):", !!navigator.gpu);
console.log("Float16Array available:", typeof Float16Array !== "undefined");
console.log("--------------------------");
const canvas = document.getElementById('board') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;

const statLoss = document.getElementById('stat-loss')!;
const statElo = document.getElementById('stat-elo')!;
const statTime = document.getElementById('stat-time')!;
const statRollout = document.getElementById('stat-rollout')!;
const statGames = document.getElementById('stat-games')!;
const statGamesPerSec = document.getElementById('stat-games-per-sec')!;
const statAvgSteps = document.getElementById('stat-avg-steps')!;
const errorOverlay = document.getElementById('error-overlay')!;

const chartCanvas = document.getElementById('elo-chart') as HTMLCanvasElement;
const chartCtx = chartCanvas.getContext('2d')!;
const eloHistory: number[] = [];

const K = 24;
const CELL_SIZE = canvas.width / K;

function drawChart() {
  chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
  if (eloHistory.length === 0) return;

  const minElo = Math.min(...eloHistory, 1000) - 50;
  const maxElo = Math.max(...eloHistory, 1000) + 50;
  const range = maxElo - minElo;
  
  chartCtx.beginPath();
  chartCtx.strokeStyle = '#4488ff';
  chartCtx.lineWidth = 2;

  for (let i = 0; i < eloHistory.length; i++) {
    const x = (i / Math.max(1, eloHistory.length - 1)) * chartCanvas.width;
    const y = chartCanvas.height - ((eloHistory[i] - minElo) / range) * chartCanvas.height;
    if (i === 0) chartCtx.moveTo(x, y);
    else chartCtx.lineTo(x, y);
  }
  chartCtx.stroke();
}

function drawBoard(packed: Uint32Array) {
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

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });

worker.onmessage = (e) => {
  if (e.data.type === 'LOG') {
    console.log("[WORKER LOG]", e.data.msg);
  } else if (e.data.type === 'ERROR') {
    errorOverlay.innerText = "Error: " + e.data.message;
  } else if (e.data.type === 'BOARD') {
    if (e.data.board[0]) console.log("Main received board:", e.data.board[0], e.data.board[1]);
    drawBoard(e.data.board);
  } else if (e.data.type === 'STATS') {
    statLoss.innerText = e.data.stats.loss.toFixed(4);
    statElo.innerText = e.data.stats.elo.toFixed(0);
    statTime.innerText = e.data.stats.timeMs.toFixed(0) + 'ms';
    statRollout.innerText = e.data.stats.rollout.toString();
    statGames.innerText = e.data.stats.games.toString();
    statGamesPerSec.innerText = e.data.stats.trainedGamesPerSec.toFixed(1);
    statAvgSteps.innerText = e.data.stats.avgStepsPerGame.toFixed(1);
    
    eloHistory.push(e.data.stats.elo);
    drawChart();
  }
};

worker.onerror = (err) => {
  console.error("Worker failed to load:", err.message, err.filename, err.lineno);
  errorOverlay.innerText = "Worker Error: " + err.message;
};

worker.postMessage({ type: 'START' });
