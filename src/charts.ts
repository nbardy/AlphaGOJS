// Self-contained canvas line-chart renderer for the training dashboard.
// Ported from v1 charts.js with TypeScript types, downsampling, and multi-series support.
// No external dependencies -- renders directly to a 2D canvas context.

// --- Theme constants ---
const CHART_BG    = '#0d0d22';
const CHART_BORDER = '#1a1a40';
const CHART_GRID  = '#1a1a3a';
const CHART_TEXT  = '#6666aa';
const CHART_TITLE = '#8888cc';
const DEFAULT_LINE_COLOR = '#00ff88';

const DEFAULT_WIDTH  = 300;
const DEFAULT_HEIGHT = 150;
const DEFAULT_MAX_POINTS = 500;

const PADDING = { top: 22, right: 12, bottom: 22, left: 48 } as const;

// --- Public types ---

export interface ChartSeries {
  data: number[];
  color: string;
  label?: string;
}

export interface ChartOptions {
  title?: string;
  color?: string;
  minY?: number;
  maxY?: number;
  refLine?: number;
  refColor?: string;
  series?: ChartSeries[];
  waitingMessage?: string;
  /** Downsample to this many points when data exceeds it. Default 500. */
  maxPoints?: number;
}

// --- Helpers ---

/** Format a Y-axis label: big numbers get 0 decimals, small get 3. */
function fmtNum(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1)   return v.toFixed(1);
  return v.toFixed(3);
}

/**
 * Evenly subsample an array down to `target` points.
 * Preserves first and last elements for visual continuity.
 */
function downsample(arr: number[], target: number): number[] {
  if (arr.length <= target) return arr;
  const out: number[] = [arr[0]];
  const step = (arr.length - 1) / (target - 1);
  for (let i = 1; i < target - 1; i++) {
    out.push(arr[Math.round(i * step)]);
  }
  out.push(arr[arr.length - 1]);
  return out;
}

/** Compute min/max across one or more number arrays. Returns [min, max]. */
function dataRange(arrays: number[][]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const arr of arrays) {
    for (const v of arr) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return [lo, hi];
}

// --- Public API ---

/** Create a styled chart canvas element ready for insertion into the DOM. */
export function createChartCanvas(
  width: number = DEFAULT_WIDTH,
  height: number = DEFAULT_HEIGHT,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.style.display = 'block';
  canvas.style.width = '100%';
  canvas.style.height = 'auto';
  canvas.style.borderRadius = '6px';
  canvas.style.border = `1px solid ${CHART_BORDER}`;
  canvas.style.background = CHART_BG;
  return canvas;
}

/**
 * Render a line chart onto a canvas.
 *
 * Supports two usage patterns:
 *   1. Single series: pass `data` directly, optionally with `color`.
 *   2. Multi-series: pass `options.series` array. `data` is ignored when series is present.
 *
 * Downsamples automatically when data exceeds `options.maxPoints` (default 500).
 */
export function drawLineChart(
  canvas: HTMLCanvasElement,
  data: number[],
  options: ChartOptions,
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  const title = options.title ?? '';
  const color = options.color ?? DEFAULT_LINE_COLOR;
  const maxPts = options.maxPoints ?? DEFAULT_MAX_POINTS;

  // Clear background
  ctx.fillStyle = CHART_BG;
  ctx.fillRect(0, 0, w, h);

  // Plot area geometry
  const plotX = PADDING.left;
  const plotY = PADDING.top;
  const plotW = w - PADDING.left - PADDING.right;
  const plotH = h - PADDING.top - PADDING.bottom;

  // Resolve series list -- single data array becomes a one-element series
  const rawSeries: ChartSeries[] = options.series
    ?? (data.length > 0 ? [{ data, color }] : []);

  const hasData = rawSeries.some(s => s.data.length > 0);

  // --- Empty state: "waiting for data" placeholder ---
  if (!hasData) {
    drawTitle(ctx, title, w);
    ctx.fillStyle = CHART_TEXT;
    ctx.font = '11px "Courier New", monospace';
    ctx.textAlign = 'center';
    const msg = options.waitingMessage ?? 'Waiting for data...';
    const lines = msg.split('\n');
    const lineH = 14;
    const startY = h / 2 - ((lines.length - 1) * lineH) / 2;
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], w / 2, startY + i * lineH);
    }
    return;
  }

  // --- Downsample each series ---
  const series: ChartSeries[] = rawSeries.map(s => ({
    ...s,
    data: downsample(s.data, maxPts),
  }));

  // --- Compute Y range ---
  const nonEmpty = series.filter(s => s.data.length > 0).map(s => s.data);
  const [autoMin, autoMax] = dataRange(nonEmpty);
  let minY = options.minY ?? autoMin;
  let maxY = options.maxY ?? autoMax;
  // Ensure non-degenerate range
  if (maxY - minY < 1e-6) {
    minY -= 0.5;
    maxY += 0.5;
  }

  // --- Grid lines (4 horizontal) ---
  ctx.strokeStyle = CHART_GRID;
  ctx.lineWidth = 0.5;
  ctx.fillStyle = CHART_TEXT;
  ctx.font = '9px "Courier New", monospace';
  ctx.textAlign = 'right';
  for (let gi = 0; gi <= 4; gi++) {
    const gy = plotY + plotH - (gi / 4) * plotH;
    const gv = minY + (gi / 4) * (maxY - minY);
    ctx.beginPath();
    ctx.moveTo(plotX, gy);
    ctx.lineTo(plotX + plotW, gy);
    ctx.stroke();
    ctx.fillText(fmtNum(gv), plotX - 4, gy + 3);
  }

  // --- Reference line (dashed) ---
  if (options.refLine !== undefined && options.refLine >= minY && options.refLine <= maxY) {
    const refColor = options.refColor ?? '#ffcc00';
    const ry = plotY + plotH - ((options.refLine - minY) / (maxY - minY)) * plotH;
    ctx.strokeStyle = refColor;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX, ry);
    ctx.lineTo(plotX + plotW, ry);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // --- Data lines ---
  for (const s of series) {
    if (s.data.length === 0) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let lastX = 0;
    let lastY = 0;
    const len = s.data.length;
    for (let i = 0; i < len; i++) {
      const dx = plotX + (i / Math.max(len - 1, 1)) * plotW;
      const dy = plotY + plotH - ((s.data[i] - minY) / (maxY - minY)) * plotH;
      lastX = dx;
      lastY = dy;
      if (i === 0) ctx.moveTo(dx, dy);
      else ctx.lineTo(dx, dy);
    }
    // For a single data point, draw a visible dot
    if (len === 1) {
      ctx.lineTo(lastX + 1, lastY);
      ctx.stroke();
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.stroke();
    }
  }

  // --- Legend for multi-series ---
  if (series.length > 1) {
    ctx.font = '9px "Courier New", monospace';
    ctx.textAlign = 'left';
    const lx = plotX + 4;
    for (let si = 0; si < series.length; si++) {
      const ly = plotY + 4 + si * 12;
      ctx.fillStyle = series[si].color;
      ctx.fillRect(lx, ly, 8, 8);
      ctx.fillStyle = CHART_TEXT;
      ctx.fillText(series[si].label ?? '', lx + 12, ly + 7);
    }
  }

  // --- Title ---
  drawTitle(ctx, title, w);
}

function drawTitle(ctx: CanvasRenderingContext2D, title: string, canvasWidth: number): void {
  if (!title) return;
  ctx.fillStyle = CHART_TITLE;
  ctx.font = '11px "Courier New", monospace';
  ctx.textAlign = 'center';
  ctx.fillText(title, canvasWidth / 2, 14);
}
