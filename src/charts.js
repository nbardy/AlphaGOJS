// Minimal canvas line-chart renderer.
// Each chart is a self-contained canvas with axes, labels, and data lines.

var CHART_BG = '#0d0d22';
var CHART_BORDER = '#1a1a40';
var CHART_GRID = '#1a1a3a';
var CHART_TEXT = '#6666aa';
var CHART_LINE = '#00ff88';
var CHART_WIDTH = 400;
var CHART_HEIGHT = 150;
var PADDING = { top: 22, right: 12, bottom: 22, left: 48 };

export function createChartCanvas() {
  var canvas = document.createElement('canvas');
  canvas.width = CHART_WIDTH;
  canvas.height = CHART_HEIGHT;
  canvas.style.display = 'block';
  canvas.style.width = '100%';
  canvas.style.height = 'auto';
  canvas.style.borderRadius = '6px';
  canvas.style.border = '1px solid ' + CHART_BORDER;
  canvas.style.background = CHART_BG;
  return canvas;
}

export function drawLineChart(canvas, data, options) {
  var ctx = canvas.getContext('2d');
  var w = canvas.width;
  var h = canvas.height;
  var title = options.title || '';
  var color = options.color || CHART_LINE;
  var minY = options.minY;
  var maxY = options.maxY;
  var refLine = options.refLine; // optional horizontal reference line value
  var refColor = options.refColor || '#ffcc00';

  // Clear
  ctx.fillStyle = CHART_BG;
  ctx.fillRect(0, 0, w, h);

  var plotX = PADDING.left;
  var plotY = PADDING.top;
  var plotW = w - PADDING.left - PADDING.right;
  var plotH = h - PADDING.top - PADDING.bottom;

  var series = options.series || [{ data: data, color: color }];
  var hasSeriesData = false;
  for (var hi = 0; hi < series.length; hi++) {
    if (series[hi].data && series[hi].data.length > 0) {
      hasSeriesData = true;
      break;
    }
  }

  if ((!data || data.length === 0) && !hasSeriesData) {
    ctx.fillStyle = CHART_TEXT;
    ctx.font = '11px "Courier New", monospace';
    ctx.textAlign = 'center';
    var waitingMsg = options.waitingMessage || 'Waiting for data...';
    var lines = waitingMsg.split('\n');
    var lineH = 14;
    var startY = h / 2 - ((lines.length - 1) * lineH) / 2;
    for (var wi = 0; wi < lines.length; wi++) {
      ctx.fillText(lines[wi], w / 2, startY + wi * lineH);
    }
    _drawTitle(ctx, title, w);
    return;
  }

  // Auto-scale Y if not provided
  if (minY === undefined || maxY === undefined) {
    var dataMin = null;
    var dataMax = null;
    if (data && data.length) {
      dataMin = data[0];
      dataMax = data[0];
      for (var i = 1; i < data.length; i++) {
        if (data[i] < dataMin) dataMin = data[i];
        if (data[i] > dataMax) dataMax = data[i];
      }
    }
    if (dataMin === null && hasSeriesData) {
      for (var si0 = 0; si0 < series.length; si0++) {
        var sd0 = series[si0].data;
        if (!sd0 || !sd0.length) continue;
        for (var di = 0; di < sd0.length; di++) {
          if (dataMin === null || sd0[di] < dataMin) dataMin = sd0[di];
          if (dataMax === null || sd0[di] > dataMax) dataMax = sd0[di];
        }
      }
    }
    if (minY === undefined) minY = dataMin;
    if (maxY === undefined) maxY = dataMax;
  }
  // Ensure range isn't zero
  if (maxY - minY < 1e-6) {
    minY -= 0.5;
    maxY += 0.5;
  }

  // Draw grid lines (4 horizontal)
  ctx.strokeStyle = CHART_GRID;
  ctx.lineWidth = 0.5;
  ctx.fillStyle = CHART_TEXT;
  ctx.font = '9px "Courier New", monospace';
  ctx.textAlign = 'right';
  for (var gi = 0; gi <= 4; gi++) {
    var gy = plotY + plotH - (gi / 4) * plotH;
    var gv = minY + (gi / 4) * (maxY - minY);
    ctx.beginPath();
    ctx.moveTo(plotX, gy);
    ctx.lineTo(plotX + plotW, gy);
    ctx.stroke();
    ctx.fillText(_fmtNum(gv), plotX - 4, gy + 3);
  }

  // Reference line (e.g., 50% for win rate)
  if (refLine !== undefined && refLine >= minY && refLine <= maxY) {
    var ry = plotY + plotH - ((refLine - minY) / (maxY - minY)) * plotH;
    ctx.strokeStyle = refColor;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX, ry);
    ctx.lineTo(plotX + plotW, ry);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw data line(s) — supports single array or multi-series via options.series
  for (var si = 0; si < series.length; si++) {
    var sd = series[si].data;
    var sc = series[si].color || color;
    if (!sd || sd.length === 0) continue;
    ctx.strokeStyle = sc;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    var lastDx = 0;
    var lastDy = 0;
    for (var i = 0; i < sd.length; i++) {
      var dx = plotX + (i / Math.max(sd.length - 1, 1)) * plotW;
      var dy = plotY + plotH - ((sd[i] - minY) / (maxY - minY)) * plotH;
      lastDx = dx;
      lastDy = dy;
      if (i === 0) ctx.moveTo(dx, dy);
      else ctx.lineTo(dx, dy);
    }
    if (sd.length === 1) {
      ctx.lineTo(lastDx + 1, lastDy);
    }
    ctx.stroke();
    if (sd.length === 1) {
      ctx.fillStyle = sc;
      ctx.beginPath();
      ctx.arc(lastDx, lastDy, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Legend for multi-series
  if (options.series && options.series.length > 1) {
    ctx.font = '9px "Courier New", monospace';
    ctx.textAlign = 'left';
    var lx = plotX + 4;
    for (var si = 0; si < series.length; si++) {
      var ly = plotY + 4 + si * 12;
      ctx.fillStyle = series[si].color;
      ctx.fillRect(lx, ly, 8, 8);
      ctx.fillStyle = CHART_TEXT;
      ctx.fillText(series[si].label || '', lx + 12, ly + 7);
    }
  }

  // Title
  _drawTitle(ctx, title, w);
}

function _drawTitle(ctx, title, w) {
  if (!title) return;
  ctx.fillStyle = '#8888cc';
  ctx.font = '11px "Courier New", monospace';
  ctx.textAlign = 'center';
  ctx.fillText(title, w / 2, 14);
}

function _fmtNum(v) {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(3);
}
