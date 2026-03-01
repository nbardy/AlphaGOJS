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
  canvas.style.borderRadius = '6px';
  canvas.style.border = '1px solid ' + CHART_BORDER;
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

  if (!data || data.length === 0) {
    ctx.fillStyle = CHART_TEXT;
    ctx.font = '11px "Courier New", monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Waiting for data...', w / 2, h / 2);
    _drawTitle(ctx, title, w);
    return;
  }

  // Auto-scale Y if not provided
  if (minY === undefined || maxY === undefined) {
    var dataMin = data[0], dataMax = data[0];
    for (var i = 1; i < data.length; i++) {
      if (data[i] < dataMin) dataMin = data[i];
      if (data[i] > dataMax) dataMax = data[i];
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

  // Draw data line
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (var i = 0; i < data.length; i++) {
    var dx = plotX + (i / Math.max(data.length - 1, 1)) * plotW;
    var dy = plotY + plotH - ((data[i] - minY) / (maxY - minY)) * plotH;
    if (i === 0) ctx.moveTo(dx, dy);
    else ctx.lineTo(dx, dy);
  }
  ctx.stroke();

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
