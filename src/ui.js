import { createGame } from './games/registry';

export class UI {
  constructor(trainer, model) {
    this.trainer = trainer;
    this.model = model;
    this.rows = trainer.rows;
    this.cols = trainer.cols;
    this.boardSize = this.rows * this.cols;
    this.gameId = trainer.gameId;
    this.renderer = trainer.renderer;

    this.gridCellSize = 3;
    this.gridCols = 10;
    this.humanCellSize = 15;

    this.gridCanvases = [];
    this.humanCanvas = null;
    this.humanGame = null;
    this.humanPlaying = false;
    this.humanTurn = true;
    this.humanGameOver = false;

    this.ticksPerFrame = 1;
    this.paused = false;
    this._destroyed = false;
    this.frameCount = 0;

    this._buildDOM();
    this._startLoop();
  }

  destroy() { this._destroyed = true; }

  _injectStyles() {
    var style = document.createElement('style');
    style.textContent = [
      '* { margin:0; padding:0; box-sizing:border-box; }',
      'body { background:#0a0a1a; color:#c0c0e0; font-family:"Courier New",monospace; }',
      '#app { max-width:1000px; margin:0 auto; padding:16px; }',
      'header { text-align:center; margin-bottom:16px; }',
      'header h1 { font-size:28px; color:#00ff88; text-shadow:0 0 20px rgba(0,255,136,0.3); margin-bottom:2px; }',
      '.subtitle { font-size:11px; color:#556; }',
      '#stats { display:flex; justify-content:center; gap:8px; flex-wrap:wrap; margin-top:10px; }',
      '.stat { background:#151530; padding:4px 10px; border-radius:5px; border:1px solid #2a2a5a; text-align:center; }',
      '.stat .label { color:#6666aa; font-size:8px; text-transform:uppercase; letter-spacing:1px; }',
      '.stat .value { font-size:14px; font-weight:bold; color:#ffcc00; }',
      '#controls { display:flex; justify-content:center; align-items:center; gap:12px; margin-bottom:14px; flex-wrap:wrap; }',
      'button { background:#2244aa; color:#fff; border:none; padding:6px 14px; border-radius:5px; cursor:pointer; font-family:inherit; font-size:12px; }',
      'button:hover { background:#3355cc; }',
      'button.play-btn { background:#008844; }',
      'button.play-btn:hover { background:#00aa55; }',
      '.speed-ctrl { display:flex; align-items:center; gap:5px; font-size:11px; }',
      '.speed-ctrl input { width:100px; accent-color:#4488ff; }',
      '#training-section h2, #human-section h2 { font-size:14px; color:#4488ff; margin-bottom:8px; text-align:center; }',
      '#game-grid { display:flex; flex-wrap:wrap; justify-content:center; gap:2px; background:#0d0d22; padding:8px; border-radius:6px; border:1px solid #1a1a40; }',
      '.gcell { border:1px solid #1a1a40; border-radius:2px; overflow:hidden; }',
      '.gcell canvas { display:block; image-rendering:pixelated; image-rendering:crisp-edges; }',
      '#trail-section { margin-top:14px; text-align:center; }',
      '#trail-section h3 { font-size:12px; color:#4488ff; margin-bottom:6px; }',
      '#trail-canvas { background:#0d0d22; border-radius:6px; border:1px solid #1a1a40; }',
      '.trail-info { font-size:11px; margin-top:4px; color:#888; }',
      '#human-section { text-align:center; }',
      '#human-wrap { display:inline-block; border:2px solid #2a2a5a; border-radius:6px; overflow:hidden; margin:8px 0; cursor:pointer; }',
      '#human-info { font-size:13px; margin:8px 0; min-height:20px; }',
      '#human-info.yt { color:#00ff88; }',
      '#human-info.at { color:#ff3366; }',
      '#human-info.go { color:#ffcc00; font-weight:bold; }',
      '#human-score { margin:6px 0; font-size:13px; }',
      '.sg { color:#00ff88; }',
      '.sr { color:#ff3366; }',
      '.controls-row { margin-top:10px; }',
      '.controls-row button { margin:0 4px; }'
    ].join('\n');
    document.head.appendChild(style);
  }

  _buildDOM() {
    this._injectStyles();
    document.body.innerHTML = '';
    var app = document.createElement('div');
    app.id = 'app';
    document.body.appendChild(app);

    var header = document.createElement('header');
    header.innerHTML = '<h1>AlphaPlague</h1>'
      + '<div class="subtitle">' + this.renderer.label + ' \u2014 A2C self-play RL \u2014 ' + this.rows + '\u00d7' + this.cols + ', ' + this.trainer.numGames + ' games</div>'
      + '<div id="stats">'
      + '<div class="stat"><div class="label">Games</div><div class="value" id="sg">0</div></div>'
      + '<div class="stat"><div class="label">Gen</div><div class="value" id="sgen">0</div></div>'
      + '<div class="stat"><div class="label">Loss</div><div class="value" id="sloss">\u2014</div></div>'
      + '<div class="stat"><div class="label">Avg Turns</div><div class="value" id="slen">\u2014</div></div>'
      + '<div class="stat"><div class="label">vs Random</div><div class="value sg" id="svr">\u2014</div></div>'
      + '<div class="stat"><div class="label">Snapshots</div><div class="value" id="ssnap">0</div></div>'
      + '</div>';
    app.appendChild(header);

    var controls = document.createElement('div');
    controls.id = 'controls';
    var self = this;

    var pauseBtn = document.createElement('button');
    pauseBtn.id = 'pbtn';
    pauseBtn.textContent = 'Pause';
    pauseBtn.onclick = function () { self.paused = !self.paused; pauseBtn.textContent = self.paused ? 'Resume' : 'Pause'; };
    controls.appendChild(pauseBtn);

    var speedDiv = document.createElement('div');
    speedDiv.className = 'speed-ctrl';
    speedDiv.innerHTML = '<span>Speed:</span><input type="range" id="sspeed" min="1" max="100" value="1"><span id="sval">1x</span>';
    controls.appendChild(speedDiv);

    var playBtn = document.createElement('button');
    playBtn.className = 'play-btn';
    playBtn.textContent = '\u25B6 Play vs AI';
    playBtn.onclick = function () { self._startHumanGame(); };
    controls.appendChild(playBtn);
    app.appendChild(controls);

    setTimeout(function () {
      var slider = document.getElementById('sspeed');
      if (slider) slider.oninput = function () {
        self.ticksPerFrame = parseInt(slider.value);
        document.getElementById('sval').textContent = slider.value + 'x';
      };
    }, 0);

    // Training grid
    var trainSection = document.createElement('div');
    trainSection.id = 'training-section';
    trainSection.innerHTML = '<h2>Self-Play Training Grid</h2>';

    var grid = document.createElement('div');
    grid.id = 'game-grid';

    var displayW = this.cols * this.gridCellSize;
    var displayH = this.rows * this.gridCellSize;
    this.gridCanvases = [];
    for (var i = 0; i < this.trainer.numGames; i++) {
      var cell = document.createElement('div');
      cell.className = 'gcell';
      var canvas = document.createElement('canvas');
      canvas.width = this.cols;
      canvas.height = this.rows;
      canvas.style.width = displayW + 'px';
      canvas.style.height = displayH + 'px';
      cell.appendChild(canvas);
      grid.appendChild(cell);
      var ctx = canvas.getContext('2d');
      var imgData = ctx.createImageData(this.cols, this.rows);
      this.gridCanvases.push({ canvas: canvas, cell: cell, ctx: ctx, imgData: imgData });
    }
    trainSection.appendChild(grid);
    app.appendChild(trainSection);

    // Trail chart
    var trailSection = document.createElement('div');
    trailSection.id = 'trail-section';
    trailSection.innerHTML = '<h3>Win Rate vs Random (training progress)</h3>';
    var trailCanvas = document.createElement('canvas');
    trailCanvas.id = 'trail-canvas';
    trailCanvas.width = 600;
    trailCanvas.height = 100;
    trailSection.appendChild(trailCanvas);
    var trailInfo = document.createElement('div');
    trailInfo.className = 'trail-info';
    trailInfo.id = 'trail-info';
    trailInfo.textContent = 'Evaluating every ' + this.trainer.evalInterval + ' generations...';
    trailSection.appendChild(trailInfo);
    app.appendChild(trailSection);

    // Human play section
    var humanSection = document.createElement('div');
    humanSection.id = 'human-section';
    humanSection.style.display = 'none';
    humanSection.innerHTML = '<h2>You (Green) vs AI (Red) \u2014 ' + this.rows + '\u00d7' + this.cols + '</h2>';

    var humanInfo = document.createElement('div');
    humanInfo.id = 'human-info';
    humanInfo.className = 'yt';
    humanInfo.textContent = 'Click an empty cell to place your piece';
    humanSection.appendChild(humanInfo);

    var wrap = document.createElement('div');
    wrap.id = 'human-wrap';
    var hCanvas = document.createElement('canvas');
    hCanvas.width = this.cols * this.humanCellSize;
    hCanvas.height = this.rows * this.humanCellSize;
    hCanvas.onclick = function (e) { self._handleClick(e); };
    wrap.appendChild(hCanvas);
    humanSection.appendChild(wrap);
    this.humanCanvas = hCanvas;

    var hScore = document.createElement('div');
    hScore.id = 'human-score';
    humanSection.appendChild(hScore);

    var ctrlRow = document.createElement('div');
    ctrlRow.className = 'controls-row';
    var newBtn = document.createElement('button');
    newBtn.textContent = 'New Game';
    newBtn.onclick = function () { self._startHumanGame(); };
    ctrlRow.appendChild(newBtn);
    var backBtn = document.createElement('button');
    backBtn.textContent = 'Back to Training';
    backBtn.onclick = function () { self._backToTraining(); };
    ctrlRow.appendChild(backBtn);
    humanSection.appendChild(ctrlRow);
    app.appendChild(humanSection);
  }

  _startHumanGame() {
    this.humanPlaying = true;
    this.humanGame = createGame(this.gameId, this.rows, this.cols);
    this.humanTurn = true;
    this.humanGameOver = false;
    document.getElementById('training-section').style.display = 'none';
    document.getElementById('trail-section').style.display = 'none';
    document.getElementById('human-section').style.display = 'block';
    var info = document.getElementById('human-info');
    info.className = 'yt';
    info.textContent = 'Your turn \u2014 click to place (Green)';
    this._renderHuman();
  }

  _backToTraining() {
    this.humanPlaying = false;
    document.getElementById('training-section').style.display = 'block';
    document.getElementById('trail-section').style.display = 'block';
    document.getElementById('human-section').style.display = 'none';
  }

  _handleClick(e) {
    if (!this.humanPlaying || !this.humanTurn || this.humanGameOver) return;
    var rect = this.humanCanvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    var col = Math.floor(x / this.humanCellSize);
    var row = Math.floor(y / this.humanCellSize);
    if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) return;
    var index = row * this.cols + col;
    if (this.humanGame.board[index] !== 0) return;

    this.humanGame.makeMove(1, index);
    this.humanTurn = false;
    this._renderHuman();
    if (this.humanGame.isGameOver() || this.humanGame.getValidMoves().length === 0) { this._endHumanGame(); return; }

    var info = document.getElementById('human-info');
    info.className = 'at';
    info.textContent = 'AI thinking...';
    var self = this;
    setTimeout(function () {
      var state = self.humanGame.getBoardForNN(-1);
      var mask = self.humanGame.getValidMovesMask();
      var action = self.model.getAction(state, mask);
      self.humanGame.makeMove(-1, action);
      self.humanGame.spreadPlague();
      self._renderHuman();
      if (self.humanGame.isGameOver() || self.humanGame.getValidMoves().length === 0) { self._endHumanGame(); }
      else { self.humanTurn = true; info.className = 'yt'; info.textContent = 'Your turn \u2014 click to place (Green)'; }
    }, 100);
  }

  _endHumanGame() {
    this.humanGameOver = true;
    var winner = this.humanGame.getWinner();
    var info = document.getElementById('human-info');
    info.className = 'go';
    var c = this.humanGame.countCells();
    if (winner === 1) info.textContent = '\u2705 You win! (' + c.p1 + '-' + c.p2 + ')';
    else if (winner === -1) info.textContent = '\u274C AI wins! (' + c.p2 + '-' + c.p1 + ')';
    else info.textContent = '\u2696 Draw! (' + c.p1 + '-' + c.p2 + ')';
    this._renderHuman();
  }

  _drawBoardFast(item, board) {
    var data = item.imgData.data;
    var size = this.boardSize;
    var colorFn = this.renderer.cellColor;
    for (var i = 0; i < size; i++) {
      var off = i * 4;
      var rgb = colorFn(board[i]);
      data[off] = rgb[0]; data[off + 1] = rgb[1]; data[off + 2] = rgb[2];
      data[off + 3] = 255;
    }
    item.ctx.putImageData(item.imgData, 0, 0);
  }

  _drawBoardHuman(canvas, board, rows, cols, cs) {
    var ctx = canvas.getContext('2d');
    var colorFn = this.renderer.humanCellColor;
    ctx.fillStyle = '#0d0d22';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        ctx.fillStyle = colorFn(board[r * cols + c]);
        ctx.fillRect(c * cs + 1, r * cs + 1, cs - 2, cs - 2);
      }
    }
  }

  _renderHuman() {
    if (!this.humanGame) return;
    this._drawBoardHuman(this.humanCanvas, this.humanGame.board, this.rows, this.cols, this.humanCellSize);
    var c = this.humanGame.countCells();
    document.getElementById('human-score').innerHTML = '<span class="sg">You: ' + c.p1 + '</span> | <span class="sr">AI: ' + c.p2 + '</span> | Empty: ' + c.empty;
  }

  _renderGrid() {
    for (var i = 0; i < this.trainer.numGames; i++) {
      this._drawBoardFast(this.gridCanvases[i], this.trainer.games[i].game.board);
    }
  }

  _renderTrail() {
    var trail = this.trainer.trail;
    if (trail.length === 0) return;
    var canvas = document.getElementById('trail-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#0d0d22';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = '#1a1a40';
    ctx.lineWidth = 1;
    for (var y = 0; y <= 1; y += 0.25) {
      var py = h - y * h;
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke();
    }

    // 50% baseline
    ctx.strokeStyle = '#333355';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, h * 0.5); ctx.lineTo(w, h * 0.5); ctx.stroke();
    ctx.setLineDash([]);

    // Data
    var n = trail.length;
    var barW = Math.max(4, Math.min(16, (w - 20) / n));
    ctx.fillStyle = '#00ff88';
    for (var i = 0; i < n; i++) {
      var x = 10 + i * barW;
      var barH = trail[i].vsRandom * (h - 10);
      ctx.fillRect(x, h - barH - 5, barW - 2, barH);
    }

    // Labels
    ctx.fillStyle = '#6666aa';
    ctx.font = '9px monospace';
    ctx.fillText('100%', 0, 12);
    ctx.fillText('50%', 0, h * 0.5 - 2);
    ctx.fillText('0%', 0, h - 2);
    if (n > 0) {
      ctx.fillText('Gen ' + trail[0].gen, 10, h - 2);
      ctx.fillText('Gen ' + trail[n - 1].gen, 10 + (n - 1) * barW, h - 2);
    }

    // Trail info
    var info = document.getElementById('trail-info');
    if (info && n > 0) {
      var latest = trail[n - 1];
      info.textContent = 'Gen ' + latest.gen + ': ' + Math.round(latest.vsRandom * 100) + '% win rate vs random | ' + this.model.snapshots.length + ' snapshots saved';
    }
  }

  _updateStats() {
    var s = this.trainer.getStats();
    document.getElementById('sg').textContent = s.gamesCompleted;
    document.getElementById('sgen').textContent = s.generation;
    document.getElementById('sloss').textContent = s.loss ? s.loss.toFixed(4) : '\u2014';
    document.getElementById('slen').textContent = s.avgGameLength ? s.avgGameLength.toFixed(1) : '\u2014';
    document.getElementById('ssnap').textContent = s.snapshots;
    var trail = s.trail;
    if (trail.length > 0) {
      document.getElementById('svr').textContent = Math.round(trail[trail.length - 1].vsRandom * 100) + '%';
    }
  }

  _startLoop() {
    var self = this;
    var errorCount = 0;
    var loop = function () {
      if (self._destroyed) return;
      try {
        if (!self.paused) {
          for (var t = 0; t < self.ticksPerFrame; t++) {
            self.trainer.tick();
          }
        }
        self.frameCount++;
        // Skip grid rendering at high speed — only render every few frames
        var renderEvery = self.ticksPerFrame > 20 ? 5 : self.ticksPerFrame > 5 ? 2 : 1;
        if (!self.humanPlaying && self.frameCount % renderEvery === 0) {
          self._renderGrid();
        }
        if (self.frameCount % 10 === 0) {
          self._updateStats();
          if (!self.humanPlaying) self._renderTrail();
        }
        errorCount = 0;
      } catch (e) {
        errorCount++;
        if (errorCount < 3) console.warn('Tick error:', e.message, e.stack);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
