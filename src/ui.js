import { Game } from './game';
import { MetricsLog } from './metrics';
import { createChartCanvas, drawLineChart } from './charts';

export class UI {
  constructor(trainer, algo, config) {
    this.trainer = trainer;
    this.algo = algo;
    this.config = config || {};
    this.pipelineType = this.config.pipelineType || 'cpu';
    this.rows = trainer.rows;
    this.cols = trainer.cols;
    this.boardSize = this.rows * this.cols;

    this.gridCellSize = 6;
    this.humanCellSize = 30;

    this.gridCanvases = [];
    this.humanCanvas = null;
    this.humanGame = null;
    this.humanPlaying = false;
    this.humanTurn = true;
    this.humanGameOver = false;

    this.ticksPerFrame = 1;
    this.paused = false;
    this._destroyed = false;
    this._frameCount = 0;  // For stats throttling and render-skipping

    this.metrics = new MetricsLog(500);
    this.lastGeneration = -1;
    this._chartsDirty = true;
    this.charts = {};

    this._buildDOM();
    this._startLoop();
  }

  destroy() {
    this._destroyed = true;
  }

  _restart(modelType, algoType, pipelineType) {
    if (!this.config.createPipeline) return;
    this.pipelineType = pipelineType || 'cpu';
    var pipeline = this.config.createPipeline(
      modelType, algoType,
      this.config.rows || this.rows,
      this.config.cols || this.cols,
      this.config.numGames || this.trainer.numGames,
      this.pipelineType
    );
    this.trainer = pipeline.trainer;
    this.algo = pipeline.algo;
    this.rows = this.trainer.rows;
    this.cols = this.trainer.cols;
    this.boardSize = this.rows * this.cols;
    this.metrics = new MetricsLog(500);
    this.lastGeneration = -1;
    this._chartsDirty = true;
    this.humanPlaying = false;

    // Rebuild grid canvases
    var grid = document.getElementById('game-grid');
    grid.innerHTML = '';
    var cw = this.cols * this.gridCellSize;
    var ch = this.rows * this.gridCellSize;
    this.gridCanvases = [];
    for (var i = 0; i < this.trainer.numGames; i++) {
      var cell = document.createElement('div');
      cell.className = 'gcell';
      var canvas = document.createElement('canvas');
      canvas.width = cw;
      canvas.height = ch;
      canvas.style.display = 'block';
      cell.appendChild(canvas);
      grid.appendChild(cell);
      this.gridCanvases.push({ canvas: canvas, cell: cell });
    }

    document.getElementById('training-section').style.display = 'block';
    document.getElementById('human-section').style.display = 'none';
    this._renderCharts();
  }

  _injectStyles() {
    var style = document.createElement('style');
    style.textContent = [
      '* { margin:0; padding:0; box-sizing:border-box; }',
      'body { background:#0a0a1a; color:#c0c0e0; font-family:"Courier New",monospace; }',
      '#app { max-width:900px; margin:0 auto; padding:20px; }',
      'header { text-align:center; margin-bottom:20px; }',
      'header h1 { font-size:32px; color:#00ff88; text-shadow:0 0 20px rgba(0,255,136,0.3); margin-bottom:4px; }',
      '.subtitle { font-size:12px; color:#556; }',
      '#stats { display:flex; justify-content:center; gap:12px; flex-wrap:wrap; margin-top:12px; }',
      '.stat { background:#151530; padding:6px 14px; border-radius:6px; border:1px solid #2a2a5a; text-align:center; }',
      '.stat .label { color:#6666aa; font-size:9px; text-transform:uppercase; letter-spacing:1px; }',
      '.stat .value { font-size:16px; font-weight:bold; color:#ffcc00; }',
      '#controls { display:flex; justify-content:center; align-items:center; gap:14px; margin-bottom:20px; flex-wrap:wrap; }',
      'button { background:#2244aa; color:#fff; border:none; padding:8px 18px; border-radius:6px; cursor:pointer; font-family:inherit; font-size:13px; transition:background 0.2s; }',
      'button:hover { background:#3355cc; }',
      'button.play-btn { background:#008844; }',
      'button.play-btn:hover { background:#00aa55; }',
      'button.restart-btn { background:#aa4422; }',
      'button.restart-btn:hover { background:#cc5533; }',
      '.speed-ctrl { display:flex; align-items:center; gap:6px; font-size:12px; }',
      '.speed-ctrl input { width:100px; accent-color:#4488ff; }',
      'select.cfg-select { background:#151530; color:#c0c0e0; border:1px solid #2a2a5a; padding:6px 10px; border-radius:6px; font-family:inherit; font-size:12px; }',
      '#training-section h2, #human-section h2 { font-size:15px; color:#4488ff; margin-bottom:10px; text-align:center; }',
      '#game-grid { display:flex; flex-wrap:wrap; justify-content:center; gap:3px; background:#0d0d22; padding:10px; border-radius:8px; border:1px solid #1a1a40; }',
      '.gcell { border:2px solid #1a1a40; border-radius:3px; transition:border-color 0.3s; }',
      '.gcell.won1 { border-color:#00ff88; }',
      '.gcell.won2 { border-color:#ff3366; }',
      '#human-section { text-align:center; }',
      '#human-wrap { display:inline-block; border:2px solid #2a2a5a; border-radius:8px; overflow:hidden; margin:10px 0; cursor:pointer; }',
      '#human-info { font-size:14px; margin:10px 0; min-height:22px; }',
      '#human-info.yt { color:#00ff88; }',
      '#human-info.at { color:#ff3366; }',
      '#human-info.go { color:#ffcc00; font-weight:bold; }',
      '#human-score { margin:8px 0; font-size:14px; }',
      '.sg { color:#00ff88; }',
      '.sr { color:#ff3366; }',
      '.controls-row { margin-top:12px; }',
      '.controls-row button { margin:0 5px; }',
      '#charts-section { margin-top:20px; }',
      '#charts-section h2 { font-size:15px; color:#4488ff; margin-bottom:10px; text-align:center; }',
      '#charts-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; max-width:820px; margin:0 auto; }',
      '@media (max-width:600px) { #charts-grid { grid-template-columns:1fr; } }'
    ].join('\n');
    document.head.appendChild(style);
  }

  _buildDOM() {
    this._injectStyles();
    document.body.innerHTML = '';
    var app = document.createElement('div');
    app.id = 'app';
    document.body.appendChild(app);

    // Header
    var header = document.createElement('header');
    header.innerHTML = '<h1>AlphaPlague</h1>'
      + '<div class="subtitle">Self-play RL on a plague territory game &mdash; watch 80 games train live</div>'
      + '<div id="stats">'
      + '<div class="stat"><div class="label">Games</div><div class="value" id="sg">0</div></div>'
      + '<div class="stat"><div class="label">Gen</div><div class="value" id="sgen">0</div></div>'
      + '<div class="stat"><div class="label">Loss</div><div class="value" id="sloss">&mdash;</div></div>'
      + '<div class="stat"><div class="label">Avg Len</div><div class="value" id="slen">&mdash;</div></div>'
      + '<div class="stat"><div class="label">Elo</div><div class="value" id="selo">1000</div></div>'
      + '<div class="stat"><div class="label">vs Ckpt</div><div class="value sg" id="sckpt">&mdash;</div></div>'
      + '<div class="stat"><div class="label">Entropy</div><div class="value" id="sentr">&mdash;</div></div>'
      + '</div>';
    app.appendChild(header);

    // Controls
    var controls = document.createElement('div');
    controls.id = 'controls';
    var self = this;

    var pauseBtn = document.createElement('button');
    pauseBtn.id = 'pbtn';
    pauseBtn.textContent = 'Pause';
    pauseBtn.onclick = function () {
      self.paused = !self.paused;
      pauseBtn.textContent = self.paused ? 'Resume' : 'Pause';
    };
    controls.appendChild(pauseBtn);

    var speedDiv = document.createElement('div');
    speedDiv.className = 'speed-ctrl';
    speedDiv.innerHTML = '<span>Speed:</span><input type="range" id="sspeed" min="1" max="50" value="1"><span id="sval">1x</span>';
    controls.appendChild(speedDiv);

    // Model select
    var modelSel = document.createElement('select');
    modelSel.id = 'model-sel';
    modelSel.className = 'cfg-select';
    modelSel.innerHTML = '<option value="dense">Dense</option><option value="spatial">Spatial (slow)</option>';
    controls.appendChild(modelSel);

    // Algo select
    var algoSel = document.createElement('select');
    algoSel.id = 'algo-sel';
    algoSel.className = 'cfg-select';
    algoSel.innerHTML = '<option value="ppo">PPO</option><option value="reinforce">REINFORCE</option>';
    controls.appendChild(algoSel);

    // Pipeline select (CPU vs GPU)
    var pipeSel = document.createElement('select');
    pipeSel.id = 'pipe-sel';
    pipeSel.className = 'cfg-select';
    pipeSel.innerHTML = '<option value="cpu">CPU</option><option value="gpu">GPU (fast)</option>';
    controls.appendChild(pipeSel);

    // Restart button
    var restartBtn = document.createElement('button');
    restartBtn.className = 'restart-btn';
    restartBtn.textContent = 'Restart';
    restartBtn.onclick = function () {
      self._restart(modelSel.value, algoSel.value, pipeSel.value);
    };
    controls.appendChild(restartBtn);

    var playBtn = document.createElement('button');
    playBtn.className = 'play-btn';
    playBtn.textContent = '\u25B6 Play vs AI';
    playBtn.onclick = function () { self._startHumanGame(); };
    controls.appendChild(playBtn);

    app.appendChild(controls);

    // Speed slider binding (after DOM insertion)
    setTimeout(function () {
      var slider = document.getElementById('sspeed');
      if (slider) {
        slider.oninput = function () {
          self.ticksPerFrame = parseInt(slider.value);
          document.getElementById('sval').textContent = slider.value + 'x';
        };
      }
    }, 0);

    // Training grid
    var trainSection = document.createElement('div');
    trainSection.id = 'training-section';
    trainSection.innerHTML = '<h2>Self-Play Training Grid</h2>';

    var grid = document.createElement('div');
    grid.id = 'game-grid';

    var cw = this.cols * this.gridCellSize;
    var ch = this.rows * this.gridCellSize;
    this.gridCanvases = [];
    for (var i = 0; i < this.trainer.numGames; i++) {
      var cell = document.createElement('div');
      cell.className = 'gcell';
      var canvas = document.createElement('canvas');
      canvas.width = cw;
      canvas.height = ch;
      canvas.style.display = 'block';
      cell.appendChild(canvas);
      grid.appendChild(cell);
      this.gridCanvases.push({ canvas: canvas, cell: cell });
    }
    trainSection.appendChild(grid);
    app.appendChild(trainSection);

    // Charts section
    var chartsSection = document.createElement('div');
    chartsSection.id = 'charts-section';
    chartsSection.innerHTML = '<h2>Training Metrics</h2>';
    var chartsGrid = document.createElement('div');
    chartsGrid.id = 'charts-grid';

    var chartDefs = [
      { key: 'elo', title: 'Elo Rating' },
      { key: 'ckptWin', title: 'Win% vs Checkpoints' },
      { key: 'loss', title: 'Training Loss' },
      { key: 'entropy', title: 'Policy Entropy' },
      { key: 'avgLen', title: 'Avg Game Length' },
      { key: 'selfPlay', title: 'Self-Play P1 Win%' }
    ];
    this.charts = {};
    for (var ci = 0; ci < chartDefs.length; ci++) {
      var cc = createChartCanvas();
      chartsGrid.appendChild(cc);
      this.charts[chartDefs[ci].key] = cc;
    }
    chartsSection.appendChild(chartsGrid);
    app.appendChild(chartsSection);

    // Draw initial empty state
    this._renderCharts();

    // Human play section
    var humanSection = document.createElement('div');
    humanSection.id = 'human-section';
    humanSection.style.display = 'none';
    humanSection.innerHTML = '<h2>You (Green) vs AI (Red)</h2>';

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
    this.humanGame = new Game(this.rows, this.cols);
    this.humanTurn = true;
    this.humanGameOver = false;

    document.getElementById('training-section').style.display = 'none';
    document.getElementById('human-section').style.display = 'block';

    var info = document.getElementById('human-info');
    info.className = 'yt';
    info.textContent = 'Your turn \u2014 click to place (Green)';
    this._renderHuman();
  }

  _backToTraining() {
    this.humanPlaying = false;
    document.getElementById('training-section').style.display = 'block';
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

    if (this.humanGame.isGameOver() || this.humanGame.getValidMoves().length === 0) {
      this._endHumanGame(); return;
    }

    var info = document.getElementById('human-info');
    info.className = 'at';
    info.textContent = 'AI thinking...';

    var self = this;
    setTimeout(function () {
      var state = self.humanGame.getBoardForNN(-1);
      var mask = self.humanGame.getValidMovesMask();
      // GPU trainer implements selectAction directly; CPU uses algo wrapper.
      var actionSource = self.algo || self.trainer;
      var action = actionSource.selectAction(state, mask);
      self.humanGame.makeMove(-1, action);
      self.humanGame.spreadPlague();
      self._renderHuman();

      if (self.humanGame.isGameOver() || self.humanGame.getValidMoves().length === 0) {
        self._endHumanGame();
      } else {
        self.humanTurn = true;
        info.className = 'yt';
        info.textContent = 'Your turn \u2014 click to place (Green)';
      }
    }, 150);
  }

  _endHumanGame() {
    this.humanGameOver = true;
    var winner = this.humanGame.getWinner();
    var info = document.getElementById('human-info');
    info.className = 'go';
    if (winner === 1) info.textContent = '\u2705 You win!';
    else if (winner === -1) info.textContent = '\u274C AI wins!';
    else info.textContent = '\u2696 Draw!';
    this._renderHuman();
  }

  _drawBoard(canvas, board, rows, cols, cs) {
    var ctx = canvas.getContext('2d');
    if (cs <= 10) {
      // Fast path: ImageData for small cells (training grid).
      // Single putImageData vs rows*cols fillRect calls.
      var w = cols * cs;
      var h = rows * cs;
      if (!canvas._imgData || canvas._imgData.width !== w) {
        canvas._imgData = ctx.createImageData(w, h);
      }
      var data = canvas._imgData.data;
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          var val = board[r * cols + c];
          var R, G, B;
          if (val === 1) { R = 0; G = 255; B = 136; }
          else if (val === -1) { R = 255; G = 51; B = 102; }
          else { R = 30; G = 30; B = 58; }
          // Fill cs×cs pixel block
          for (var py = 0; py < cs; py++) {
            var rowOffset = ((r * cs + py) * w + c * cs) * 4;
            for (var px = 0; px < cs; px++) {
              var idx = rowOffset + px * 4;
              data[idx] = R;
              data[idx + 1] = G;
              data[idx + 2] = B;
              data[idx + 3] = 255;
            }
          }
        }
      }
      ctx.putImageData(canvas._imgData, 0, 0);
    } else {
      // Slow path for human play board (large cells, gap between cells)
      ctx.fillStyle = '#0d0d22';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      var gap = 1;
      for (var r = 0; r < rows; r++) {
        for (var c = 0; c < cols; c++) {
          var val = board[r * cols + c];
          if (val === 1) ctx.fillStyle = '#00ff88';
          else if (val === -1) ctx.fillStyle = '#ff3366';
          else ctx.fillStyle = '#1e1e3a';
          ctx.fillRect(c * cs + gap, r * cs + gap, cs - gap * 2, cs - gap * 2);
        }
      }
    }
  }

  _renderHuman() {
    if (!this.humanGame) return;
    this._drawBoard(this.humanCanvas, this.humanGame.board, this.rows, this.cols, this.humanCellSize);
    var c = this.humanGame.countCells();
    var el = document.getElementById('human-score');
    el.innerHTML = '<span class="sg">You: ' + c.p1 + '</span> | <span class="sr">AI: ' + c.p2 + '</span> | Empty: ' + c.empty;
  }

  _renderGrid() {
    if (this.trainer.getBoardsForRender) {
      this._renderGridGPU();
    } else {
      this._renderGridCPU();
    }
  }

  _renderGridCPU() {
    for (var i = 0; i < this.trainer.numGames; i++) {
      var gs = this.trainer.games[i];
      var item = this.gridCanvases[i];
      this._drawBoard(item.canvas, gs.game.board, this.rows, this.cols, this.gridCellSize);
      item.cell.className = 'gcell';
      if (gs.done) {
        if (gs.winner === 1) item.cell.className = 'gcell won1';
        else if (gs.winner === -1) item.cell.className = 'gcell won2';
      }
    }
  }

  _renderGridGPU() {
    var renderData = this.trainer.getBoardsForRender();
    for (var i = 0; i < this.trainer.numGames; i++) {
      var item = this.gridCanvases[i];
      var boardView = renderData.boards.subarray(i * this.boardSize, (i + 1) * this.boardSize);
      this._drawBoard(item.canvas, boardView, this.rows, this.cols, this.gridCellSize);
      item.cell.className = 'gcell';
      if (renderData.done[i]) {
        if (renderData.winners[i] === 1) item.cell.className = 'gcell won1';
        else if (renderData.winners[i] === -1) item.cell.className = 'gcell won2';
      }
    }
  }

  _updateStats() {
    var s = this.trainer.getStats();
    document.getElementById('sg').textContent = s.gamesCompleted;
    document.getElementById('sgen').textContent = s.generation;
    document.getElementById('sloss').textContent = s.loss ? s.loss.toFixed(4) : '\u2014';
    document.getElementById('slen').textContent = s.avgGameLength ? s.avgGameLength.toFixed(1) : '\u2014';

    // Detect generation change → snapshot metrics (eval runs async)
    if (s.generation > this.lastGeneration && s.generation > 0) {
      this.lastGeneration = s.generation;
      this._snapshotMetrics(s);
    }

    // Update stat bar from trainer stats (Elo is live, not per-snapshot).
    // GPU pipeline has no checkpoint pool, so elo/checkpointWinRate are undefined.
    var hasElo = typeof s.elo === 'number' && s.elo > 0;
    document.getElementById('selo').textContent = hasElo ? s.elo.toFixed(0) : (this.pipelineType === 'gpu' ? '--' : '1000');
    var ckptWr = s.checkpointWinRate;
    document.getElementById('sckpt').textContent = ckptWr > 0 ? (ckptWr * 100).toFixed(0) + '%' : '\u2014';
    var last = this.metrics.last();
    if (last) {
      document.getElementById('sentr').textContent = last.entropy.toFixed(3);
    }
  }

  _snapshotMetrics(stats) {
    // GPU pipeline has no algo object; entropy unavailable (reported as 0).
    var entropy = (this.algo && this.algo.lastEntropy) ? this.algo.lastEntropy : 0;
    var totalSelf = stats.p1Wins + stats.p2Wins + stats.draws;
    var selfP1Rate = totalSelf > 0 ? stats.p1Wins / totalSelf : 0.5;

    this.metrics.push({
      generation: stats.generation,
      loss: stats.loss,
      elo: stats.elo || 1000,
      checkpointWinRate: stats.checkpointWinRate || 0,
      selfPlayP1Rate: selfP1Rate,
      entropy: entropy,
      avgGameLength: stats.avgGameLength,
      bufferSize: stats.bufferSize
    });
    this._chartsDirty = true;
  }

  _renderCharts() {
    var data = this.metrics.length > 0;

    drawLineChart(this.charts.elo, data ? this.metrics.getSeries('elo') : [], {
      title: 'Elo Rating',
      color: '#ffaa00',
      refLine: 1000,
      refColor: '#444466'
    });
    drawLineChart(this.charts.ckptWin, data ? this.metrics.getSeries('checkpointWinRate').map(function (v) { return v * 100; }) : [], {
      title: 'Win% vs Checkpoints',
      color: '#00ff88',
      minY: 0,
      maxY: 100,
      refLine: 50,
      refColor: '#444466'
    });
    drawLineChart(this.charts.loss, data ? this.metrics.getSeries('loss') : [], {
      title: 'Training Loss',
      color: '#ffcc00'
    });
    drawLineChart(this.charts.entropy, data ? this.metrics.getSeries('entropy') : [], {
      title: 'Policy Entropy',
      color: '#ff66aa'
    });
    drawLineChart(this.charts.avgLen, data ? this.metrics.getSeries('avgGameLength') : [], {
      title: 'Avg Game Length',
      color: '#66ccff'
    });
    drawLineChart(this.charts.selfPlay, data ? this.metrics.getSeries('selfPlayP1Rate').map(function (v) { return v * 100; }) : [], {
      title: 'Self-Play P1 Win%',
      color: '#4488ff',
      minY: 0,
      maxY: 100,
      refLine: 50,
      refColor: '#444466'
    });
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
        self._frameCount++;

        // Adaptive render-skipping: at high speed, render grid less often.
        // At 1x render every frame; at 10x+ render every 4th frame.
        var renderSkip = self.ticksPerFrame > 5 ? 4 : 1;
        if (!self.humanPlaying && (self._frameCount % renderSkip === 0)) {
          self._renderGrid();
        }

        // Throttle stats DOM updates to every 10 frames
        if (self._frameCount % 10 === 0) {
          self._updateStats();
        }

        if (self._chartsDirty) {
          self._chartsDirty = false;
          self._renderCharts();
        }
        errorCount = 0;
      } catch (e) {
        errorCount++;
        if (errorCount < 5) console.warn('Tick error:', e.message);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
