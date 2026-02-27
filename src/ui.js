import { Game } from './game';

export class UI {
  constructor(trainer, model) {
    this.trainer = trainer;
    this.model = model;
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

    this._buildDOM();
    this._startLoop();
  }

  destroy() {
    this._destroyed = true;
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
      '.speed-ctrl { display:flex; align-items:center; gap:6px; font-size:12px; }',
      '.speed-ctrl input { width:100px; accent-color:#4488ff; }',
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
      '.controls-row button { margin:0 5px; }'
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
      + '<div class="subtitle">Self-play RL on a plague territory game &mdash; watch 40 games train live</div>'
      + '<div id="stats">'
      + '<div class="stat"><div class="label">Games</div><div class="value" id="sg">0</div></div>'
      + '<div class="stat"><div class="label">Gen</div><div class="value" id="sgen">0</div></div>'
      + '<div class="stat"><div class="label">Loss</div><div class="value" id="sloss">&mdash;</div></div>'
      + '<div class="stat"><div class="label">Avg Len</div><div class="value" id="slen">&mdash;</div></div>'
      + '<div class="stat"><div class="label">P1 Wins</div><div class="value sg" id="sp1">0</div></div>'
      + '<div class="stat"><div class="label">P2 Wins</div><div class="value sr" id="sp2">0</div></div>'
      + '</div>';
    app.appendChild(header);

    // Controls
    var controls = document.createElement('div');
    controls.id = 'controls';

    var pauseBtn = document.createElement('button');
    pauseBtn.id = 'pbtn';
    pauseBtn.textContent = 'Pause';
    var self = this;
    pauseBtn.onclick = function () {
      self.paused = !self.paused;
      pauseBtn.textContent = self.paused ? 'Resume' : 'Pause';
    };
    controls.appendChild(pauseBtn);

    var speedDiv = document.createElement('div');
    speedDiv.className = 'speed-ctrl';
    speedDiv.innerHTML = '<span>Speed:</span><input type="range" id="sspeed" min="1" max="50" value="1"><span id="sval">1x</span>';
    controls.appendChild(speedDiv);

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
      var action = self.model.getAction(state, mask);
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
    ctx.fillStyle = '#0d0d22';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    var gap = cs > 10 ? 1 : 0;
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

  _renderHuman() {
    if (!this.humanGame) return;
    this._drawBoard(this.humanCanvas, this.humanGame.board, this.rows, this.cols, this.humanCellSize);
    var c = this.humanGame.countCells();
    var el = document.getElementById('human-score');
    el.innerHTML = '<span class="sg">You: ' + c.p1 + '</span> | <span class="sr">AI: ' + c.p2 + '</span> | Empty: ' + c.empty;
  }

  _renderGrid() {
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

  _updateStats() {
    var s = this.trainer.getStats();
    document.getElementById('sg').textContent = s.gamesCompleted;
    document.getElementById('sgen').textContent = s.generation;
    document.getElementById('sloss').textContent = s.loss ? s.loss.toFixed(4) : '\u2014';
    document.getElementById('slen').textContent = s.avgGameLength ? s.avgGameLength.toFixed(1) : '\u2014';
    document.getElementById('sp1').textContent = s.p1Wins;
    document.getElementById('sp2').textContent = s.p2Wins;
  }

  _startLoop() {
    var self = this;
    var loop = function () {
      if (self._destroyed) return;
      if (!self.paused) {
        for (var t = 0; t < self.ticksPerFrame; t++) {
          self.trainer.tick();
        }
      }
      if (!self.humanPlaying) {
        self._renderGrid();
      }
      self._updateStats();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
