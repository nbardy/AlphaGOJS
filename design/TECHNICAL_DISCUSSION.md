# AlphaPlague — Technical Discussion

## Current Architecture

### Performance Profile (4-core Xeon, no GPU, TF.js CPU backend)

| Component | Throughput | Latency | Engine |
|---|---|---|---|
| `spreadPlague()` | **228k**/sec | 4μs | Pure JS `Int8Array` |
| `getBoardForNN()` | **1.1M**/sec | 0.9μs | Pure JS `Float32Array` |
| `getValidMovesMask()` | **1.0M**/sec | 1μs | Pure JS `Float32Array` |
| Full random game (no NN) | **13,739**/sec | 73μs / 8.4 turns | Pure JS |
| NN single inference | **3,040**/sec | 0.33ms | TF.js CPU |
| NN batch=40 inference | **129**/sec | **7.7ms** | TF.js CPU |
| Training step (batch=256) | **3.5**/sec | **288ms** | TF.js CPU |

**Key insight**: Game logic is ~100x faster than NN inference. The bottleneck is entirely TF.js on CPU. With WebGPU, batch=40 inference should drop to <1ms and training to ~20ms.

### Why the Game Logic is So Fast

`spreadPlague()` and `getBoardForNN()` are pure JavaScript on typed arrays — zero TensorFlow involvement. The original prototype used TF.js tensors for game state (`tf.zeros`, `tf.pad`, `tf.slice`), which required GPU round-trips, dtype casting, and tensor allocation/disposal. The rewrite uses `Int8Array` for the board (100 bytes) and `Float32Array` for NN input (400 bytes). These are contiguous memory, cache-friendly, and JIT-optimized by V8. No framework overhead.

---

## Rendering Analysis

### Current: Canvas 2D `fillRect`

40 canvases × 100 cells = 4,000 `fillRect` calls per frame. At 60fps = 240k `fillRect`/sec. Measured overhead: <2ms per frame. **Not the bottleneck.**

### Alternative Rendering Methods

| Method | Mechanism | Expected Speedup | Complexity |
|---|---|---|---|
| **ImageData + putImageData** | Write raw RGBA pixels to buffer, 1 call per canvas | 2-3x over fillRect | Trivial |
| **Single large canvas** | Render all 40 boards on one canvas, eliminate per-canvas overhead | 3-5x | Low |
| **WebGL texture atlas** | Upload all 40 boards as one texture, render as single quad with UV mapping | 10-50x | Medium |
| **WebGPU render pipeline** | Compute shader updates board texture, render shader displays it — zero CPU involvement per frame | 100x+ | High |
| **Skip rendering on fast ticks** | Only render every Nth tick when speed slider > 10x | Instant, free | Trivial |

**Recommendation**: For <100 games, `ImageData` on a single canvas is the sweet spot. For 1000+ games, WebGPU compute+render is the right move. Skipping frames at high speed is free and should be done regardless.

### WebGPU for Both Compute and Render

The plague spread mechanic is a cellular automaton — the exact use case WebGPU compute shaders excel at. A WGSL compute shader could:

1. Run `spreadPlague()` for ALL games in parallel (one workgroup per game, one thread per cell)
2. Write results directly to a texture
3. Render that texture — zero CPU readback

This would move game logic from 228k calls/sec (CPU, sequential) to millions of games/sec (GPU, massively parallel). The board is only 10×10 = 100 cells — a single workgroup can handle one game trivially.

### WebGPU Hardware Support

- **Intel integrated GPUs**: Supported on Chrome 144+ (Linux Gen12+), Chrome 113+ (Windows via D3D12), Safari 26+ (macOS via Metal)
- **Web Workers**: WebGPU is available in dedicated, shared, and service workers since Chrome 124
- **Chrome 145** (Jan 2026): Added experimental `mapSync()` for synchronous buffer access in workers

---

## RL Algorithm Analysis

### Current: Vanilla REINFORCE

```
Algorithm: Monte Carlo Policy Gradient (REINFORCE)

For each game played to completion:
  For each (state, action, player) in game history:
    reward = +1 if player won, -1 if lost, 0 if draw
    Store (state, action, reward)

Every 20 completed games:
  Sample batch of 256 from experience buffer
  Loss = -E[reward · log π(action|state)] - 0.01 · H(π)
  Adam step (lr=0.001)
```

**What's good:**
- Simple, correct, works
- Entropy bonus prevents collapse
- Board flipping (×player) lets one network play both sides
- Batched inference amortizes NN overhead

**What's bad:**

1. **No value baseline → extreme variance.** Every move in an 8-turn game gets identical reward (+1 or -1). Move 1 (critical positioning) and move 7 (board already decided) get the same gradient signal. A learned value baseline `V(s)` would let us compute advantage `A = R - V(s)`, focusing gradients on moves that actually mattered.

2. **Binary reward → lost information.** Winning 90-10 and winning 51-49 produce identical +1 reward. The margin of victory is useful signal. Should use continuous reward: `(my_cells - opp_cells) / total_cells`.

3. **No temporal credit assignment.** No discount factor γ. No GAE (Generalized Advantage Estimation). All moves weighted equally regardless of when they occurred.

4. **Single policy self-play → strategy cycling.** The policy finds a trick, beats itself, then the trick becomes both players' strategy, cancels out. No historical opponents to anchor against. This is the failure mode AlphaZero's snapshot league was designed to prevent.

5. **Synchronous batch with dead slots.** Finished games wait 300ms before restarting. During that time the batch is smaller than 40. Should immediately replace finished games.

6. **Training blocks main thread.** The 288ms training step freezes rendering. Should run in a Web Worker.

7. **FIFO experience buffer.** No prioritized replay. Rare decisive moments have the same sampling probability as routine moves.

8. **No lookahead.** Pure reactive policy. AlphaGo/AlphaZero used MCTS (Monte Carlo Tree Search) with the policy-value network to search ahead. Even 1-step lookahead would help.

### Reward Design Deep Dive

**Current**: Binary endpoint reward. Every move gets `+1` (winner) or `-1` (loser).

**Problem**: This is a 100-cell board with ~8 turns per game = ~16 decisions. Each decision gets the same reward regardless of its actual impact. The signal-to-noise ratio is terrible.

**Better options (ranked by impact):**

1. **Margin-based final reward**: `reward = (my_cells - opp_cells) / total_cells` ∈ [-1, +1]. Winning big gives stronger signal. Easy to implement.

2. **Value function baseline (A2C)**: Train `V(s)` to predict expected game outcome from state `s`. Advantage `A = R - V(s)` isolates each move's contribution. Dramatically reduces variance.

3. **Intermediate reward shaping**: Give small rewards for spreading (territory gained this turn). Risk: can cause reward hacking (optimize spread at the expense of winning).

4. **Discounted returns**: `G_t = Σ γ^(T-t) · R_T`. Moves closer to the end get more credit. Simple, no value network needed.

### Batch Management Deep Dive

**Current flow:**
```
tick():
  1. Collect P1 states from active games → batch predict (≤40)
  2. Execute P1 moves
  3. Collect P2 states from active games → batch predict (≤40)
  4. Execute P2 moves
  5. Spread plague for all active games
  6. Check game over → mark done, wait 300ms
  7. Restart finished games (after 300ms)
```

**Problems:**
- 2 separate NN calls per tick (P1 and P2). Could be 1 call with batch=80.
- Dead slots during 300ms cooldown.
- All games are lockstep — an 8-turn game and a 12-turn game both advance 1 tick at a time.

**Ideal: Fully streaming async batch:**
```
Maintain fixed batch of N game states.
Each slot is either:
  - PLAYING: needs next move
  - DONE: game finished, needs replacement

On each step:
  1. Collect ALL slots needing moves (both P1 and P2 in one batch)
  2. Single batched inference
  3. Apply moves + spread
  4. For finished games: harvest experiences, immediately replace with fresh game
  5. Never wait, never have dead slots
```

This keeps the batch always full and maximizes GPU utilization.

---

## ELO Scoring in Self-Play

### The Problem

If the bot always plays against the current version of itself, win rate is always ~50%. There's no external reference to measure improvement.

### Solutions (ranked by complexity)

**1. Win rate vs Random Agent** (trivial)
- Play 100 games against uniform random policy
- Track over time. Should go from ~50% to >95%.
- Cheap sanity check, but saturates quickly.

**2. Win rate vs Heuristic Bots** (easy)
- Implement 3-4 simple strategies: center-first, greedy-spread, block-opponent, random
- Measure win rate against each
- Provides a richer signal than random alone

**3. Snapshot League** (medium, the right approach)
- Every N generations, freeze a copy of the network weights
- Play round-robin between current policy and all snapshots
- Compute ELO from pairwise win rates
- This is exactly what AlphaZero did
- Storage cost: ~300KB per snapshot (the dense layers)

**4. Population-Based Training (PBT)** (hard)
- Maintain K separate networks with different hyperparameters
- Each trains via self-play against the pool
- Full ELO leaderboard across the population
- Explores hyperparameter space automatically
- Heavy: K× the compute

**5. TrueSkill / Bayesian Rating** (medium)
- More principled than ELO for 1v1 games
- Tracks uncertainty (σ) alongside skill (μ)
- Better calibrated with fewer games
- Microsoft's TrueSkill2 handles draws, partial information

**Recommendation**: Start with Snapshot League. Save weights every 100 generations. Play 50 games per matchup. Plot ELO curve. It's the best bang for the buck and directly mirrors AlphaZero's approach.

---

## Parallelism in JavaScript

### Web Workers

| Feature | Support | Notes |
|---|---|---|
| Dedicated Workers | Universal | Separate thread, message passing via `postMessage` |
| SharedArrayBuffer | Chrome 92+, Firefox 79+ | Zero-copy shared memory between threads |
| Atomics | Same as SAB | Lock-free synchronization primitives |
| OffscreenCanvas | Chrome 69+, Firefox 105+ | Render in worker, transfer to main thread |
| WebGPU in Workers | Chrome 124+ | Full GPU access from worker threads |
| `mapSync()` on Workers | Chrome 145+ (experimental) | Synchronous GPU buffer reads |

### Ideal Architecture

```
Main Thread
├── UI event handling (clicks, sliders)
├── Stats display updates
└── Canvas compositing (OffscreenCanvas transfer)

Worker 1: Game Simulation
├── Run N games in parallel (pure JS, typed arrays)
├── Batch states → SharedArrayBuffer
├── Read actions ← SharedArrayBuffer
└── Harvest experiences → training buffer

Worker 2: Neural Network (WebGPU)
├── Read states from SharedArrayBuffer
├── Batched inference via WebGPU
├── Write actions to SharedArrayBuffer
└── Training loop (when buffer full)

Worker 3: Rendering (Optional)
├── OffscreenCanvas
├── Read board states from SharedArrayBuffer
└── Render via WebGPU or Canvas 2D
```

**SharedArrayBuffer** is the key primitive: zero-copy shared memory between workers. Game states, NN inputs/outputs, and experience buffers can all live in shared memory. No `postMessage` serialization overhead.

---

## WebGPU Acceleration Opportunities

### 1. NN Inference (biggest win)

TF.js 4.x supports `@tensorflow/tfjs-backend-webgpu`. Drop-in replacement for CPU backend.

Expected speedup: **3-10x** for our small dense model. Batch=40 inference: 7.7ms → ~1-2ms. Training: 288ms → ~30-60ms.

Caveat: TF.js WebGPU backend is inference-focused. Training gradient ops may not all be implemented. May need to keep training on CPU or use a custom WebGPU training kernel.

### 2. Game Simulation (moderate win, fun engineering)

Plague spread is a stencil operation on a 2D grid — textbook compute shader material.

```wgsl
@compute @workgroup_size(10, 10)
fn spreadPlague(@builtin(global_invocation_id) id: vec3<u32>) {
    let game_idx = id.z;
    let r = id.y;
    let c = id.x;
    let i = game_idx * 100 + r * 10 + c;
    
    if (board[i] != 0) { return; }
    
    var sum: f32 = 0.0;
    if (r > 0)  { sum += f32(board[i - 10]) * random(i, 0); }
    if (r < 9)  { sum += f32(board[i + 10]) * random(i, 1); }
    if (c > 0)  { sum += f32(board[i - 1])  * random(i, 2); }
    if (c < 9)  { sum += f32(board[i + 1])  * random(i, 3); }
    
    new_board[i] = clamp(i32(trunc(sum * 2.0)), -1, 1);
}
```

One dispatch handles ALL games simultaneously. For 1000 games: 1000 × 100 = 100k threads — trivial for any GPU.

### 3. Rendering (visual polish)

Write game states directly to a storage texture. Render with a simple fragment shader that maps -1→red, 0→dark, 1→green. One draw call for all 40+ canvases.
