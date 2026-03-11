# AlphaPlague Pipeline Architectures

## Pipeline stages per tick (40 parallel games)

```
1. Spread plague    — update all boards (cellular automaton step)
2. Encode state     — prepare board states for NN
3. NN forward       — predict policy logits + value
4. Action select    — softmax + sampling from policy
5. Apply action     — place piece on board
6. Check game end   — detect winner/draw
7. Store experience — save (state, action, reward) for training
8. Render           — draw board to screen

Training (every N games):
9. Batch sample     — gather states, actions, rewards
10. Forward + loss  — compute loss through NN
11. Backprop        — gradient descent step
```

## Data sizes (10x10 board, batch=40)

| Data                | Size per game | Batch (40) |
|---------------------|---------------|------------|
| Board state (f32)   | 400 B         | 16.0 KB    |
| Policy logits (f32) | 400 B         | 16.0 KB    |
| Value (f32)         | 4 B           | 160 B      |
| Action mask (f32)   | 400 B         | 16.0 KB    |
| Action index (i32)  | 4 B           | 160 B      |
| Game-over flag      | 1 B           | 40 B       |

---

## The 9 Architectures

### Arch 1: All-CPU Pure JS (no TF.js)

```
CPU: spread → flatten → matmul → softmax → sample → apply → render
```

- Hand-coded NN (matmul loops), hand-coded backprop
- Canvas 2D rendering
- **Transfer**: 0 bytes. Everything in JS heap.
- **Pro**: JIT-optimized tight loops, zero overhead, simplest
- **Con**: No GPU parallelism for NN, scales poorly with model size
- **When**: Small models where GPU dispatch overhead > compute time

### Arch 2: CPU game + TF.js CPU backend

```
CPU: spread → tf.tensor2d (wrap, no copy) → predict (CPU) → dataSync (no-op) → softmax → sample → apply → render
```

- TF.js runs on CPU anyway, tensors wrap existing ArrayBuffers
- **Transfer**: ~0 bytes. Same memory space.
- **Pro**: TF.js API with auto-diff, near-zero transfer
- **Con**: No GPU acceleration, CPU-bound

### Arch 3: CPU game + TF.js WASM backend

```
CPU: spread → [JS→WASM copy: 16 KB] → predict (WASM SIMD) → [WASM→JS copy: 16.2 KB] → softmax → sample → apply → render
```

- TF.js WASM uses SIMD for vectorized matmul
- **Transfer**: 32.2 KB/tick (JS↔WASM linear memcpy, very fast)
- **Pro**: SIMD vectorization, consistent perf, no GPU needed
- **Con**: Still CPU-bound, small copy overhead

### Arch 4: CPU game + GPU NN + CPU action (CURRENT)

```
CPU: spread → [Upload: 16 KB] → GPU: predict → [Download: 16.2 KB] → CPU: softmax → sample → apply → render
```

- Current architecture. Game state on CPU, NN on GPU.
- Full policy logits downloaded for CPU-side masked softmax.
- **Transfer**: 32.2 KB/tick + training uploads
- **Pro**: Simple, well-understood
- **Con**: Downloads ALL logits just to pick ONE action per game

### Arch 5: CPU game + GPU NN + GPU action select

```
CPU: spread → [Upload: 16 KB state + 16 KB mask] → GPU: predict → softmax → argmax → [Download: 160 B] → CPU: apply → render
```

- Move softmax + argmax to GPU. Only download action indices.
- Mask must also be uploaded (or computed on GPU from state).
- **Transfer**: 32.16 KB upload + 160 B download = 32.3 KB/tick
- **Improvement**: 50% less download vs Arch 4
- **Note**: If mask = (state == 0), GPU can derive it: no extra upload needed
- **Caveat**: tf.multinomial has WebGPU bug (first sample always 0). Use argmax for greedy or workaround with slice.

### Arch 5b: Arch 5 with GPU-derived mask (no mask upload)

```
CPU: spread → [Upload: 16 KB] → GPU: mask=equal(state,0) → predict → mask_logits → softmax → argmax → [Download: 160 B] → CPU: apply → render
```

- **Transfer**: 16 KB upload + 160 B download = 16.16 KB/tick
- **Improvement**: ~50% less total transfer vs Arch 4

### Arch 6: GPU game state + GPU NN + GPU action + CPU render

```
GPU: spread_kernel(state) → predict(state) → softmax → argmax → [Download: 160 B actions + 40 B game_over] → CPU: render via tf.browser.draw or Canvas 2D
```

- Game state lives on GPU as a tensor.
- Spread implemented via TF.js ops: pad + conv2d (neighbor sum) + random + clip.
- Action applied via scatter: `state + oneHot(action) * player`
- Game-over check: `tf.sum(tf.abs(state)) == boardSize` (all cells filled)
- Only download action indices + game-over flags for game management.
- Rendering options:
  a. tf.browser.draw(stateTensor, canvas) — GPU direct, no CPU roundtrip
  b. dataSync() the state for Canvas 2D — adds 16 KB download, only needed for display
- **Transfer**: 200 B/tick (actions + game-over). Render adds 16 KB only when displaying.
- **Pro**: Near-zero transfer in game loop
- **Con**: Plague spread as TF.js ops is awkward (random element, neighbor logic). Must track game metadata (whose turn, winner) somehow.

### Arch 7: Full WebGPU custom (no TF.js)

```
GPU: spread_wgsl(buf) → matmul_wgsl(buf) → softmax_wgsl → sample_wgsl → apply_wgsl → render_pipeline(same buf)
```

- Everything via custom WGSL compute shaders.
- Game state: GPUBuffer with ping-pong pattern.
- NN: Custom matmul/conv compute shaders.
- Render: WebGPU render pipeline reading same GPUBuffer as instance data.
- **Transfer**: 0 bytes. Nothing ever leaves GPU.
- **Pro**: Absolute maximum GPU utilization, zero overhead
- **Con**: Must implement entire NN + backprop in WGSL. Enormous engineering effort.
  No auto-diff. Debugging in WGSL is painful. No tensor library.
- **When**: Production game engine, not research/prototyping

### Arch 8: TF.js-native GPU pipeline (RECOMMENDED FOR EXPLORATION)

```
GPU tensor state throughout:
  spreadOp(stateTensor) → model.predict(stateTensor) → mask+softmax+argmax → scatterUpdate(stateTensor, actions) → tf.browser.draw(stateTensor, canvas)
  [Download: 160 B action indices for game logic bookkeeping]
```

- All game state lives as tf.Tensor on GPU.
- Spread: `tf.conv2d` with ones kernel for neighbor count, `tf.mul(tf.randomUniform(...))`, `tf.clipByValue`, etc.
- Mask: `stateTensor.equal(0).cast('float32')`
- NN forward: `model.predict(stateTensor)` — state already on GPU, zero upload
- Action: `logits.add(mask.sub(1).mul(1e9)).softmax().argMax(1)` — stays on GPU
- Apply: `state.add(tf.oneHot(actions, boardSize).mul(player))` — stays on GPU
- Render: `tf.browser.draw(stateTensor.reshape([rows, cols, 1]), canvas)` — GPU direct
- Training: `optimizer.minimize(...)` — all tensors already on GPU
- **Transfer**: ~160 B/tick (only action indices for game logic metadata)
- **Pro**: Uses TF.js API (auto-diff works), minimal engineering, near-zero transfer
- **Con**: Game logic via TF.js ops is less natural than JS. Some ops may be slower
  on GPU than CPU for small boards. Random element needs care.

### Arch 9: Hybrid — CPU game + GPU NN + WebGPU render

```
CPU: spread → [Upload: 16 KB to shared GPUBuffer] → GPU: predict → softmax → argmax → [Download: 160 B]
CPU: update game state
WebGPU: render pipeline reads shared buffer (same GPU memory as NN input)
```

- Game logic stays on CPU (simple, debuggable JS)
- NN on GPU via TF.js WebGPU
- Render via custom WebGPU render pipeline that reads the state GPUBuffer
- State uploaded once, used for both NN and rendering
- **Transfer**: 16 KB upload + 160 B download = 16.16 KB/tick
- **Pro**: Clean separation, JS game logic stays simple, GPU rendering
- **Con**: Needs custom WebGPU render pipeline. Upload still needed.

---

## Bandwidth Comparison (10x10, batch=40, per tick)

| Arch | Upload     | Download   | Total     | Complexity |
|------|-----------|------------|-----------|------------|
| 1    | 0         | 0          | **0**     | Low        |
| 2    | 0         | 0          | **0**     | Low        |
| 3    | 16 KB     | 16.2 KB    | 32.2 KB   | Low        |
| 4    | 16 KB     | 16.2 KB    | **32.2 KB** | Low (current) |
| 5    | 16 KB     | 160 B      | 16.16 KB  | Low        |
| 5b   | 16 KB     | 160 B      | 16.16 KB  | Low        |
| 6    | 0         | 200 B      | **200 B** | Medium     |
| 7    | 0         | 0          | **0**     | Very High  |
| 8    | 0         | 160 B      | **160 B** | Medium     |
| 9    | 16 KB     | 160 B      | 16.16 KB  | Medium     |

## Scaling: What matters at 19x19 (361 cells)?

| Arch | Upload      | Download     | Total       |
|------|------------|--------------|-------------|
| 4    | 57.8 KB    | 57.9 KB      | **115.7 KB** |
| 5b   | 57.8 KB    | 160 B        | 58.0 KB     |
| 8    | 0          | 160 B        | **160 B**   |

At 19x19, current Arch 4 transfers ~116 KB/tick. Arch 8 transfers 160 bytes.
That's a **723x reduction**.

---

## Training data transfer comparison

Training involves uploading a batch of (state, action, reward) tuples.

| Arch | Training upload (batch=256, 10x10)     |
|------|----------------------------------------|
| 4    | 256 × (400 + 400 + 4) B = **201 KB**  |
| 8    | 0 (experience buffer lives on GPU)     |
| 7    | 0                                      |

For Arch 8, the experience buffer can be a circular GPU tensor.
States are already tensors, actions are tensors, rewards are tensors.
Training stays entirely on GPU.

---

## Plague spread as TF.js ops (for Arch 6/8)

```js
function spreadTF(stateTensor, rows, cols) {
  // stateTensor: [batch, rows*cols] → reshape to [batch, rows, cols, 1]
  var grid = stateTensor.reshape([-1, rows, cols, 1]);

  // Neighbor kernel: sum of 4 neighbors (no diagonal)
  var kernel = tf.tensor4d([
    [[0],[1],[0]],
    [[1],[0],[1]],
    [[0],[1],[0]]
  ], [3, 3, 1, 1]);

  // Neighbor sum (each cell gets sum of its 4 neighbors)
  var neighborSum = tf.conv2d(grid, kernel, 1, 'same');

  // Random factor per cell
  var rand = tf.randomUniform(neighborSum.shape);
  var weighted = neighborSum.mul(rand).mul(2);

  // Only update empty cells (state == 0)
  var emptyMask = grid.equal(0).cast('float32');

  // Truncate to {-1, 0, 1}: sign of weighted sum, clamped
  var newVals = weighted.sign().mul(emptyMask);

  // Keep existing non-empty cells + fill empty cells
  var existing = grid.mul(grid.abs()); // non-empty cells keep their value
  // Actually: result = state * (1 - emptyMask) + newVals
  var result = grid.mul(tf.scalar(1).sub(emptyMask)).add(newVals);

  return result.reshape([-1, rows * cols]);
}
```

This runs entirely on GPU. No CPU involvement.

---

## Recommendation

**Start with Arch 5b** (GPU action select, GPU-derived mask) as a quick win:
- Minimal code change from current Arch 4
- Eliminates downloading full logits (biggest transfer)
- Only change: replace CPU maskedSoftmax with GPU ops in selectActions

**Then explore Arch 8** (full TF.js GPU pipeline) for maximum benefit:
- Plague spread as TF.js ops (see above)
- State stays on GPU permanently
- Training stays on GPU permanently
- Only download action indices for game management
- Render via tf.browser.draw

**Benchmark all architectures** by measuring:
- Per-tick latency (wall clock)
- GPU utilization
- Bytes transferred per tick
- Memory footprint (GPU + CPU)

This replaces isolated stage benchmarks with end-to-end architecture comparisons.
The question isn't "how fast is dataSync?" but "which architecture avoids it entirely?"
