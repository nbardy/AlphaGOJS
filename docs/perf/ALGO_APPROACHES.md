# Backward Pass Optimization Approaches
## AlphaGOJS PPO Kernel Performance Analysis

**Context:** The fused WebGPU PPO kernel in `src/fused_ppo.wgsl` has a pathologically slow backward pass—~503 ms/epoch vs 58 ms rollout at D=8, B=256 (M4). The backward is ~96% of training time, despite computing fundamentally the same operations as a forward pass in reverse. This is 2–10× slower than CPU PyTorch baselines and suggests the kernel is occupancy-starved, not compute-bound.

**Diagnostic facts:**
- **Current kernel:** ONE workgroup (64 threads) per board, running a **serial 64-iteration patch loop** (line 604–668).
- **Recomputation strategy (gradient checkpointing):** Activations are NOT stored; they are recomputed during backward → trades memory for compute.
- **Memory footprint:** Per-thread private arrays accumulating gradients:
  - `l_dWf[2*D*D]` (e.g., 128 f32 at D=8) — conv2 filter gradients
  - `l_dE_cell[4*D]` (e.g., 32 f32 at D=8) — embedding gradients
  - `l_dWpi[D]`, `l_dWv[D]`, `l_dbv` — small policy/value heads
  - Total per thread: ~200 f32 at D=8 (800 bytes) → **51 KB per workgroup**.
- **Barrier overhead:** 30 `workgroupBarrier()` calls in `ppo_step` alone; the serial patch loop has **nested barriers** (line 645, 667).
- **Occupancy starvation:** Only 64 active threads processing 576 cells serially → average utilization ~11% if GPU has >4 wavefronts/SM and other warps are not starved by private memory.

---

## Approach 1: STORE-ACTIVATIONS (Lean on Memory)

### Core Idea
Instead of recomputing forward activations during backward (`conv1_reg`, `decoded`, `fused`, `cell_e` embeddings), **store them to global memory during the forward pass**, then **read them back in the backward**.

### Why Faster
- **Eliminates recomputation:** Each (patch, cell, output-channel) tuple that recomputes costs ~10–20 FLOPs (conv2 @ 9×D inputs, ReLU check). At 64 patches × 576 cells × 9 channels ≈ **331k recomputation ops** → **3–6 ms of GPU time** if not memory-starved.
- **Straightforward backward:** The backward loop becomes pure gradient accumulation—no function evaluation, no divergent branches (ReLU mask is precomputed).
- **Predictable memory access:** Stores are sequential (coalesced); reads during backward are structured (all threads pull from adjacent memory).

### Memory ↔ Compute Tradeoff

**Storage cost per board:**
```
per-step activations:
  sh_a (patch embed):           P×P×D  = 8×8×D   f32 = 2 KB (D=8)
  conv1_reg (per cell):         K×K×D  = 24×24×D f32 = 55 KB (D=8)
  decoded (conv2 output):       K×K×D  = 55 KB
  fused (post-fuse):            K×K×D  = 55 KB
  cell embeddings already used  → no extra cost (already in backward)
  
total per step:  ~160 KB (D=8), ~640 KB (D=16)
total per board (T≈22 steps):  ~3.5 MB (D=8), ~14 MB (D=16)
```

**At B=256 (typical batch):**
- D=8: 256 × 3.5 MB = **900 MB additional VRAM** (M4 has 10–20 GB; ~5% overhead)
- D=16: 256 × 14 MB = **3.6 GB** (more concerning; ~20–35% of budget)

**Compute saved:**
- Patch embed: recomputed P×P times per step (64 times) → eliminate 64 × 72D × 1ops = 1.4K ops/step
- Conv1: recomputed P×P times per step (64 times) → eliminate 64 × 9D² × 1 = 4.6K ops/step
- Conv2+Fuse: recomputed K×K times per step (576 times) → eliminate **576 × 9D² × 2 = 83K ops/step** (dominant)

**Total saved:** ~90K ops/step × 22 steps = **2M ops/board**, or **~0.5–2 ms** per board per PPO epoch (scales with D²).

### Occupancy Impact
- **Shared memory:** Frees ~51 KB per workgroup (all the private arrays stay, but the forward can re-use scratch).
- **Occupancy:** Minimal change; the forward is already bandwidth-bound on stores; the backward becomes read-bound instead.
- **Latency:** Forward latency increases slightly (one extra write per cell), but overlap hides most of it.

### Implementation Difficulty & Risk
**Difficulty:** Medium. Requires a second global buffer (`activation_cache`), synchronized allocation per board, and careful indexing.

**Risk factors:**
1. **Memory explosion at D=16:** 3.6 GB is feasible but tight; may evict other GPU resources.
2. **Cache coherency:** WebGPU does NOT have atomic memory operations for f32; reads/writes are incoherent across workgroups. Safe only if forward and backward are synchronized (they are, via CPU dispatch).
3. **Precision:** Storing f32 → reading f32 is exact; no numerical concern.

### Validation
1. **Correctness:** Run forward, freeze activations, backward with checkpointed activations vs. stored activations. Should match exactly (f32 == f32).
2. **Performance:** Measure forward time (should be ~5% slower due to stores), backward time (should drop 10–20%), net PPO epoch time.
3. **Memory:** Check VRAM utilization with `WebGPU.device.getAdapterInfo().maxBufferBindingSize`.

### Expected ROI at D=8, B=256
- **Time saved:** 0.5–2 ms per epoch, backward goes 503 ms → 490–500 ms. **1–2% win**, not transformative.
- **Feasibility:** High (no algorithmic complexity).
- **Verdict:** Low priority. The memory cost is acceptable, but the compute savings are tiny. **Better suited to D=16 where recomputation overhead is cubic in D.**

---

## Approach 2: PER-CELL / PER-REGION / PER-PATCH Backward Parallelization

### Core Idea
**Restructure to parallelize across cells, patches, or channels** instead of looping serially. Today, 64 threads iterate through 64 patches sequentially (line 604: `for patch_idx = 0..63`), with nested inner loops (line 657–667). Reorganize so threads process **distinct (patch, channel) or (cell, output-dim) pairs in parallel**.

**Two variants:**
1. **PER-PATCH parallelization:** Assign P×P=64 threads to patches (1 thread per patch). Each thread accumulates gradients for that patch's cells locally, then tree-reduces across channels within shared memory.
2. **PER-CELL + PER-CHANNEL:** Assign threads to (cell, output-channel) tuples. All 64 threads stay busy; each accumulates a small gradient for its (y, x, d) tuple, then reduce across the full backward graph.

### Why Faster
- **Occupancy:** All 64 threads active on each instruction (vs. 1 active in the serial loop). Latency hiding improves dramatically.
- **Barrier reduction:** Today, ~20 barriers in the patch loop (nested); parallelization reduces to ~5 global reductions.
- **Memory efficiency:** Instead of `l_dWf[2*D*D]` per thread (51 KB × 64), share a smaller **per-channel scratch** (64 f32, reused across patches).

### Memory ↔ Compute Tradeoff

**Variant 2A (per-patch):**
```
per-thread workspace shrinks from:
  l_dWf[2*D*D] = 128 f32  (D=8)
  l_dE_cell[4*D] = 32 f32
to:
  l_dWf_patch[D*D] = 64 f32  (one patch's filter gradient)
  (amortize l_dE_cell via atomic shared memory, see Approach 3)

workspace per thread: ~100 f32 (vs. 160 currently)
savings: ~240 bytes per thread × 64 = **15 KB per workgroup**
```

**Variant 2B (full parallelization):**
```
Each thread accumulates gradients for a single output channel across all patches:
  per-thread: [D values] for (convW, cellE, policy, value, fuse gradients)
  ~6D f32 per thread = 48 f32 (D=8)
  
total: 48 × 64 = 3 KB (huge win, but requires full reduction at end)
```

### Occupancy Impact
- **Threads per iteration:** 64 vs. 1 currently. **64× improvement in parallelism.**
- **Wavefront occupancy:** If each thread uses <100 registers and shared memory <10 KB, occupancy can hit 80–100% (typical GPUs run 4–8 wavefronts per SM; 64 threads is one full wave on modern SIMD).
- **Latency:** Per-thread computation increases slightly (each thread does more bookkeeping), but the *total* latency drops due to parallelism.

### Implementation Difficulty & Risk
**Difficulty:** High. Requires rewriting the patch loop and gradient accumulation logic.

**Risk factors:**
1. **Correctness:** ReLU masks and state-dependent branches become thread-local. Must ensure no off-by-one indexing.
2. **Shared memory bank conflicts:** If all 64 threads read/write `sh_patch_delta2` at the same indices, there will be conflicts. Mitigated by padding or careful stride.
3. **Reduction complexity:** Instead of reducing a few values (7 in `RED_TOTAL`), now reducing D×D values. Tree-reduce becomes more complex.

### Validation
1. **Numerical correctness:** Per-thread gradients computed independently, then reduced. Should match the serial version exactly.
2. **Performance:** Forward time unchanged; backward time should drop 25–40% if occupancy improves from 11% to 80%+.
3. **Regression test:** Run both serial and parallel, compare Adam weight updates (should be identical up to FP32 rounding).

### Expected ROI at D=8, B=256
- **Time saved:** Backward 503 ms → 300–375 ms (25–40%). **~130–200 ms win per epoch.**
- **At 3 PPO epochs:** 390–600 ms saved per training step = **5–8% overall training speedup.**
- **Feasibility:** Medium-high (requires careful design, but no new GPU features).
- **Verdict:** **High priority.** Largest ROI of the pure-parallelization approaches. Risk is manageable with thorough testing.

---

## Approach 3: ATOMIC Gradient Accumulation to Shared/Global Memory

### Core Idea
Instead of **per-thread private arrays** (`l_dWf[128]`, etc.), use **atomic operations** (`atomicAdd` equivalent, or bitcast tricks for f32) to accumulate gradients directly to **shared memory** or **global memory** (per-board buffer). Eliminates the massive per-thread workspace and makes all threads synchronize on the same memory.

### Why Faster
- **Private memory elimination:** Frees 51 KB per workgroup. Modern GPUs cache-flush private memory at barriers; less private memory = fewer cache flushes.
- **Reduced per-thread state:** Threads don't carry large arrays. Context-switch overhead drops.
- **Atomic operations are cheap:** Modern GPUs (Apple Metal, NVIDIA, AMD) have fast f32 atomics (even if not in WGSL spec, can be emulated via `atomicOr` + bitcast).

### Memory ↔ Compute Tradeoff

**Shared memory for gradient accumulation:**
```
Instead of per-thread:
  64 threads × 160 f32 = 10 KB (currently private/spilled)

Use shared atomic target:
  dWf: 128 f32 (one patch's filter gradients)
  dE_cell: 32 f32 (cell embeddings)
  total per "batch": ~160 f32 shared
  
trade-off: all patches must serialize updates via atomics (1 per step)
```

**Atomic contention:**
- At 64 threads per patch, ~9 threads per cell (64 threads / 7 regions), contention is **O(D²) atomics per cell**. Metal/AMD handle this at L1 cache speeds (~1–2 ns per atomic).
- **Bandwidth:** At 576 cells × 9D atomics = ~4k atomics per epoch. Negligible compared to memory bandwidth.

### Occupancy Impact
- **Shared memory:** Now 160 f32 (shared) vs. 51 KB private (freeing ~25× capacity for other warps).
- **Occupancy:** Can increase significantly if shared-memory was the limiter (today, likely not; private memory is the culprit via spill).
- **Synchronization:** Atomics introduce **acquire/release semantics**. Each atomic has ~10–100 ns latency; at 64 threads, this adds up. Mitigated by coalescing: batch atomics per patch, then one barrier per patch (vs. many barriers today).

### Implementation Difficulty & Risk
**Difficulty:** Medium-High. Requires WebGPU atomic support.

**Critical issue: WGSL does NOT have f32 atomics (as of Feb 2025).** Only i32/u32 atomics are standard.

**Workaround (bitcast trick):**
```wgsl
// Emulate atomicAddF32 via bitcast
fn atomicAddF32(ptr: ptr<shared, f32>, val: f32) -> f32 {
  var cast = bitcast<u32>(val);
  var old = atomicAdd(ptr_cast<shared, u32>(ptr), cast);
  return bitcast<f32>(old);
}
```
**Problem:** This is **incorrect for floating-point addition** (not associative). Reordering due to atomics can cause different rounding errors. Acceptable for gradients (1e-4 tolerance), unacceptable for loss/convergence-critical values.

**Alternative: Compare-and-swap loop:**
```wgsl
loop: if atomicCompareExchangeWeak(..., old, new) succeeds, break
```
**Problem:** WebGPU does NOT have compare-and-swap for f32. Must use i32 cas + bitcast + retry. Very slow.

### Validation
1. **Bitcast-atomic gradients:** Allow ~1e-4 relative error (order-of-accumulation variance).
2. **Match forward:** Compare Adam-updated weights with serial version; should match to ~1e-3 (due to atomic reordering).
3. **Convergence:** Run full training, check loss curve and final Elo; should not diverge.

### Expected ROI at D=8, B=256
- **Time saved:** Reduced private memory spilling saves ~10–20 ms per epoch (if spill is the bottleneck). But atomics overhead might add 5–10 ms.
- **Net:** 0–10 ms, likely **not a win** without evidence of spilling.
- **Feasibility:** Low (requires workarounds; atomics are slow; numerical correctness is suspect).
- **Verdict:** **Low priority.** Only pursue if Approach 2 (parallelization) doesn't resolve occupancy. Atomic overhead often negates memory savings.

---

## Approach 4: SUBGROUP-Accelerated Reductions

### Core Idea
Replace the **barrier-heavy 2-at-a-time tree reductions** (line 325–330, 341–346, etc.) with **WebGPU subgroup operations** (`subgroupAdd`, `subgroupShuffle`, etc.). Subgroups are intra-wave primitives (no barrier needed), so reductions complete in 1–3 instructions per layer instead of a barrier + refetch.

### Why Faster
- **Barrier elimination:** 64→32→16→8→4→2→1 tree needs 6 barriers + 6 syncs. Subgroups do it in 6 instructions (shuffles within wave).
- **Latency hiding:** Subgroup operations execute in-order within a wavefront. No stall waiting for all 64 threads.
- **Memory locality:** Subgroup data stays in registers; no L1/L2 roundtrip.

**Concrete example (current):**
```wgsl
for (var stride: u32 = WG >> 1u; stride > 0u; stride >>= 1u) {
  if (lid < stride) { sh_pool[lid] += sh_pool[lid + stride]; }
  workgroupBarrier();  // all 64 threads must reach here
}
```
Takes ~6 barriers × (50–200 ns per barrier) = **300–1200 ns**, plus memory round-trips.

**With subgroups (gradient shuffle-down):**
```wgsl
var result = sh_pool[lid];
result = subgroupAdd(result);  // or: result += subgroupShuffle(result, lid ^ 1), etc.
```
Takes ~1–2 instructions = **5–10 ns**, with automatic scalar evolution.

### Memory ↔ Compute Tradeoff
None. Subgroups are **free** in terms of memory. They are hardware primitives (no shared memory, no global memory involved).

### Occupancy Impact
- **No shared memory change:** Reductions still need shared memory targets (to broadcast to other lanes), but the **critical path** (latency from one thread to the next) drops 100×.
- **Wave occupancy:** Subgroup operations are **intra-wave**, so no impact on global occupancy. But they reduce the time-to-completion for reductions, freeing up execution ports.

### Implementation Difficulty & Risk
**Difficulty:** Low. Subgroups are a small API extension. The swap is almost 1-to-1 in code.

**WebGPU support:** Subgroups are **graduated** in WebGPU as of late 2024 (proposal stabilized). Chrome 130+, Firefox 133+. **Must check feature support** at runtime:
```typescript
const hasSubgroups = adapter.features.has("subgroups");
```

**Risk factors:**
1. **Portability:** Safari (iOS) may lag. Conditional compilation needed.
2. **Subgroup size:** Apple Metal uses 32-wide subgroups; Intel/AMD often 64. Code must not assume WG=64 equals subgroup size.

### Validation
1. **Correctness:** Compare reduction results (pool sums, entropy sums) before/after. Should match exactly (same arithmetic).
2. **Performance:** Measure barrier cost directly: insert timers before/after a 6-layer reduction.
3. **Fallback:** If subgroups unavailable, code degrades to current barrier-based reduction (feature-flag).

### Expected ROI at D=8, B=256
- **Time saved per reduction:** ~300 ns → ~10 ns = 290 ns saved. At ~30 barriers in `ppo_step` × 22 steps × 3 epochs × 256 boards = **~500M reductions**. Total: **~150 ms saved per training run.**
- **Per-epoch:** ~5–25 ms.
- **Overall training speedup:** 5–8%.
- **Feasibility:** High (low risk, high confidence).
- **Verdict:** **High priority.** Pure win if subgroups are available; zero-cost on unsupported platforms (fallback to barriers). **Recommend combining with Approach 2.**

---

## Approach 5: FUSION / BATCHING Restructure (Serial → Parallel Patches)

### Core Idea
**Eliminate the serial patch loop** (line 604: `for patch_idx = 0..63`). Instead, **process ALL patches in parallel** by:
1. Assigning threads to (patch, channel) pairs or (cell, output) pairs.
2. Each thread accumulates its gradients locally (small arrays, shared-memory-backed).
3. Reduce across threads via barriers (fewer, but larger).

This is a generalization of Approach 2, but with a focus on **eliminating nested loops entirely**.

### Why Faster
- **Instruction-level parallelism:** 64 threads on 64 patches (or 64 channels) = 100% utilization.
- **Loop elimination:** Saves branch mispredictions (loops are bad on GPUs).
- **Cache reuse:** All threads pull from the same activations (`sh_a`, `sh_bar_a1`), so L1/L2 cache efficiency improves.
- **Nested-barrier reduction:** Instead of barriers at lines 645, 667 inside the patch loop (applied 64 times), consolidate to **one barrier per gradient type** (~5 total).

### Memory ↔ Compute Tradeoff

**Current cost of serial patch loop:**
```
per-thread private arrays: 51 KB × 64 = 3.3 MB
cost per board: 3.3 MB / 256 boards = 12.8 KB per board on average
```

**Restructured (parallel patches):**
```
per-thread shared gradient sink: [dWf: 128, dE_cell: 32, ...] = 160 f32 = 640 bytes
but allocated ONCE across patches (not per thread): 640 bytes
cost per board: 640 bytes  ← dramatic shrinkage
```

Wait — this is the atomic approach (Approach 3) in disguise. The key difference: **Approach 5 uses barriers instead of atomics** to synchronize patch contributions.

### Occupancy Impact
- **Shared memory:** Drops from 51 KB (private spill) to ~1 KB (shared atomic sink).
- **Occupancy:** Can increase from 11% to 60–100% if private memory was the bottleneck.

### Implementation Difficulty & Risk
**Difficulty:** High. Requires a complete rewrite of the gradient accumulation loop.

**Risk factors:**
1. **Thread-to-patch assignment:** Must ensure every (patch, gradient element) pair is assigned to exactly one thread, with no gaps or conflicts.
2. **Shared memory layout:** dWf, dE_cell, etc. must be carefully indexed to avoid bank conflicts.
3. **Race conditions:** Without atomics, need to ensure one thread per (patch, gradient) pair. Hard to guarantee with 64 threads and 64 patches (many-to-many mapping).

**Mitigation:** Use a **staged reduction**:
1. **Stage 1 (patches in parallel):** Threads 0–63 each handle patch 0–63, accumulating into private `l_dWf`, etc. (as today).
2. **Stage 2 (reduction):** All threads reduce their private arrays via tree-reduce, writing to shared `sh_dWf`.
3. **Stage 3 (global):** One thread applies Adam to dense_w.

This is a **incremental parallel refactor**—safer than a complete rewrite.

### Validation
1. **Identical private arrays:** Check that each thread's `l_dWf` before reduction matches the serial version.
2. **Tree-reduce correctness:** Compare reduced values with serial sum.
3. **Adam updates:** Check final weight changes match serial.

### Expected ROI at D=8, B=256
- **Time saved:** If occupancy improves 11% → 80%, latency can drop 50–60%. But the patch loop overhead is only ~200 ms / 503 ms = 40% of backward. So ceiling is ~200 ms saved. **Realistic: 100–150 ms.**
- **Overall training speedup:** 7–10%.
- **Feasibility:** Medium (significant refactor, but incremental staging reduces risk).
- **Verdict:** **High priority, but high risk.** Combine with Approach 2 for maximum impact. **Recommend as the "second" optimization after Approach 4 (subgroups).**

---

## Approach 6: PRECISION Optimizations (fp16 MACs, int8 Dot Products)

### Core Idea
1. **fp16 accumulation for conv MACs:** Conv1 and Conv2 are dominated by dot products (9×D inputs). Accumulate results in fp16, then round to f32 for downstream use.
2. **int8 dot-products (inference path only):** If applicable, quantize activations to int8 and use `dot4I8Packed` (WebGPU packed integer ops) for the initial MACs.

### Why Faster
- **Arithmetic throughput:** FP16 arithmetic is 2–4× faster than f32 on most GPUs (ALU pipelining).
- **Bandwidth:** fp16 is 2× smaller, reducing L1/L2 pressure.
- **Dot products:** int8 + packed ops (4 values per 32-bit word) reduce register pressure and loop iterations.

**Note:** This applies to **forward pass**; backward pass is usually unaffected (backward is still f32). However, if forward is recomputed during backward (Approach 1 not taken), then fp16 speedups apply.

### Memory ↔ Compute Tradeoff

**fp16 convolutions:**
```
current: 9D × D = 72D MACs per (patch, output) = 5.8k FLOPs/patch @ D=8
each FLOP is f32 → ~1.2ns on Apple M4

fp16 variant: same ops, but arithmetic at 2–3× speed
estimated speedup: 2–3× on conv = 3–5 ms per epoch (~5% of backward)

cost: precision loss. fp16 has 2-3 fewer decimal places than f32.
⇒ conv output ranges from 0–1 (ReLU); quantization noise ~1e-3.
⇒ gradients are noisy, but within acceptable tolerance for SGD.
```

**int8 dot products (forward only, if inference-like path exists):**
```
backward doesn't use int8 (gradients need precision).
forward could use int8 for conv1 if activations fit [0, 255] range.
⇒ requires quantization table + dequant after each layer.
⇒ adds overhead; net win only if dequant cost < arithmetic saving.
⇒ unlikely for this kernel (arithmetic is memory-bound, not compute-bound).
```

### Occupancy Impact
- **No change:** Same number of threads, same control flow.
- **Register usage:** fp16 code uses fewer registers (half the bit-width), but WGSL doesn't expose this. Modern GPUs auto-pack, so minimal impact.

### Implementation Difficulty & Risk
**Difficulty:** Low for fp16 (just change `f32` to `f16` in conv loops). High for int8 (requires quantization infrastructure).

**Risk factors:**
1. **Precision loss:** Conv outputs accumulate error. At D=8, 9 MACs per output = 9× epsilon (rule of thumb). Final output has ~1e-3 absolute error.
2. **Gradient flow:** Noisy activations = noisy gradients = slower convergence. Acceptable for PPO (stochastic anyway), but must verify empirically.
3. **int8 portability:** `dot4I8Packed` is WebGPU native, but quantization must be done on CPU (WebGPU has no built-in quantize).

### Validation
1. **Numerical correctness:** Run forward with f32 vs. fp16 conv; check outputs match to ~1e-3 per-channel.
2. **Gradient matching:** Backward gradients should match to ~1e-2 (one order of magnitude looser than forward).
3. **Training stability:** Run 50 generations with fp16 conv, check loss curve and final Elo. Must not diverge or plateau earlier.
4. **End-to-end test:** Compare fp32-fully vs. fp16-conv final weights (head-to-head).

### Expected ROI at D=8, B=256
- **Time saved (fp16 conv):** 3–5 ms per epoch (~1% of backward). **Marginal.**
- **Time saved (int8 if applicable):** Would be 10–20%, but int8 infrastructure overhead likely negates it.
- **Feasibility:** High (fp16 is a simple type swap).
- **Verdict:** **Low priority.** Precision-related wins are often masked by algorithmic speedups. **Only pursue after Approaches 2, 4, 5 are exhausted, and if backward is still slow.**

---

## Summary: Ranking by ROI × Feasibility

| # | Approach | ROI (ms saved/epoch) | Feasibility | Risk | Effort | **Ranking** |
|---|----------|--:|:---:|:---:|:---:|---|
| 4 | **Subgroups** | 5–25 ms (5–8%) | High | Low | Low | **🥇 TIER 0** |
| 2 | **Parallelization (per-patch)** | 130–200 ms (25–40%) | Medium | Medium | Medium | **🥈 TIER 1** |
| 5 | **Fusion/Batching Restructure** | 100–150 ms (7–10%) | Medium | High | High | **🥉 TIER 1.5** |
| 1 | **Store-Activations** | 5–20 ms (1–2% @ D=8; 5–10% @ D=16) | High | Low | Low | **TIER 2** |
| 6 | **Precision (fp16)** | 3–5 ms (1%) | High | Medium | Low | **TIER 3** |
| 3 | **Atomic Accumulation** | 0–10 ms (0–2%) | Low | High | Medium | **⚠ AVOID** |

### Recommended Execution Order

1. **Start with Approach 4 (Subgroups):** 5–8% speedup, zero risk, low effort. Unblocks further parallelization.
2. **Then Approach 2 (Parallelization):** 25–40% speedup (the big win). Requires careful design but directional alignment with Approach 4.
3. **If still needed, add Approach 5 (Fusion):** Another 7–10%, but very high refactor cost.
4. **Consider Approach 1 (Store) for D≥12:** Recomputation cost scales as D³; storage as D; crossover at ~D=12.
5. **Skip Approach 3 (Atomics):** Modern atomics are not worth the numerical risk and complexity.
6. **Polish with Approach 6 (fp16):** Only if other optimizations have plateaued.

---

## Appendix: Correctness Validation Checklist

For each optimization, verify before committing to production:

- [ ] **Forward pass bit-identical:** New kernel produces same board states (if applicable).
- [ ] **Gradient direction check:** Finite-difference gradient check on a small test case (B=1, 1 step). Numerical gradients should match AD within ~1e-3 (absolute).
- [ ] **Loss convergence:** Train for 20 epochs on D=8/B=256, compare loss curve to baseline. Should not diverge.
- [ ] **Entropy health:** Policy entropy should start at ln(576)≈6.36 and decay to 1–2 by epoch 50. No sudden collapse.
- [ ] **Head-to-head Elo:** After 50 self-play generations, play 60 games optimized vs. baseline. 2σ significance threshold is 55–65% win rate; within that range is a pass.
- [ ] **Batch scaling:** Check that 2× batch size (B=256→512) scales linearly in time, or uncovers a hidden bottleneck.

---

## Future Work: GPU-Specific Tuning

If the above approaches hit a ceiling (say, 15% total speedup), consider:

1. **Async computation:** Process multiple boards in a single wave (requires re-architecting workgroup assignment).
2. **Shared-memory double-buffering:** Prefetch next patch state while reducing current (pipelining).
3. **Warp-level scheduling:** Wrap WGSL code to expose warp granularity (not standard, requires Metal/HIP backend).
4. **Quantization-aware training:** Train with fp16 from the start (not a kernel optimization, but a training regime change).
5. **Mixed-precision backward:** Store activations in fp16, compute backward in f32 (combines Approaches 1 + 6).
