# Benchmarking AlphaGOJS

This project's training loop runs entirely client-side on WebGPU. To measure and
tune it without a browser, we run the **exact same `src/gpu_harness.ts` code path**
headless under [Bun](https://bun.sh) via [`bun-webgpu`](https://www.npmjs.com/package/bun-webgpu).

> **Headless ≈ browser.** On an Apple M4, the headless harness reports **141 games/sec
> at D=8**, matching the ~140 games/sec the live site shows in Chrome. bun-webgpu and
> browser WebGPU hit the same Metal backend, so these numbers transfer.

## Running the bench

```bash
bun bench/headless.ts                       # default: arch=standard (D=8), B=256, 20 steps
ARCH=compact STEPS=15 bun bench/headless.ts # D=4
ARCH=wide    STEPS=15 bun bench/headless.ts # D=16
B=512        STEPS=20 bun bench/headless.ts # batch-size override (no source edit)
```

Env vars: `ARCH` ∈ `compact`(D=4) | `standard`(D=8) | `wide`(D=16); `B` (batch /
numBoards override); `STEPS` (timed runSteps); `WARMUP` (untimed runSteps first).

### What it measures

One `runStep()` = a full training iteration: rollout (self-play to terminal) + GAE +
`ppoEpochs` PPO/Adam updates + Elo eval + GPU→CPU readbacks. Reported:

- **games/sec** = `B / wallclock_per_runStep` — the same definition the dashboard shows.
- **board-steps/sec** = `games/sec × avg_steps_per_game` — comparable to FUSED_js's
  `steps/sec` unit (avg game ≈ 22 plies at D=8).
- **training sanity**: loss and entropy (first→last), Elo, recorded win-rates.

### How it runs headless

- `bun-webgpu`'s `setupGlobals()` provides `navigator.gpu`, `Float16Array`, and the
  `GPU*` constant namespaces.
- `bench/idb_stub.ts` is a ~40-line in-memory IndexedDB (no `fake-indexeddb` dep) so the
  real `CheckpointPool`/`IDBStorage` persistence path runs unmodified.
- The canvas/worker are simply never used (`onBoard` is left unset → `captureBoard()`
  early-returns).

## ⚠️ Correctness fix that changed the conclusions: D-hardcoded backward

The fused backward/PPO pass in `fused_ppo.wgsl` was pervasively **hardcoded to D=8**
(literal `8u/16u/72u/128/145u/177u/512u/576u` for channel counts) while the forward used
`D`. So **only D=8 ever trained correctly**: D=4 read out of bounds → corrupt gradients
→ entropy collapse + NaN loss (NOT genuine under-capacity); D=16 trained only 8 of 16
channels. League mode and any earlier D≠8 comparison were invalid. Fixed by generalizing
every channel literal to a D-expression (validated: D=8 unchanged, D=4/16 now healthy).
**Any D-comparison numbers from before this fix are garbage — ignore them.**

## Results (Apple M4, B=256, maxSteps=600, ppoEpochs=3) — post-fix

| Arch | D | games/sec | per-runStep | entropy first→last |
|------|---|----------:|------------:|--------------------|
| compact  | 4  | **333–894** | 286–768 ms | 6.36 → 1.3 ✅ healthy |
| standard | 8  | **141–166** | 1,536–1,814 ms | 6.36 → 1.4 |
| wide     | 16 | **26–49**  | 5,250–9,888 ms | 6.36 → 2.0 |

**`D` is the dominant throughput lever** — roughly the O(D²) convolution cost. Entropy now
starts at the true uniform max `ln(576) ≈ 6.36` for all archs and decays sensibly, with
**final entropy rising with D** (more capacity → richer policy). D=4 does NOT collapse;
the earlier collapse was the backward bug above.

### Expressiveness: equal-generations + cross-arch head-to-head (50 gens, post-fix)

Self-play Elo is a separate ladder per arch (not comparable); head-to-head with the final
weights is. After 50 generations:

| Matchup | Result | Notes |
|---|---|---|
| D=4 vs D=8 | D=4 60% (12–8) | small sample — within noise of 50/50 |
| D=4 vs D=16 | D=4 60% (12–8) | " |
| D=8 vs D=16 | D=8 60% (12–8) | " |

**D=4 is not less expressive at this budget** — it converges fastest per generation and is
at least competitive head-to-head (the opposite of the pre-fix buggy result). Caveat: 50
generations is early-training and 20 eval games is a small sample — this measures early
*efficiency*, not the asymptotic capacity ceiling. At equal wall-clock D=4 gets ~5–30× more
generations, so per-unit-compute it dominates decisively.

### Safe-lever tuning stack (no kernel changes) — `bench/sweep.ts`

| Config | games/sec | vs D=8/ep=3 baseline |
|---|---:|---|
| D=8, ppoEpochs=3 (baseline) | 161 | 1× |
| D=8, ppoEpochs=1 | 443 | 2.7× |
| D=4, ppoEpochs=3 | 878 | 5.4× |
| **D=4, ppoEpochs=1** | **2341** | **~14.5×** |

Most of the predicted "10×" is reachable from **config alone** (D=4 + fewer PPO epochs),
now that D=4 actually trains correctly — before reaching for the riskier kernel-numeric
flags. ppoEpochs trades throughput for sample-efficiency; confirm learning quality over a
longer run before locking it in.

### Levers that were tested and did **not** help (full pipeline)

These contradict rollout-only microbenchmark intuition; the full training loop behaves
differently, which is why we measure.

| Change | Result | Why |
|--------|--------|-----|
| **B=256 → 512** | games/sec flat (160.8 → 160.6) | per-runStep scaled exactly 2× (1592 → 3189 ms). The loop is **throughput-bound on per-board work**, not latency-bound on fixed overhead, so a bigger batch costs proportionally more for the same games/sec — and 2× VRAM. **Not adopted.** |
| **Fuse 3 readbacks → 1** | games/sec flat (within noise) | All post-rollout GPU→CPU syncs together are ~1% of a runStep. **Kept anyway** (correctness + houses the entropy-alignment fix + removes 2 sync points), but it is not a speed win. |
| **allGamesDone double-buffer** | not implemented | Measurements above show the in-rollout poll (~3 mapAsyncs) is in the same ~1% bucket. Not the bottleneck; skipped in favor of `D`/kernel work. |

The time is in **per-board GPU compute** (rollout dispatches + PPO epochs), so the real
speed paths are: reduce `D`, reduce `ppoEpochs`, shrink the board `K`, or kernel-level
work reduction (vec4/mat4x4 conv packing — see `../FUSED_js/TODO_ablation.md`).

## Bug found by running it

The headless run immediately surfaced a **real GPU bug that `tsc`/`vite build` could not
catch**: `readEntropy()` copied from byte offset `160+14 = 174`, but
`copyBufferToBuffer` requires 4-byte-aligned offsets. The submit silently failed and the
buffer kept the previous (loss) bytes — so the dashboard would have shown
`entropy === loss`. Fixed by reading the `_pad` f16 through the 4-aligned word at offset
172 (now folded into `readTrainingStats()`). This is the payoff of "run it to confirm."

## The "separate version" (FUSED_js)

FUSED_js has its own bun benches (`bench_peak.ts`, `bench_mixed.ts`, `ablation.ts`) using
the same `bun-webgpu`. **As of this writing they are broken on this machine**: the kernel's
params struct grew but the benches still allocate an 80-byte uniform, so the bind group is
invalid —

```
Binding size (80) of [Buffer] is smaller than the minimum binding size (84).
```

`bench_peak` then "succeeds" with a nonsense `~29M steps/s` (it times empty/no-op
submits). This is consistent with FUSED_js's `HANDOFF.md` noting a mid-refactor kernel.
Fixing it is a one-liner there (bump the params buffer to ≥84 bytes) but out of scope for
this repo. **Until then, the AlphaGOJS headless harness is the only source of
reproducible numbers**, and it matches the live browser.
