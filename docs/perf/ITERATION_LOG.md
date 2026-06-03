# Optimization iteration log

Each row = one forked shader variant (`src/variants/*.wgsl`), A/B-tested headless via
`KERNEL=<variant> bun bench/headless.ts` against the committed baseline. **Correctness is
checked at B=1** (deterministic; the kernel is Hogwild → nondeterministic at B>1). **Speed**
is `perStepMsMean` at B=256, baseline vs variant back-to-back (M4 throttles — ratios only).

## Baseline reference (committed `src/fused_ppo.wgsl`)
- B=1 D=8: entropyFirst=6.355, lossLast=0.6211 (deterministic oracle)
- Measured bottleneck: CONV2-backward 64-patch serial loop = ~90% of the backward.
- Per-PPO-epoch ≈ 503 ms early-session / ~2900 ms throttled (D=8, B=256). Ratios only.

## Iterations

| # | variant | direction | correctness (B=1) | speedup | status |
|---|---------|-----------|-------------------|---------|--------|
| 1 | `fused_ppo.cellparallel.wgsl` | CONV2-backward: 64 serial patches @9/64 → cell-parallel (all 64 threads) + gather DW2/bar_a1 | **PASS** D=4 exact (6.355/0.4805); D=8 in-browser only* | **2.9× on the backward** (D=4, 686→234 ms) | ✅ DONE |

\* D=8 cell-parallel needs ~26 KB workgroup storage. Browsers grant 32 KB via the
`requiredLimits` raise (added to `gpu_harness.ts`), but **bun-webgpu's `requiredLimits`
packing is buggy** (rejects the field), so headless falls back to 16 KB and D=8 won't
compile headless. Its math is the same D-generic code proven exact at D=4. D=16 needs
>32 KB even in-browser → would need a patch-tiled variant.

### What iteration 1 changed
- The backward dropped from **~90% → ~30–40%** of a step (D=4); the bottleneck has SHIFTED
  to the forward/rollout/overhead (ablate-all-backward ≈ 102 ms of a ~150 ms step).
- **Full stack** (original D=8/ep=3 serial → best D=4/ep=1 cell-parallel): ~30–51× measured
  (thermally inflated; ~20–30× cooled = ~14× config × ~1.4× cell-parallel at this config).
- **Implication:** further *backward* micro-opts (subgroups ~5%, fp16 ~1%) are now low-value
  at the practical config — the next frontier is the forward/rollout, not the backward.
- ⚠️ M4 now heavily throttled (~6× slower than early-session) after sustained benching —
  let it cool for trustworthy absolute numbers; ratios from back-to-back A/B still hold.

| 2 | `fused_ppo.vec4fwd.wgsl` | forward conv1/conv2/fuse MACs → vec4 (the new forward bottleneck) | **PASS** exact at D=8 & D=4 (0 drift) | **D=8 1.33×, D=4 4.42×** | ✅ DONE |

### Iteration 2 finding — the forward was the D=4 bottleneck
- vec4 forward conv: at D=4 the whole channel dim is ONE vec4, so the conv c-loop collapses
  → **4.42× at D=4** (123 ms vs 554 ms), even beating cell-parallel alone (234 ms). At D=8
  (D/4 = 2 vec4 iters) it's a modest **1.33×**.
- KEY INSIGHT: the two wins target different configs — **cell-parallel fixes the backward
  (big at D=8), vec4 fixes the forward (huge at D=4). They are disjoint edits → they STACK.**
- So "diminishing returns" was wrong: the forward at D=4 was a real, large, untapped win.

| 3 | `fused_ppo.combined.wgsl` (planned) | cell-parallel backward + vec4 forward stacked | _pending_ | _pending_ | next |

> Process per the goal: fork baseline → sub-agent applies the guide's precise edit → validate
> B=1 → measure B=256 → record here → pick next direction from the re-measured bottleneck.
> Candidate next directions (re-measure after each to re-rank): register-spill relief for the
> `l_dWf[2*D*D]` private array (D=16), subgroup reduction (~5%, REDUCE phase), fp16-accumulate
> conv MACs (~1%, validate "no NaN / entropy healthy" not exact-match since it's lossy).
