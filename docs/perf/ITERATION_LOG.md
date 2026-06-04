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

| 2 | `fused_ppo.vec4fwd.wgsl` | forward conv1/conv2/fuse MACs → vec4 | **PASS** exact at D=8 & D=4 (0 drift) | D=8 1.33×, D=4 **~1.45×** (cool) | ✅ DONE |
| 3 | `fused_ppo.combined.wgsl` | cell-parallel backward + vec4 forward (stacked, disjoint edits) | **PASS** exact at D=4 (0 drift); D=8 in-browser only | **D=4 2.29×** (cool, order-balanced) | ✅ DONE |

### Iterations 2–3 findings + a measurement correction
- ⚠️ **Correction:** iter-2's "4.42× at D=4" was **thermal-inflated** — measured while the M4
  was heavily throttled, which penalizes the scalar-heavy baseline more than the vec4 variant.
  The rigorous **cool-state, order-balanced** number (iter-3 agent) is **vec4 ~1.45×** at D=4.
  Lesson: thermal control matters — use single-process, order-alternated, cool-regime pairs.
- Trustworthy cool-state D=4 full-step speedups vs baseline: cell-parallel **1.64×** (backward),
  vec4 **1.45×** (forward), **combined 2.29×** — they stack multiplicatively (1.64×1.45≈2.38).
- The two wins are complementary: cell-parallel helps the backward (dominant at D=8), vec4
  helps the forward (dominant at D=4). Combined wins everywhere.
- Full stack vs original default (D=8/ep=3 serial → D=4/ep=1 combined) ≈ 14× config × 2.29× kernel.

| 4 | promote to baseline | graduate vec4-forward into `src/fused_ppo.wgsl` (exact, all-D, no shared-mem cost) | **PASS** baseline B=1 unchanged at all D; build green | ~1.3–1.45× live, all configs | ✅ DONE |

### Where the loop landed
- **Baseline now includes vec4 forward** — a safe, universal ~1.3–1.45× for every config
  (incl. D=16), exact, headless-testable. Re-snapshotted in `bench/baselines/`.
- **`fused_ppo.combined.wgsl` (cell-parallel backward + vec4) = 2.29× at D=4**, validated,
  kept as a variant for a future **browser-gated** baseline promotion (the cell-parallel
  backward needs 32 KB workgroup storage → in-browser only; D=16 needs a patch-tiled version).
- Remaining levers are now genuinely marginal: subgroups (~5% of the now-small backward),
  fp16-accumulate (~1%). Not worth it vs the config + cell-parallel + vec4 stack.
- **Total available stack** vs original default: ~14× (config: D=4 + ppoEpochs=1) × ~2.3×
  (combined kernel) ≈ **~30×**, with the safe vec4 portion already live in the baseline.

| 5 | gated promotion → DEFAULT | combined kernel made the baseline, gated `USE_CELLPAR_BWD` (cell-parallel D≤8, serial D=16) | headless B=1 exact (D=4 0.4805, D=16 0.4985) + **in-browser D=8** | **D=8 default ~4.2×** (162→687 g/s) | ✅ DONE |

### The default is now the combined kernel (browser-validated)
- `src/fused_ppo.wgsl` IS the gated combined kernel. The live app's default (**D=8 single
  mode**) was validated **in a real headed Chrome** (bun-webgpu can't bench D=8): **no GPU
  errors, 686 games/s (~4.2× over the original ~162), Elo climbing 1000→1251, entropy
  decaying ~6.4→1.5, win-rate smooth.** D=16/league uses the serial fallback (unbroken).
- Kernel-only speedup at the *original config* (D=8, ep=3): **~4.2× measured in-browser** —
  this is the apples-to-apples number (cell-parallel dominates since the backward is ~96%
  at ep=3). At ep=1 the kernel win shrinks to ~2.3× because the config already shrank the backward.
- Design note: the gating uses generateKernel const-injection (USE_CELLPAR_BWD/PD2_ALL_FACTOR)
  because WGSL workgroup-array sizes are compile-time — a runtime `if` can't shrink the D=16
  array. The cleaner *single-path* alternative is patch-tiling the cell-parallel to fit 16 KB
  at all D (no gating, headless-testable at D=8 too) — a future refinement.

> Process per the goal: fork baseline → sub-agent applies the guide's precise edit → validate
> B=1 → measure B=256 → record here → pick next direction from the re-measured bottleneck.
> Candidate next directions (re-measure after each to re-rank): register-spill relief for the
> `l_dWf[2*D*D]` private array (D=16), subgroup reduction (~5%, REDUCE phase), fp16-accumulate
> conv MACs (~1%, validate "no NaN / entropy healthy" not exact-match since it's lossy).
