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

| # | variant | direction | correctness (B=1) | speedup (D=8 / D=4) | status |
|---|---------|-----------|-------------------|---------------------|--------|
| 1 | `fused_ppo.cellparallel.wgsl` | CONV2-backward: 64 serial patches @9/64 → cell-parallel (all 64 threads) + gather DW2/bar_a1 | _pending_ | _pending_ | running |

> Process per the goal: fork baseline → sub-agent applies the guide's precise edit → validate
> B=1 → measure B=256 → record here → pick next direction from the re-measured bottleneck.
> Candidate next directions (re-measure after each to re-rank): register-spill relief for the
> `l_dWf[2*D*D]` private array (D=16), subgroup reduction (~5%, REDUCE phase), fp16-accumulate
> conv MACs (~1%, validate "no NaN / entropy healthy" not exact-match since it's lossy).
