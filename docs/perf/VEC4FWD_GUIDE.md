# Optimization Guide: vec4 forward conv (iteration 2)

**Why:** after iteration 1 (cell-parallel backward) the backward is ~30–40% of a step; the
**forward** (run in rollout AND in the ppo replay) is now the larger share. This variant
vectorizes the forward conv MACs with `vec4<f32>` (4 multiply-adds per op). **Expected gain
is modest (~1.2–1.5× on the forward; less on the whole step)** and may be partly swamped by
M4 thermal noise — run multiple back-to-back reps. It is a clean *exact-math* change so
correctness is cleanly checkable.

## Workflow (do NOT edit the baseline)
1. `cp src/fused_ppo.wgsl src/variants/fused_ppo.vec4fwd.wgsl`
2. Apply the edits below to the VARIANT only.
3. Validate B=1 (deterministic) + measure B=256, per "Validation".

## What to vectorize (variant only)
All three forward inner loops sum over the input-channel index `c` in steps of `D`
(D ∈ {4,8,16}, all multiples of 4). `sh_a[... + c]` is contiguous in `c`; the weight
accessors (`c1w`/`c2w`/`fw`) are strided in `c`. Replace each scalar `for c { acc += w*a }`
with a `vec4` accumulation over `c` in steps of 4:

- **`conv2_at_patch`** (the `for (var c...)` accumulating `c2w(ky,kx,c,o) * sh_a[...c]`)
- **conv1** in `forward_pass` (the `for (var c...)` accumulating `c1w(ky,kx,c,o) * sh_a[...c]`)
- **fuse** in `forward_pass` (the `for (var c...)` accumulating `fw(c,o)*decoded[c] + fw(D+c,o)*cell_e(state,c)`)

Pattern (illustrative, for the conv1 inner loop):
```wgsl
var acc4 = vec4<f32>(0.0);
for (var c = 0u; c < D; c += 4u) {
  let a4 = vec4<f32>(sh_a[base+c], sh_a[base+c+1u], sh_a[base+c+2u], sh_a[base+c+3u]);
  let w4 = vec4<f32>(c1w(ky,kx,c,o), c1w(ky,kx,c+1u,o), c1w(ky,kx,c+2u,o), c1w(ky,kx,c+3u,o));
  acc4 += a4 * w4;
}
acc += dot(acc4, vec4<f32>(1.0));   // or acc4.x+acc4.y+acc4.z+acc4.w
```
Keep the surrounding loops (over `o`, `ky`, `kx`, cells) and everything else identical.
Do NOT change the backward, the math semantics, or any indexing — only the c-loop shape.
`D` is always a multiple of 4 here, so no remainder handling is needed.

## Validation
The kernel is Hogwild (nondeterministic at B>1) → validate at **B=1**. vec4 reorders the sum
so expect ~1e-3 (not exact) drift vs baseline.
```bash
# baseline vs variant, B=1, D=8 and D=4
B=1 PPO_EPOCHS=1 ARCH=standard STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
KERNEL=src/variants/fused_ppo.vec4fwd.wgsl B=1 PPO_EPOCHS=1 ARCH=standard STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
# (repeat with ARCH=compact)
```
PASS = entropyFirst/lossLast match baseline within ~1e-3 at D=8 AND D=4, no GPU errors.
Then measure speed at B=256 (PPO_EPOCHS=1, STEPS=8 WARMUP=2), baseline vs variant
back-to-back, **3 reps** (GPU throttles). Report per-rep ratios; the median is the result.
If correctness fails, report the mismatch — do not claim success.
