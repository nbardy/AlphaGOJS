# Optimization Guide: cell-parallel CONV2-backward

**Target:** the measured bottleneck — the CONV2/fuse backward "64-patch serial loop" in
`ppo_step` is ~90% of the backward (~2470 ms of ~2740 ms at D=8). It runs **64 patches
serially with only 9 of 64 threads active** (`if (lid < 9u)`).

**Idea:** mirror the *forward's* parallelization — run all 576 cells across all 64 threads
(9 cells/thread), store each patch's conv2-delta to shared memory, then do the two
accumulation phases as **gathers** (no scatter races). Same math, reordered → at D=8 it
must produce the *same* gradients as the baseline (validation below).

## Workflow (do NOT edit `src/fused_ppo.wgsl`)

1. Copy the baseline: `cp src/fused_ppo.wgsl src/variants/fused_ppo.cellparallel.wgsl`
2. Apply the two edits below **to the variant file only**.
3. Validate + measure with the harness: `KERNEL=src/variants/fused_ppo.cellparallel.wgsl bun bench/headless.ts ...` (see Validation).
4. Report numbers. The baseline stays untouched and is the comparison point.

## Edit 1 — shared array for all patches' deltas

In the variant, replace the per-patch delta buffer:
```wgsl
var<workgroup> sh_patch_delta2: array<f32, PATCH_CH>;
```
with an all-patches buffer:
```wgsl
var<workgroup> sh_patch_delta2_all: array<f32, P * P * PATCH_CH>;
```
> Shared-memory note: `P*P*PATCH_CH = 64*9*D` floats = 18.4 KB at D=8, 9.2 KB at D=4, but
> **36.8 KB at D=16 which exceeds the 32 KB workgroup limit** → the variant may fail to
> compile at D=16. That is EXPECTED for this variant; validate at **D=4 and D=8** only. (A
> later variant can tile patches to fit D=16.)

## Edit 2 — replace the serial patch loop with 3 parallel phases

Find this block (the only `for (var patch_idx = 0u; patch_idx < P * P * RUN_CONV2BWD; ...`
loop, ~lines 612–676) and replace the ENTIRE loop (from `for (var patch_idx ...` through its
closing `}` after the bar_a1 `workgroupBarrier();`) with:

```wgsl
    // === CONV2/FUSE BACKWARD — cell-parallel (replaces 64 serial patches @ 9/64 threads) ===
    if (RUN_CONV2BWD == 1u) {
      // Phase 1: all 64 threads, ~9 cells each. Per-cell delta_f + small weight grads
      // (l_dWpi/l_dWv/l_dWf/l_dE_cell accumulate per-thread; the later REDUCE sums threads).
      // Each cell writes its own unique slot of sh_patch_delta2_all → no races.
      for (var cell = lid; cell < N; cell += WG) {
        let y = cell / K; let x = cell % K;
        let py = y / 3u; let px = x / 3u;
        let sub = (y % 3u) * 3u + (x % 3u);
        let patch_idx = py * P + px;
        let state = get_sh_cell_state(y, x);

        var decoded: array<f32, D>;
        for (var c = 0u; c < D; c++) { decoded[c] = conv2_at_patch(i32(py), i32(px), sub * D + c); }

        var af: array<f32, D>;
        for (var o = 0u; o < D; o++) {
          var acc = 0.0;
          for (var c = 0u; c < D; c++) { acc += fw(c, o) * decoded[c] + fw(D + c, o) * cell_e(state, c); }
          af[o] = max(acc, 0.0);
        }

        let delta_pi_i = sh_b[cell];
        let delta_v_N = sh_delta_v / f32(N);
        var delta_f: array<f32, D>;

        for (var o = 0u; o < D; o++) {
          l_dWpi[o] += delta_pi_i * af[o];
          l_dWv[o] += delta_v_N * af[o];
          let bar_af = delta_pi_i * pw(o) + delta_v_N * vw(o);
          delta_f[o] = select(0.0, bar_af, af[o] > 0.0);
          for (var c = 0u; c < D; c++) {
            l_dWf[c * D + o] += delta_f[o] * decoded[c];
            l_dWf[(D + c) * D + o] += delta_f[o] * cell_e(state, c);
            l_dE_cell[state * D + c] += delta_f[o] * fw(D + c, o);
          }
        }

        for (var d = 0u; d < D; d++) {
          var bar_decoded_d = 0.0;
          for (var o = 0u; o < D; o++) { bar_decoded_d += delta_f[o] * fw(d, o); }
          sh_patch_delta2_all[patch_idx * PATCH_CH + sub * D + d] = select(0.0, bar_decoded_d, decoded[d] > 0.0);
        }
      }
      workgroupBarrier();

      // Phase 2: conv2 weight gradient (DW2). Each thread owns DW2_PER_THREAD weights and
      // gathers each weight's gradient over ALL patches (no races; was per-patch scatter).
      for (var i = 0u; i < DW2_PER_THREAD; i++) {
        let w_idx = lid + i * WG;
        if (w_idx < DW2_SIZE) {
          let k = w_idx % PATCH_CH; let c = (w_idx / PATCH_CH) % D;
          let kx = (w_idx / (PATCH_CH * D)) % 3u; let ky = (w_idx / (PATCH_CH * D * 3u)) % 3u;
          var acc = 0.0;
          for (var patch_idx = 0u; patch_idx < P * P; patch_idx++) {
            let py = patch_idx / P; let px = patch_idx % P;
            acc += sh_patch_delta2_all[patch_idx * PATCH_CH + k] * sh_a_at(i32(py) + 2*(i32(ky)-1), i32(px) + 2*(i32(kx)-1), c);
          }
          local_dW2[i] += acc;
        }
      }

      // Phase 3: conv1-output gradient (sh_bar_a1) as a GATHER — one thread per output
      // (patch,channel), summing the 3x3 neighborhood of SOURCE patches. This is the
      // transpose of the original scatter: original did target=(src_py+2(u-1), src_px+2(v-1)),
      // so for target (opy,opx) the source is (opy-2(u-1), opx-2(v-1)).
      for (var out_idx = lid; out_idx < P * P * D; out_idx += WG) {
        let oc = out_idx % D;
        let opx = (out_idx / D) % P;
        let opy = (out_idx / D) / P;
        var acc = 0.0;
        for (var u = 0u; u < 3u; u++) {
          for (var v = 0u; v < 3u; v++) {
            let spy = i32(opy) - 2 * (i32(u) - 1);
            let spx = i32(opx) - 2 * (i32(v) - 1);
            if (spy >= 0 && spy < i32(P) && spx >= 0 && spx < i32(P)) {
              let sp = u32(spy) * P + u32(spx);
              var sum = 0.0;
              for (var k = 0u; k < PATCH_CH; k++) { sum += sh_patch_delta2_all[sp * PATCH_CH + k] * c2w(u, v, oc, k); }
              acc += sum;
            }
          }
        }
        sh_bar_a1[out_idx] += acc;
      }
      workgroupBarrier();
    }
```

Notes for correctness:
- Phase 1's `sh_b[cell]` is the per-cell policy delta (same as the original `sh_b[y*K+x]`).
- `l_dWpi/l_dWv/l_dWf/l_dE_cell/local_dW2` are already declared just above the original loop; keep those declarations.
- The original cleared `sh_bar_a1` to 0 before the loop (`for i < P*P*D: sh_bar_a1[lid+i]=0`); KEEP that clear (Phase 3 uses `+=`).
- Do NOT touch the REDUCE / CONV1BWD / EMBEDBWD phases that follow — they consume `l_*` and `sh_bar_a1` exactly as before.

## Validation (REQUIRED — this is gradient-correctness-critical)

IMPORTANT: the kernel is **Hogwild** — at B=256, the 256 board-workgroups race-update shared
weights with no sync, so loss is ~1% nondeterministic run-to-run *even for an identical
kernel*. Do NOT validate correctness at B=256. **Validate at B=1**, which is a single
workgroup → fully deterministic (verified: baseline B=1 gives entropyFirst=6.355,
lossLast=0.6211 identically across runs). The cell-parallel variant is the same math
reordered, so at B=1 it must match the baseline to ~1e-3 (fp-reorder drift only).

### Step A — correctness at B=1 (deterministic oracle)
```bash
# baseline B=1
B=1 PPO_EPOCHS=1 ARCH=standard STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
# variant B=1 (must match within ~1e-3, no GPU device errors)
KERNEL=src/variants/fused_ppo.cellparallel.wgsl B=1 PPO_EPOCHS=1 ARCH=standard STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
# also at D=4
KERNEL=src/variants/fused_ppo.cellparallel.wgsl B=1 PPO_EPOCHS=1 ARCH=compact STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
B=1 PPO_EPOCHS=1 ARCH=compact STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
```
CORRECTNESS PASS = variant `entropyFirst` and `lossLast` match baseline within ~1e-3 at BOTH
D=8 and D=4, with NO GPU device errors. If they diverge or NaN, the gradient is wrong —
report the mismatch, do not claim success.

### Step B — speed at B=256 (only if Step A passed)
```bash
# baseline then variant, back-to-back (GPU is shared + thermally throttles — never interleave)
PPO_EPOCHS=1 ARCH=standard STEPS=8 WARMUP=2 bun bench/headless.ts | grep -oE '"perStepMsMean":[0-9.]+'
KERNEL=src/variants/fused_ppo.cellparallel.wgsl PPO_EPOCHS=1 ARCH=standard STEPS=8 WARMUP=2 bun bench/headless.ts | grep -oE '"perStepMsMean":[0-9.]+'
```
Report the perStepMsMean ratio (baseline/variant) at D=8 and D=4 = the backward speedup.

D=16 may fail to compile (shared-mem >32KB) — that is expected; note it and move on.
