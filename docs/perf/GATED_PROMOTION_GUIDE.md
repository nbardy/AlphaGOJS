# Gated-promotion guide: make the combined kernel the default

Goal: build `src/variants/fused_ppo.gated.wgsl` = the current baseline (has vec4 forward +
SERIAL backward) with the cell-parallel backward added BEHIND A GATE, so:
- D≤8 → cell-parallel backward (fast; needs 32 KB workgroup storage → browser-granted)
- D=16 → serial backward fallback (fits, league mode keeps working)

`generateKernel` (already updated) flips `USE_CELLPAR_BWD`→false and `PD2_ALL_FACTOR`→1 for D>8.

## Edits (variant only — do NOT touch baseline, kernel_template.ts, bench/*)

1. `cp src/fused_ppo.wgsl src/variants/fused_ppo.gated.wgsl`

2. Add two gating consts next to the `RUN_*` consts (near the top const block):
```wgsl
const USE_CELLPAR_BWD: bool = true; // generateKernel sets false for D>8 (serial fallback)
const PD2_ALL_FACTOR: u32 = 64u;    // = P*P for the cell-parallel scratch; generateKernel sets 1u for D>8
```

3. Add the cell-parallel scratch array next to the existing `sh_patch_delta2` declaration
   (KEEP `sh_patch_delta2` — the serial path uses it):
```wgsl
var<workgroup> sh_patch_delta2_all: array<f32, PD2_ALL_FACTOR * PATCH_CH>;
```

4. In `ppo_step`, find the SERIAL backward block (the `for (var patch_idx = 0u; patch_idx <
   P * P * RUN_CONV2BWD; ...)` loop through its closing `}` after the bar_a1
   `workgroupBarrier();`). Wrap it so the cell-parallel block runs when gated on, else the
   serial block:
```wgsl
    if (USE_CELLPAR_BWD) {
      <<< PASTE the cell-parallel 3-phase block here — copy it VERBATIM from
          src/variants/fused_ppo.combined.wgsl (the `if (RUN_CONV2BWD == 1u) { ... }`
          block: Phase 1 cell loop + Phase 2 DW2 gather + Phase 3 bar_a1 gather).
          It uses sh_patch_delta2_all. >>>
    } else {
      <<< the ORIGINAL serial block (the for-patch_idx loop) exactly as it is in the
          current baseline — it uses sh_patch_delta2. >>>
    }
```
   Tip: `diff src/fused_ppo.wgsl src/variants/fused_ppo.combined.wgsl` shows exactly the
   cell-parallel block (and the vec4 forward, which is ALREADY in the baseline — ignore that
   part). Only the backward block differs in a way you need here.

Do not change the REDUCE/CONV1BWD/EMBEDBWD phases that follow — both paths feed them identically.

## Validation (headless, B=1 deterministic oracle)
```bash
# D=4 → cell-parallel path (fits headless). Must match baseline exactly.
B=1 PPO_EPOCHS=1 ARCH=compact STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
KERNEL=src/variants/fused_ppo.gated.wgsl B=1 PPO_EPOCHS=1 ARCH=compact STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
# D=16 → serial fallback path (fits headless). Must match baseline exactly.
B=1 PPO_EPOCHS=1 ARCH=wide STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
KERNEL=src/variants/fused_ppo.gated.wgsl B=1 PPO_EPOCHS=1 ARCH=wide STEPS=4 WARMUP=0 bun bench/headless.ts | grep -oE '"entropyFirst":[0-9.]+|"lossLast":[-0-9.]+'
```
PASS = gated matches baseline EXACTLY at D=4 (cell-parallel) AND D=16 (serial), no GPU errors.
D=8 will fail headless (cell-parallel needs >16 KB; bun-webgpu cap) — that is EXPECTED and is
validated in-browser separately, not by you. Report the 4 numbers + PASS/FAIL. Sole GPU user,
one bench at a time. If anything diverges, report it — do not claim success.
