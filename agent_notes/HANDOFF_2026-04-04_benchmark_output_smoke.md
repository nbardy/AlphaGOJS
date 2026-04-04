# Handoff - benchmark output smoke run

Date: 2026-04-04

## Scope

The benchmark-output cleanup itself is already present in `HEAD` at commit `061c543` (`feat: patch-3 discrete models, packed WebGPU spread, benches`).

That implementation does the following:

- writes bulky benchmark payloads to `benchmarks/results/...`
- keeps terminal output terse
- saves per-step JSON and markdown summaries
- writes a bundle-level `bundle_summary.md` and `bundle_manifest.json`

This note records the validation run and the current state for the next person.

## Validation run

Command:

```sh
npm run bench:all:smoke
```

Run output directory:

```text
benchmarks/results/2026-04-04T10-49-29-492Z-bench-all-smoke
```

Bundle summary:

`benchmarks/results/2026-04-04T10-49-29-492Z-bench-all-smoke/bundle_summary.md`

## Smoke results

- `bench:webgpu:spread`: GPU `19144 spreads/s` vs CPU `16964 spreads/s` (`+12.9%`, ratio `1.1x`)
- `bench:loop`: `sim_random 238.9 games/s`, `sim_forward 31.0 games/s` (`-87.0%` vs random), `full 26.7 games/s` (`-14.0%` vs forward, `-88.8%` vs random)
- `bench:system:headless`: `cpu_actors_gpu_learner 33.68 games/s` vs `single_gpu_phased 20.00 games/s` (`+68.4%`)
- `bench:system:headless` latency: `cpu_actors_gpu_learner` busy p95 `19.815 ms` vs `2389.635 ms`; idle p95 `3.585 ms` vs `1425.430 ms`
- `single_gpu_phased` hit inference timeouts in this smoke run (`5/2` busy/idle); `cpu_actors_gpu_learner` had `0/0`

## Interpretation

- The new summary/log split worked as intended. The terminal stayed readable and the large payloads landed in files.
- The smoke run is still only `runs=1` on the loop/system steps, so the relative deltas are useful for triage but too noisy for strong conclusions.
- On this machine and this run, `cpu_actors_gpu_learner` clearly outperformed `single_gpu_phased` on the system benchmark.

## Files worth reading first

- `benchmarks/results/2026-04-04T10-49-29-492Z-bench-all-smoke/bundle_summary.md`
- `benchmarks/results/2026-04-04T10-49-29-492Z-bench-all-smoke/system_interface_benchmark.json`
- `benchmarks/results/2026-04-04T10-49-29-492Z-bench-all-smoke/loop_decomposition_benchmark.json`
- `benchmarks/results/2026-04-04T10-49-29-492Z-bench-all-smoke/webgpu_plague_spread_throughput.json`

## Suggested next step

If you want less noise, run:

```sh
npm run bench:all
```

Then compare the new run directory against the smoke run above instead of using the smoke numbers alone.

## Worktree note

At the time of writing this note, there are unrelated local changes not included in this handoff commit:

- `package.json`
- `benchmarks/patch3_token_forward_smoke.mjs`
