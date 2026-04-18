# Benchmark index (AlphaPlague)

Use this table to pick the right bench for a question. **Different benches measure different layers** — do not compare unrelated headline numbers (e.g. WGSL spread cells/s vs full-app games/s).

## Terminology

| Term | Meaning |
|------|---------|
| **`ticksPerFrame`** | UI knob: simulation ticks batched per animation frame (default often **20**). Not the same as a single “physics kernel launch” in the WebGPU spread bench. |
| **`benchAvg*MsPerSimTick`** | GPU worker: mean policy vs physics wall time **per simulated tick**, averaged over ticks since the last stats flush (`getStats` / tick result). Not a per-tick time series. |
| **System inference phase** | App **paused**; measures **`selectActionAsync`** RPC latency (busy vs idle queue), not sim-tick throughput. |

## npm scripts and entry points

| Script | File | What it measures |
|--------|------|------------------|
| `npm run bench:all` | `benchmarks/run_all_benchmarks.mjs` | Bundle: build (optional), WebGPU spread, loop decomposition, system headless. Optional `--with-native-webgpu` adds parity. |
| `npm run bench:all:smoke` | same | Shorter `--duration` / `--runs` / `--warmup` and **`--inferenceRuns=12`** (vs **24** in default non-smoke); two pipelines unless `--full-system`. |
| `npm run bench:system:headless` | `benchmarks/system_interface_benchmark.mjs` | Built **`docs/index.html`**: RL throughput windows + inference latency sweeps. Default **`--instrument`** adds worker policy/physics ms/tick into JSON/summary (GPU worker pipelines). |
| `npm run bench:system:algos` | same with `--algos=ppo,reinforce` | Multi-algorithm variant of system bench. |
| `npm run bench:loop` | `benchmarks/loop_decomposition_benchmark.mjs` | `sim_random` / `sim_forward` / full RL on one pipeline; **`--instrument` default on** → policy/physics ms per sim tick. |
| `npm run bench:webgpu:spread` | `benchmarks/webgpu_plague_spread_throughput.mjs` | Node WebGPU WGSL **spread** vs CPU reference; optional `--packed`. |
| `npm run bench:webgpu:parity` | `benchmarks/webgpu_plague_spread_parity.mjs` | Correctness: unpacked spread vs CPU. |
| `npm run bench:webgpu:parity:packed` | `benchmarks/webgpu_plague_spread_parity_packed.mjs` | Correctness: packed spread vs CPU. |
| `npm run bench:webgpu:node` | `benchmarks/node_webgpu_benchmark.mjs` | Generic WebGPU microbenches (Dawn Node). |
| `npm run bench:webgpu:headless` | `benchmarks/headless_browser_webgpu_benchmark.mjs` | Puppeteer + `browser_webgpu_benchmark.html`. |
| `npm run bench:webgpu:browser` | *(echo only)* | Opens `benchmarks/browser_webgpu_benchmark.html` manually. |
| `npm run bench:patch3-smoke` | `benchmarks/patch3_discrete_forward_smoke.mjs` | TF forward / discrete patch; **not** full plague sim. |
| `npm run bench:patch3-token-smoke` | `benchmarks/patch3_token_forward_smoke.mjs` | TF `patch3_token` forward smoke. |
| `npm run bench:matrix:smoke` | `benchmarks/runtime_matrix_smoke.mjs` | Quick system bench matrix (default vs `webgpuEnv=1`). |

## Artifacts

Benchmarks that use **`benchmarks/benchmark_output.mjs`** write **`{benchmarkId}.json`** and **`.summary.md`** under `benchmarks/results/…` (or `BENCH_OUTPUT_DIR`). Flags: **`--quiet`**, **`--printJson`** (`--printJson` still dumps JSON when `--quiet` is set; human summary and `saved …` lines are suppressed).

Parity scripts and patch3 smokes are mostly **stdout-only** unless extended separately.

## Fair comparison

- **`bench:loop` games/s** includes policy + sim + (in full mode) training — **not** comparable to **`bench:webgpu:spread`** (spread kernel only).
- **System** throughput uses the real UI + trainer; **inference** numbers isolate action selection with training paused.

See also **`benchmarks/run_all_benchmarks.mjs`** header and **`AGENTS.md`** (benchmarks section).
