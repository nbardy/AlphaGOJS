# Next-Gen Implementation Plan

Date: 2026-03-04
Status: Completed (Scoped Pass + 3 Runtime Modular Rework)
Owner: Codex

## Scope Lock
Execute only these tasks, complete them fully, and do not expand scope:
1. Final architecture doc present and current.
2. Worker-first runtime wired (`gpu_worker`).
3. Startup capability tier selection wired.
4. Async human-AI inference path wired.
5. Benchmark artifacts:
   - Browser benchmark HTML.
   - Node WebGPU benchmark script.
6. Validation: build + benchmark smoke checks.
7. Implement and wire 3 modular runtime modes end-to-end.

## Task Tracker

- [x] Create `NEXTGEN_FINAL_ARCHITECTURE.md`.
- [x] Implement `gpu_worker` pipeline modules.
- [x] Wire `gpu_worker` option in app pipeline creation.
- [x] Add capability probing and runtime tier planner.
- [x] Auto-select startup pipeline by capability tier with safe fallback.
- [x] Add async action path for human mode.
- [x] Keep async Elo eval opt-in (off by default).
- [x] Add browser WebGPU benchmark HTML.
- [x] Add Node WebGPU benchmark script.
- [x] Add npm script entrypoint for Node benchmark.
- [x] Preserve NEXTGEN docs during `npm run build` output cleaning.
- [x] Run build.
- [x] Run benchmark smoke checks.
- [x] Mark this plan complete.
- [x] Add runtime registry with 3 mode IDs and legacy aliases.
- [x] Wire runtime registry into app bootstrap + pipeline creation.
- [x] Wire runtime dropdown to runtime registry list.
- [x] Add runtime-specific queue/backpressure controls for worker pipeline.
- [x] Validate legacy benchmark compatibility after runtime ID change.

## Delivered Files

### Architecture / planning
1. `NEXTGEN_FINAL_ARCHITECTURE.md`
2. `docs/NEXTGEN_IMPLEMENTATION_PLAN.md`

### Next-gen runtime
1. `src/nextgen/capability_probe.js`
2. `src/nextgen/runtime_planner.js`
3. `src/nextgen/protocol/messages.js`
4. `src/nextgen/workers/gpu_owner.worker.js`
5. `src/nextgen/runtime/gpu_owner_runtime.js`
6. `src/nextgen/gpu_worker_trainer_proxy.js`
7. `src/nextgen/create_gpu_worker_pipeline.js`

### Integration changes
1. `src/app.js`
2. `src/ui.js`
3. `src/orchestration/gpu_orchestrator.js`
4. `src/checkpoint_pool.js`
5. `webpack.prod.js`
6. `package.json`
7. `src/runtime/runtime_registry.js`
8. `src/nextgen/create_gpu_worker_pipeline.js`
9. `src/nextgen/gpu_worker_trainer_proxy.js`
10. `src/nextgen/runtime/gpu_owner_runtime.js`

### Benchmarks
1. `benchmarks/browser_webgpu_benchmark.html`
2. `benchmarks/node_webgpu_benchmark.mjs`

## Validation Log

### Build
Command:
`npm run build`

Result:
- Pass
- Known warnings: bundle size only

### Node benchmark smoke
Command:
`npm run bench:webgpu:node`

Result in this environment:
- Script executed correctly
- Runtime reported no WebGPU support and exited with error message
- This is expected on non-WebGPU Node builds

### Browser benchmark smoke
Command:
`npm run bench:webgpu:browser`

Result:
- Instructions emitted for opening benchmark page in a WebGPU-capable browser

## Notes
1. `gpu_worker` runtime is implemented directly and no longer wraps `GPUOrchestrator`.
2. Remaining throughput hotspot (explicitly out of this locked scope): internal engine readbacks (`dataSync`) in `GPUGameEngine`.

## Benchmark Hardening (Post-Scope Follow-up)
- [x] Added warmup + repeated measured runs with `median` and `p95` summaries.
- [x] Added transformer-like compute benchmark path (`QKV -> scores -> softmax -> context -> out proj`).
- [x] Added explicit end-to-end timing path that includes transfer + compute + readback.
- [x] Added detailed stage timing breakdown (`upload`, `qkv`, `attn_scores`, `softmax`, `context`, `out_proj`, `readback`, `total`).

## Runtime Enablement (Headless/WebGPU)
- [x] Added Node fallback provider path using Dawn `webgpu` package when native `navigator.gpu` is absent.
- [x] Added headless Chromium benchmark runner (`bench:webgpu:headless`) using WebGPU launch flags.
- [x] Verified benchmark commands execute with provider detection and clear runtime reporting.
- [x] Added full-system interface benchmark runner (`bench:system:headless`) for app-level throughput/latency metrics.

## Latest Validation Run (2026-03-03)
1. `npm run bench:webgpu:node -- --warmup=2 --runs=5 --seq=128 --hidden=128`
- Exit code: `0`
- Provider: `node-webgpu`
- Key medians: upload `5.464 GB/s`, dispatch `24124.4 /s`, readback `0.71 ms`, transformer e2e `264599.5 tokens/s`

2. `npm run bench:webgpu:headless -- --warmup=2 --runs=5 --seq=128 --hidden=128`
- Exit code: `0`
- Key medians: upload `6.188 GB/s`, dispatch `23529.4 /s`, readback `1.0 ms`, transformer e2e `256000.0 tokens/s`

3. `npm run build`
- Exit code: `0`
- Status: pass (bundle-size warnings only)

## Runtime Stability Fix (Post Validation)
- [x] Added `gpu_worker` tick backpressure via bounded request batching (`maxTickBatch`) in `GPUWorkerTrainerProxy`.
- [x] Defaulted worker pipeline to `maxTickBatch: 8` to prevent giant single tick jobs and reduce tail-latency spikes.
- [x] Updated system benchmark to report both `inference_busy` and `inference_idle` latency, separating queue-delay from model latency.
- [x] Extended benchmark harness timeout controls (`protocolTimeoutMs`, per-inference timeout) to avoid Puppeteer call-timeout false failures during heavy queue scenarios.
