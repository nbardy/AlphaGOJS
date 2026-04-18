# Agent notes (AlphaPlague / AlphaGOJS)

Living notes for runtime choice, performance, benchmarks, and roadmap. **Not** user-facing product docs.

| Doc | Contents |
|-----|----------|
| [README.md](./README.md) | This index + **default vs fastest** |
| [key_learnings.md](./key_learnings.md) | **Dense** takeaways: masking, PPO, defaults, Elo/league, pitfalls, file map, **§7 TF.js WebGPU upload patch + memory notes**, **§8 benchmarks / instrumentation semantics** |
| [exploration_log.md](./exploration_log.md) | Narrative of **what we tried** (readbacks, worker path, benches, experiments) |
| [plans_and_ideas.md](./plans_and_ideas.md) | Roadmap, queue starvation, readbacks, bench modes, future architectures |
| [THREAD_RECAP.md](./THREAD_RECAP.md) | Consolidated Q&A: PPO, Elo, runtimes, TF.js readback, artifacts, blockers |
| [HANDOFF_2026-04-04_benchmark_output_smoke.md](./HANDOFF_2026-04-04_benchmark_output_smoke.md) | Handoff note for the terse benchmark-output flow and the `bench:all:smoke` validation run on 2026-04-04 |
| [GPU_WORKER_WRITEBUFFER_HANDOFF.md](./GPU_WORKER_WRITEBUFFER_HANDOFF.md) | **2026-04-18** — GPU worker `writeBuffer` “too large” investigation: league discrete fix, TF patches, queue shim, hypotheses, **still failing**, next-agent todos |
| [../docs/BENCHMARKS.md](../docs/BENCHMARKS.md) | **User-facing** bench matrix (scripts, layers measured, tick vs inference terminology) — complements this folder |

## Fastest approach right now (configured)

**Intended fastest worker preset:** `full_gpu_resident` — large `maxTickBatch` / `maxQueuedSteps`, training does **not** pause ticks (`pauseTicksWhenTraining: false`). See `src/runtime/runtime_registry.js`.

**Caveat:** Resident mode can still **saturate** if the UI offers ticks far faster than the worker drains. The proxy now applies a **soft queue cap** (default **75%** of `maxQueuedSteps` for resident; see `gpu_worker_trainer_proxy.js` and `getStats().queueDepth`). Re-bench after changes; treat resident as **high-throughput when tuned**, not guaranteed fastest out of the box.

**Often healthiest default for GPU tier:** `single_gpu_phased` — smaller batches, **pauses ticks while training**, lower risk of runaway queue depth.

## Is the fastest mode the default?

**No.**

Startup (`src/app.js`):

1. **Fallback** if capability probe fails: `cpu_actors_gpu_learner`.
2. **After probe** (`chooseRuntimeTier` in `src/nextgen/runtime_planner.js`):
   - **Tier A** (worker WebGPU path): `single_gpu_phased`
   - **Tier B** (main-thread WebGPU): `cpu_actors_gpu_learner`
   - **Tier C/D**: CPU-oriented paths

Override anytime: URL `?pipeline=full_gpu_resident` or the in-app pipeline selector (preserved across restart when extras are wired).

## Quick commands

- End-to-end throughput + inference latency sweep: `npm run build && npm run bench:system:headless` (default **`--instrument`** → worker policy/physics ms per sim tick in JSON/summary when on GPU worker pipelines; **`--instrument=false`** to disable).
- Loop decomposition (worker modes): `npm run bench:loop` (instrument **on** by default there too; **`--instrument=false`** if you want a cleaner A/B without timing hooks).
- Full bundle: `npm run bench:all` or shorter **`npm run bench:all:smoke`** (smoke uses fewer pipelines / shorter duration / lower **`--inferenceRuns`** — see `benchmarks/run_all_benchmarks.mjs`).
- Artifact flags (benches using `benchmark_output.mjs`): **`--quiet`** suppresses stdout summary + `saved …` lines; **`--printJson`** still dumps full JSON to stdout even when quiet. Outputs land under `benchmarks/results/…` or **`BENCH_OUTPUT_DIR`**.
- Dev URL flags (bench only): `docs/BENCHMARKS.md`, `plans_and_ideas.md`, root **`AGENTS.md`**
