# Agent notes (AlphaPlague / AlphaGOJS)

Living notes for runtime choice, performance, benchmarks, and roadmap. **Not** user-facing product docs.

| Doc | Contents |
|-----|----------|
| [README.md](./README.md) | This index + **default vs fastest** |
| [key_learnings.md](./key_learnings.md) | **Dense** takeaways: masking, PPO, defaults, Elo/league, pitfalls, file map |
| [exploration_log.md](./exploration_log.md) | Narrative of **what we tried** (readbacks, worker path, benches, experiments) |
| [plans_and_ideas.md](./plans_and_ideas.md) | Roadmap, queue starvation, readbacks, bench modes, future architectures |
| [THREAD_RECAP.md](./THREAD_RECAP.md) | Consolidated Q&A: PPO, Elo, runtimes, TF.js readback, artifacts, blockers |

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

- End-to-end throughput: `npm run build && npm run bench:system:headless`
- Loop decomposition (worker): `npm run bench:loop`
- Dev URL flags (bench only): see `plans_and_ideas.md` and root `AGENTS.md`
