# Plans, ideas, and technical context

Consolidated from perf work: readbacks, UI sampling, entropy wiring, benchmarks, and future directions.

---

## 1. Runtime registry (summary)

Defined in `src/runtime/runtime_registry.js`:

| `id` | Kind | Notes |
|------|------|--------|
| `single_gpu_phased` | `gpu_worker` | **Autosel for tiers A & C**; `pauseTicksWhenTraining` + `trainInterval` 30 — smoother than resident. |
| `full_gpu_resident` | `gpu_worker` | **Manual** max throughput; can stall (`agent_notes` §3). `maxQueuedSteps` 4096. |
| `cpu_actors_gpu_learner` | `cpu` | Tier B / D fallback; main-thread sim. |

Legacy aliases: `cpu`, `gpu`, `gpu_worker` map to the rows above.

---

## 2. Completed / in-tree work

### GPU → CPU readback hot path (`src/engine/gpu_game_engine.js`)

- **Inference:** `tf.gather` + one `dataSync` for **active rows** instead of full board every time.
- **Terminals:** per-game **3 scalars** (empty / P1 / P2 counts) via GPU ops, small sync — not full board.
- **Reset finished games:** tensor mask + multiply — no download → edit → upload.
- **Telemetry:** `beginReadbackFrame` / `consumeReadbackFrame` → `gpuReadbackCalls`, `gpuReadbackBytes` (worker stats).

### UI / worker transfer

- **Sampled board snapshots** after first full sync: `uiSnapshotMaxGames` (default 48), rotating pages; `boardSampleSlots` in messages.
- **Proxy** merges partial updates into a stable `Int8Array` buffer.

### Entropy metrics (worker)

- UI prefers **`stats.entropy`** then `algo.lastEntropy`; worker proxy had no `lastEntropy` — fixed.
- `createGPUWorkerPipeline` must forward **full** `runtimeOptions` into worker init (`Object.assign`) so bench flags and options are not dropped.

### Benchmarks

- **`benchmarks/system_interface_benchmark.mjs`** — `npm run bench:system:headless`; full app, multiple pipelines.
- **`benchmarks/loop_decomposition_benchmark.mjs`** — `npm run bench:loop`; modes via URL:
  - `benchLoop=sim_random` — legal random moves, no train/replay churn.
  - `benchLoop=sim_forward` — real policy forward, no train.
  - (omit) — full RL.
  - `benchInstrument=1` — worker-reported policy vs physics ms per sim tick (+ train wall when train runs).
  - `benchMinimalUi=1` — rare board snapshots (less postMessage noise).

### Instrumentation safety

- Bench flags only via **URL query**; when absent, `_benchInstrument` is false — **no** `performance.now()` splits on the hot path.

### Misc

- `gpu_trainer.js` `getStats()` includes **`entropy`** for shape parity.
- `app.js` warns if `benchLoop` is set on a non-worker trainer.

### Apr 2026 — training hot path, checkpoints, presets, build, WebGPU plague prototype

- **`src/ppo.js`:** Single **`masksFull`** tensor; per-minibatch **`tf.gather`** for states and masks; **`logSoftmax` × one-hot** for new log-probs (fewer JS temporaries); dispose **`masksFull`** with **`statesFull`** after epochs.
- **`src/checkpoint_pool.js`:** Batched TF forward on masked logits, **`multinomial`** sampling, read back **actions only**; **`_selectActionsCpuFallback`** on failure (legacy full sync + masked softmax).
- **`src/engine/gpu_game_engine.js` + `src/nextgen/runtime/gpu_owner_runtime.js`:** **`gatherSlotsTensor`**, **`_buildActionBatchGpu`** (sub-gather after dropping no-legal rows), tensor-first **`_selectWithAlgorithm`** when **`_obsTensor` / `_maskTensor`** present; **`obsTensor.dataSync()`** for snapshot/trajectory rows; fallback to **`extractStatesMasksCPU`** + batched CPU path. Checkpoint P2 slots still use CPU batch build + pool.
- **`src/app.js`:** **`preset=fast`** / **`preset=interactive`** fill only **missing** query keys **`rows` / `cols` / `numGames`** with **10 / 10 / 40** (documented in root **`AGENTS.md`** and **`THREAD_RECAP.md`** §7).
- **Tooling:** **Webpack 5** (`webpack.config.js`, `webpack.prod.js`), **`copy-webpack-plugin`** / **`html-webpack-plugin`**, vendor split + worker chunk outputs under **`dist/`** and **`docs/`** bundles.
- **WebGPU plague spread (experimental):** **`src/engine/webgpu_plague_spread*.js`**, **`src/engine/wgsl/`**, CPU reference + RNG helpers, **`docs/WEBGPU_PLAGUE_GAME_SPEC.md`**, benches **`bench:webgpu:parity`** / **`bench:webgpu:spread`** (`package.json`). Not wired as the main TF.js game engine yet.
- **Models:** **`src/spatial_lite_model.js`** in tree as default spatial-lite architecture (see **`THREAD_RECAP`** spatial vs spatial_lite).

---

## 3. Known issues & planned fixes

### P0 — `full_gpu_resident` queue starvation

- **Symptom:** `queueDepth` pegged at `maxQueuedSteps`, **0 games/s** in system bench, inference timeouts under load.
- **Cause:** Main-thread rAF × `ticksPerFrame` offers work faster than worker `tick(steps)` drains; **no back-pressure**.
- **Update:** **Soft queue cap landed** — `GPUWorkerTrainerProxy` uses `queueSoftCapFraction` (default **0.75** for `full_gpu_resident`, **1.0** for phased); `getStats()` exposes `queueDepth`, `queueSoftCap`, `queueSoftCapFraction`, `tickInFlight`.
- **Plan:**
  1. Instrument proxy: `_queuedSteps`, batch size, optional worker tick duration.
  2. ~~**Back-pressure:** skip or reduce enqueue when queue > threshold (e.g. 75% of max).~~ **Done** (soft cap).
  3. Re-tune defaults after behavior is stable.
  4. Re-run `bench:system:headless` and `bench:loop` on `full_gpu_resident`.

### P1 — Remaining readbacks / TF.js

- Audit any remaining full-board `dataSync` on the hot path.
- Policy: reduce logits readback (GPU-side sampling or smaller outputs) — larger change.

### P2 — System bench JSON

- Optionally record `gpuReadbackBytes` / bench fields for regression baselines.

### P3 — CPU parity for `bench:loop`

- Same `sim_random` / `sim_forward` hooks on `SelfPlayTrainer` for apples-to-apples vs worker (optional).

---

## 4. Bigger “new approach” ideas (not scheduled)

| Direction | Idea | Tradeoff |
|-----------|------|----------|
| **A** | Split sim worker vs inference worker | TF.js weight sharing across workers is painful; SAB + custom infer or duplicate models. |
| **B** | Small **WebGPU** forward + on-GPU sampling | High ceiling, replaces TF `dataSync` hot path; training still TBD. |
| **C** | **CPU/WASM sim**, batched GPU policy only | Simpler env step; may beat GPU sim + readback on some devices. |
| **D** | Scale defaults | Fewer games / smaller board for “interactive” tier. |

---

## 5. Async / threading (reality check)

- **Worker GPU pipeline:** sim + TF run **off the UI thread**; UI stays responsive.
- **Inside the worker:** TF.js + `dataSync` are **synchronous** on that thread; `AsyncJobQueue` only **defers** work to the next macrotask — **not** parallel GPU + sim.
- **True overlap** of train vs sim needs either another thread with a second model copy or a different inference stack.

---

## 6. Historical: when learning felt faster / smoother

Commits (newest context first):

| When | Commit | What changed |
|------|--------|----------------|
| **Convergence fix (correct learning)** | `3e832bf` | Masked softmax + PPO defaults + metrics. PPO was **4 epochs × minibatch 64** ⇒ **16 gradient steps per `train()`** (heavy but aggressive per generation). |
| **Wall-clock PPO cut** | `b94665d` | **2 epochs × 128** ⇒ **4 steps per `train()`** — faster `train()` wall time, **less optimization per generation** (trade “snappy train” vs “stronger update”). |
| **Scale / parallel games** | `4b255a2`, `b8a8ca5` | **80 games**, **trainInterval 30** (after `4b255a2`: batch 512, interval 30 on main-thread trainer era), ImageData grid, no dead-slot delay. |
| **Worker + registry** | `b7a54ef` | GPU **worker** path: **`trainInterval` 90** for phased + CPU runtime, **`pauseTicksWhenTraining`** on phased — **visible freezes** while `train()` runs; fewer gens per wall-clock than interval 30. |
| **Full GPU resident** | same + later tuning | Large queues; can **starve** (`queueDepth` pegged) — feels stuck / janky (`§3` above). |

**Takeaway:** The “aggressive convergent” PPO profile is closer to **`3e832bf`** (before `b94665d`). The “smooth continuous sim” profile is **shorter `trainInterval`** + **phased** or **CPU** without queue starvation — **not** full resident with a huge backlog. Restored defaults (2026): autosel **phased**, **`trainInterval` 30** for phased and CPU.

---

## 7. Reference files

| Area | Path |
|------|------|
| Runtime IDs | `src/runtime/runtime_registry.js` |
| Default pipeline choice | `src/app.js` (`start`, `mapTierToPipelineType`) |
| Tier probe | `src/nextgen/capability_probe.js`, `src/nextgen/runtime_planner.js` |
| Worker runtime | `src/nextgen/runtime/gpu_owner_runtime.js` |
| Worker proxy | `src/nextgen/gpu_worker_trainer_proxy.js` |
| GPU game engine | `src/engine/gpu_game_engine.js` |
| User-facing caveats | `AGENTS.md` |

---

## 8. Changelog (manual)

Add a one-line date note when you land a major perf or runtime change.

- **2026-04:** Readback reductions, sampled UI boards, entropy/stats wiring, `bench:loop`, bench URL flags, worker init `Object.assign` fix, `agent_notes` added.
- **2026-04:** Default runtime back to **single GPU phased** + **`trainInterval` 30** (autosel tier A/C); full GPU resident manual-only; `agent_notes` §6 training history table.
- **2026-04-04:** PPO gather + log-softmax path, checkpoint batched action readback, GPU **`_buildActionBatchGpu`** / tensor policy select with CPU fallback, URL **`preset=fast|interactive`**, **Webpack 5** + refreshed **`dist`/`docs`** bundles, **WebGPU plague spread** prototype + benches + spec, **`spatial_lite_model.js`**, proxy/orchestrator/trainer/registry/UI tweaks (see §2 “Apr 2026” above).
