# Key learnings — AlphaPlague RL (dense)

Durable takeaways for **correct learning**, **stable PPO**, and **what actually moved the needle** vs throughput-only work. For Q&A and file map see [THREAD_RECAP.md](./THREAD_RECAP.md). For **which bench answers which question**, see [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) and **§8** below.

---

## Snapshot (empirical, commit-time)

At the revision when this file was added, **training was observed to converge** (policy improving in self-play) and **league Elo was climbing** against the checkpoint pool. That is a **runtime observation**, not a guarantee for all devices, seeds, or future code changes. Treat it as a **health signal** that the stack below is internally consistent.

---

## 1. Highest leverage: legal moves and a single policy definition

- **Mask before softmax everywhere** that turns logits into a distribution: illegal cells must contribute **zero** probability. Use the same masking story for **rollout** (`maskedSoftmax` / TF masked logits + `multinomial`) and for **PPO’s `newLogProb`** (masked logits → `logSoftmax` at the taken action). If rollout and training disagree on what “π(a|s)” means, ratios and gradients lie.
- **Stable softmax**: subtract max over **legal** logits before `exp` (see `src/action.js`). Avoid naked softmax over the full board when most cells are illegal.
- **Sparse terminal rewards** are fine if **GAE** (`γ`, `λ`) and **value head** are in the loop; credit assignment is the bottleneck, not “missing dense reward,” for this game shape.

---

## 2. Defaults that match “learns + stays interactive”

- **Model:** `spatial_lite` — spatial inductive bias with **lower cost** than full deep `spatial`; one logit per cell aligned with legal-move masks (`src/spatial_lite_model.js`, `src/model_registry.js`).
- **Algorithm:** **PPO** with **GAE**, clipped surrogate, value loss, entropy bonus; **adaptive entropy** toward a target (reduces early collapse / excessive randomness) — see `src/ppo.js` header and `entropyCoeff` / `targetEntropy`.
- **Update budget:** **2 epochs × minibatch 128** ⇒ **4 gradient steps per `train()`** (tuned for wall-clock vs strength tradeoff; heavier schedules exist in git history).
- **Scale:** `app.js` defaults **20×20**, **80** parallel games, **`trainInterval` 30** on CPU pipeline; GPU worker registry may differ — check `src/runtime/runtime_registry.js` and `src/app.js` for live numbers.
- **League / Elo:** **`CHECKPOINT_POOL_CONFIG`** — periodic saves + **fraction of games vs frozen checkpoints** (`checkpointFraction`, `saveInterval`). Elo only **moves meaningfully** when a slice of play is **not** pure self-play against the current weights.

---

## 3. Worker throughput vs gradient quality

- **More `games/s` ≠ better learning** if the pipeline is wrong (masks, ratios) or the UI/worker **stalls** (queue saturation). **Phased** GPU (`single_gpu_phased`) trades peak throughput for **steady progress** and fewer “stuck” queues vs **full GPU resident**.
- **Readback reduction** (partial board sync, batched policy, action-only checkpoint sampling) mainly cuts **GPU→CPU** cost; it **does not replace** correct masking and PPO math. It can **indirectly** help by allowing more useful samples per wall-clock.
- **Re-forward on replayed states during PPO** is **required** for gradients w.r.t. **current** weights; storing rollout-time gradients is **not** a drop-in substitute (stale θ, huge memory). See THREAD_RECAP “rollout vs train gradients.”

### Mental model: WebGPU / worker — rollouts cheap, gradients expensive

- **Rollouts** (GPU sim + batched policy forward + `multinomial` / readback of actions, log π, V): dominated by **inference** and bounded **GPU→CPU** sync — typically **cheap per env step** relative to training once the hot path is tuned.
- **Gradients** (`ppo.train()`, TF.js **forward + backward** over the replay batch, **Adam**, multiple minibatches): **much more work per sample** than a single rollout forward; in practice often the **wall-clock bottleneck** under fixed model size and update schedule.
- **Design lever:** extra **parallel games** or **higher tick rate** may not shorten “time to good policy” if **`train()`** or **queue/back-pressure** caps learning throughput — profile **train interval**, **batch size**, and **epochs** alongside `games/s`.

---

## 4. Pitfalls we explicitly avoided or fixed along the way

- **Unmasked or half-masked policy** → mass on illegal actions, nonsense **importance ratios**.
- **Resident mode without back-pressure** → `queueDepth` pegged, **0 effective progress**; **soft queue cap** on the proxy mitigates (`src/nextgen/gpu_worker_trainer_proxy.js`).
- **Dropping worker `runtimeOptions`** at init → bench flags / entropy path broken; init must **merge** full options into the worker.
- **Checkpoint inference** doing full **logit `dataSync`** per batch → slow; **batched forward + multinomial + actions-only readback** (with CPU fallback) aligns cost with what league play needs.
- **PPO `oldLogProb` vs train:** Buffer log π must match **`tf.logSoftmax(maskedLogits)`** at the taken action. CPU rollouts use **`logProbMaskedLogits` / `logProbOfAction`** (`src/action.js`, `ppo.js` `selectActions`); GPU rollouts use the same masked normalization via **TF `logSoftmax` + one-hot gather** (`gpu_owner_runtime.js`). Mismatch breaks importance ratios.
- **`plague_walls` GPU–CPU parity:** **`plague_walls_layout`** shares wall RNG/placement; **`gpu_game_engine`** spread/neighbors/terminals track CPU rules; policy tensors use **wall = 0.5** like **`getBoardForNN`** (`gpu_orchestrator.js`, `gpu_owner_runtime.js`).
- **`gpu_orchestrator`:** Happy path **GPU `gatherSlotsTensor`** → obs/mask; **`extractStatesMasksCPU`** only on **fallback** after batched TF select failures (plus compact CPU snapshots for trajectory replay per file comments).
- **TF.js WebGPU CPU→GPU upload (`mappedAtCreation`):** Stock **TF.js 4.22** `uploadToGPU` used **`createBuffer({ mappedAtCreation: true })`** for host uploads. **WebKit/Safari** (and similar) enforce a **small max size** for that path (~tens–low hundreds of KiB). **Larger tensors** (typical **PPO batches**) throw (`size … too large … mappedAtCreation == true`), the **GPU worker errors**, and the UI can **lose canvases / WebGPU instances** (cascade). **Fix:** `patch-package` patch on **`@tensorflow/tfjs-backend-webgpu`** replaces that path with **`queue.writeBuffer`** into a non-mapped buffer — see **`patches/@tensorflow+tfjs-backend-webgpu+4.22.0.patch`** and **§7** below.

---

## 5. Where to change behavior (map)

| Concern | Primary files |
|--------|----------------|
| PPO math, buffer, GAE, train schedule | `src/ppo.js` |
| Masked probs / flatten states | `src/action.js` |
| Model architecture | `src/spatial_lite_model.js`, `src/model_registry.js` |
| App defaults, URL presets, checkpoint pool config | `src/app.js` |
| GPU sim + gather / batched policy path | `src/engine/gpu_game_engine.js`, `src/nextgen/runtime/gpu_owner_runtime.js`, `src/orchestration/gpu_orchestrator.js` (main-thread GPU pipeline: same gather → tensor select + fallbacks) |
| Checkpoint opponent sampling | `src/checkpoint_pool.js` |
| Pipeline presets, queue cap | `src/runtime/runtime_registry.js`, `src/nextgen/gpu_worker_trainer_proxy.js` |
| Benchmarks, JSON/summary artifacts | `benchmarks/*.mjs`, `benchmarks/benchmark_output.mjs`, `benchmarks/run_all_benchmarks.mjs` |

---

## 6. Quick verification when “it stopped learning”

1. Confirm **masks** on rollouts match **empty/legal** semantics for the state encoding you feed the net.  
2. Confirm **PPO** `oldLogProb` and **training** `newLogProb` use the **same** masking convention.  
3. Check **entropy** (collapse vs blow-up); adaptive coeff may need retuning if board size or valid-move rate changes.  
4. If using **league**, confirm **nonzero** `checkpointFraction` and that saves occur (`saveInterval`).  
5. If the worker **feels stuck**, inspect **`queueDepth`** / soft cap before blaming the optimizer.
6. If **throughput regressed**, run **`bench:loop`** + **`bench:system:headless`** and compare **policy_ms/tick** vs **physics_ms/tick** (§8) before assuming the optimizer or PPO schedule.

---

## 7. TF.js WebGPU worker: upload crash, console noise, and memory sleuthing

### Confirmed failure mode (2026-04)

- **Symptom:** Console shows **`Failed to execute 'createBuffer' on 'GPUDevice' … mappedAtCreation == true`** (sizes like **128000** or **~900KiB**), then **`GPU worker error`** / **`GPUOwnerRuntime train error`**, canvases flash or die, **`A valid external Instance reference no longer exists`**.
- **Cause:** Not a generic “OOM” message — it is an **implementation limit on mappable buffers at creation**. Training allocates **larger** CPU-backed tensors → upload hits the limit → **throw** in the worker.
- **Mitigation in-repo:** Patched **`dist/backend_webgpu.js`** `uploadToGPU` to use **`queue.writeBuffer`**, and **`dist/buffer_manager.js`** to **never** pass **`mappedAtCreation`** into **`createBuffer`** (WebKit rejects **STORAGE + mappable-at-create**; error text can mention tiny sizes like **320**). Reapplied via **`npm` `postinstall` → `patch-package`**.

### Related console lines (usually not the crash)

- **`readSync` / Softplus / backend_webgpu “synchronously reading data from GPU to CPU is poor”:** TF.js **warning** about **sync GPU→CPU** readbacks (performance). Distinct from the **`createBuffer`** throw unless you are doing pathological sync in a tight loop.
- **`lockdown-install.js` / SES:** Almost always a **browser extension** or injected script, not this app.
- **`favicon.ico` 404:** Harmless dev-server noise.

### If you suspect a leak after the patch

- Browsers **do not expose** fine-grained **GPU VRAM usage** like desktop tools. **`adapter.limits` / `device.limits`** are caps, not meters.
- **Actionable in JS:** Periodic **`tf.memory()`** in the **worker** (tensor **count** / **numBytes**) — rising without bound suggests **missing `tf.tidy` / disposal** on train or rollout paths. **`performance.memory`** (Chrome-only, when defined) helps **heap** trends.
- **Symptoms of real GPU pressure:** slowdowns, **different** errors (OOM wording), or **context loss** — triangulate with Activity Monitor / Chrome’s task manager, not only in-app logs.

---

## 8. Benchmarks: what the numbers mean (avoid comparing apples to orbitals)

- **Canonical index:** [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) lists npm scripts, entry files, and **fair-comparison** warnings (e.g. **`bench:loop` games/s** vs **`bench:webgpu:spread`**).
- **`ticksPerFrame`:** UI batching — sim ticks per **animation frame**, not “one WGSL dispatch” in the spread microbench.
- **`benchInstrument=1` (URL) / `--instrument` (CLI):** Enables **`GPUOwnerRuntime._tickOne`** timers in `src/nextgen/runtime/gpu_owner_runtime.js`. Exported as **`benchAvgPolicyMsPerSimTick`** and **`benchAvgPhysicsMsPerSimTick`** on stats — these are **means over sim ticks since the last stats flush** (each **`TICK_RESULT`** / **`getStats`** read resets the batch). **Not** a per-tick time series; for histograms or traces you would add new buffers and payload fields.
- **Policy vs “physics” split (worker instrument):** “Policy” ≈ selection / forward path for the tick; “physics” ≈ applies, spread, terminals/resets — coarse wall-clock split, not TF op profiling. **`ensureBoardCacheForPolicy`** is outside the first policy timer block.
- **`bench:loop`:** Modes **`sim_random` / `sim_forward` / full** via URL; CLI **`--instrument` defaults true** (same as loop script). Good for **where time goes inside a sim tick** vs headline games/s.
- **`bench:system:headless`:** **Two phases:** (1) **RL running** with throughput + rAF + **`estimatedTicksPerSec`**; (2) **paused** **`selectActionAsync`** latency (busy vs idle) — that phase is **not** sim-tick throughput. CLI **`--instrument` defaults true** so **`runs[].endStats`** and **`summary`** include worker ms/tick when the pipeline is the GPU worker; CPU-only paths may omit those fields.
- **Artifacts:** `prepareBenchmarkOutput` / `emitBenchmarkReport` write **`.json` + `.summary.md`**; **`--quiet`** silences progress-style logs in system bench too when paired with the shared quiet flag. **`--printJson`** prints the full payload even if **`--quiet`** (human summary still suppressed).
- **Gaps (intentional today):** Parity and patch3 smokes are mostly **stdout-only**; no built-in **per-tick arrays** in benchmark JSON.
