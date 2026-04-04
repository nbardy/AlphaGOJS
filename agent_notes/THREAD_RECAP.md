# Thread recap — AlphaPlague perf, runtime, and learning

Durable summary of discussions and decisions around GPU worker pipelines, TF.js readbacks, PPO tuning, and model choices. For living detail and issue tracking, see [plans_and_ideas.md](./plans_and_ideas.md). For dev setup and repo defaults, see [AGENTS.md](../AGENTS.md).

---

## 1. Q&A reference

| Topic | Summary |
|-------|---------|
| **Elo / checkpoints** | League-style evaluation uses a **checkpoint pool** (see `CHECKPOINT_POOL_CONFIG` in `src/app.js`): checkpoints saved every **15** generations; a fraction of games (**30%**) run as league matchups against stored opponents. Elo-style curves in the UI reflect that mix of self-play and checkpoint games—not a single isolated rating system. |
| **Pipeline speed: phased vs full GPU** | **`single_gpu_phased`** (autosel for capable tiers): `pauseTicksWhenTraining` + moderate `trainInterval` → **smoother UI**. **`full_gpu_resident`**: higher theoretical **games/s**, but the tick queue can saturate. **Mitigation:** `GPUWorkerTrainerProxy` applies a **soft cap** (`queueSoftCapFraction`, default **0.75** for resident) so the main thread stops enqueueing before `maxQueuedSteps`; `getStats()` adds `queueDepth`, `queueSoftCap`, `tickInFlight`. Phased mode uses fraction **1.0** (cap equals `maxQueuedSteps`). |
| **Spatial vs spatial_lite** | **`spatial_model`**: deeper full 3×3 conv stack and a heavier policy head. **`spatial_lite`** (`src/spatial_lite_model.js`): **2× separable 3×3** blocks (lower MACs vs full conv at same H×W), **1×1 conv with one filter** → **one logit per cell** (row-major index aligned with `getBoardForNN`), **global average pool + small MLP** for value. Default model is **`spatial_lite`** for speed while keeping spatial bias. |
| **TF.js readback** | Hot path cost is **GPU → CPU sync** (`dataSync` / gathers). Mitigations: **`src/engine/gpu_game_engine.js`** partial readbacks; **worker** (`gpu_owner_runtime.js`) and **`GPUOrchestrator`** (`gpu_orchestrator.js`) use the same batched **forward + `multinomial` + `logSoftmax` gather** on GPU-resident obs/mask when enabled (PPO/REINFORCE/PPG), with **CPU extract fallback**. Training still uses full forward+backward in TF.js. |
| **Rollout vs train gradients** | **Rollout (self-play)**: forward pass, **sample** actions, store transitions—**no** backprop through the env step. **Training**: PPO **replays** buffered trajectories; gradients flow through the **policy/value network** on stored `(state, action, logProb, value, …)` with clipped surrogate + value loss + entropy—not through the game simulator. |
| **Board Float32 encoding** | States are **`Float32Array`** of length **`rows * cols`** for `tf.tensor2d` (`flattenStates` in `src/action.js`). Encoding: perspective **own / opp / empty** plus **wall = 0.5** in walls mode — not raw 4-way categorical ints (those would need an embedding or one-hot). |
| **TF dtypes** | Training and layers use **float32** end-to-end. TF.js core does **not** offer full **uint8 / fp16 training** like PyTorch AMP; bool uses **uint8** storage. |
| **Custom WebGPU tradeoffs** | A small **custom WebGPU** forward + on-GPU sampling could cut readback volume (**high ceiling**). **Downsides**: reimplementing pieces TF.js gives for free (autodiff, ops coverage), **training** still on TF.js unless duplicated; **multi-worker** weight sharing is awkward. Hybrid ideas (CPU/WASM sim, GPU policy-only) trade implementation complexity vs readback story. |
| **PPO log-prob alignment + `plague_walls` parity** | **`oldLogProb`** in the buffer uses the same **masked log-sum-exp** story as training **`newLogProb`**: CPU **`logProbMaskedLogits`** (`action.js` / `ppo.selectActions`) and GPU **TF `logSoftmax` on masked logits** (`gpu_owner_runtime.js`). **`plague_walls`**: shared **`plague_walls_layout`**, GPU engine spread/terminals vs CPU, **NN wall = 0.5** in gathered tensors (`gpu_game_engine`, orchestrator/runtime). |

---

## 2. Key insights

- **Throughput vs feel** are not the same objective function: **full GPU resident** can win on paper yet **feel broken** under queue backlog; **phased** pipeline often **feels** more continuous at a modest **trainInterval** (e.g. **30** games) with pauses only around `train()`.
- **Readback** dominates many worker-tick budgets once sim is GPU-resident; shrinking **bytes per sync** and **frequency** matters more than micro-optimizing pure JS game logic.
- **Commit `b94665d`** shortened PPO **wall-clock per `train()`** by moving to **fewer optimizer steps per call** (2×128 vs heavier epoch/minibatch schedules)—**snappier** updates, **weaker** optimization per generation unless compensated elsewhere.
- **Historical scale**: classic docs describe **10×10** boards and **40** parallel games for the main-thread trainer era; current defaults and GPU paths may use **different** parallel counts and intervals—always check `app.js`, `trainer.js`, and runtime registry for the live numbers.

---

## 3. Algorithmic details

- **Masked softmax**: Illegal moves are zeroed before normalization; **numerically stable** softmax over **legal** actions only (`src/action.js`). Prevents mass on invalid cells and stabilizes PPO log-probs.
- **PPO buffer / GAE / sparse reward**: `src/ppo.js` — experience buffer filled during play; **GAE** on **per-player sub-trajectories** after terminals; **terminal / sparse** rewards (win/loss style) rather than dense per-step shaping unless added elsewhere.
- **Per-cell logits**: **spatial_lite** exposes **one logit per board cell**; action dimension equals **`rows * cols`**, consistent with masks and game legal-move sets.
- **Checkpoint league**: Periodic saves plus a **fraction of games** against checkpoint policies (`CHECKPOINT_POOL_CONFIG`) to stress the policy against **frozen** historical weights.

---

## 4. Improvement roadmap and blockers

### Priorities (aligned with `plans_and_ideas.md`)

| Priority | Item |
|----------|------|
| **P0** | **Soft queue cap (landed):** `queueSoftCapFraction` on proxy + stats (`queueDepth`, `queueSoftCap`, …). **Remaining:** re-bench `bench:system:headless` / `bench:loop` on resident; optional stricter policies if needed. |
| **P1** | **Readback audit**: eliminate stray full-board `dataSync` on hot paths; explore smaller downloads (e.g. GPU-side sampling) as a larger follow-on. |
| **P2** | **System bench JSON**: optional capture of `gpuReadbackBytes` and bench fields for regressions. |
| **P3** | **CPU `bench:loop` parity** (`sim_random` / `sim_forward`) for apples-to-apples vs worker (optional). |

### Performance blockers

- **Queue backlog** on full resident (mitigated by **soft cap**; still tune `maxQueuedSteps` / UI tick rate if needed).
- **Phased pause**: intentional **freeze** during `train()`—good for fairness, visible if train is heavy.
- **TF.js**: sync readbacks and **single-threaded** worker execution for TF work limit overlap.

### Convergence blockers

- **Too few PPO steps per generation** (`b94665d` tradeoff) vs desire for strong policy improvement per save.
- **Sparse terminal rewards** and **small board** → high variance unless batch sizes / intervals / entropy schedule are tuned.
- **League / checkpoint mix** changes effective training distribution—useful for robustness but can confuse naive “self-play only” intuitions.

---

## 5. Repo artifacts touched in this workstream

| Area | Paths / notes |
|------|----------------|
| **spatial_lite** | `src/spatial_lite_model.js`, registration in `src/model_registry.js`, default `DEFAULT_MODEL` in `src/app.js`. |
| **GPU batched select / engine** | `src/nextgen/runtime/gpu_owner_runtime.js` — `_selectWithAlgorithmGpuBatched` (PPO/REINFORCE/PPG). `src/engine/gpu_game_engine.js` — batched sim, reduced readbacks. |
| **Runtime defaults & worker pipeline** | `src/runtime/runtime_registry.js` — `single_gpu_phased` vs `full_gpu_resident` vs `cpu_actors_gpu_learner`; `src/app.js` tier → pipeline mapping; `src/nextgen/create_gpu_worker_pipeline.js`, `gpu_worker_trainer_proxy.js`, `gpu_owner_runtime.js`, orchestration in `src/orchestration/gpu_orchestrator.js`. |
| **Training / benchmarks** | `src/trainer.js`, `src/gpu_trainer.js`; `benchmarks/system_interface_benchmark.mjs`, `benchmarks/loop_decomposition_benchmark.mjs`; `package.json` scripts `bench:system:headless`, `bench:loop`. |
| **UI / stats** | `src/ui.js` — sampled board snapshots, entropy/stats wiring. |

---

## 6. Related documents

- **[agent_notes/plans_and_ideas.md](./plans_and_ideas.md)** — runtime table, completed work, known issues, async/threading notes, commit history (`3e832bf`, `b94665d`, etc.), reference file index.
- **[AGENTS.md](../AGENTS.md)** — Node 18, webpack/OpenSSL caveats, default model (`spatial_lite`), PPO, runtime autosel, `trainInterval`, benchmark commands, URL flags (including **`?rows=&cols=&numGames=`** for smaller grids).

---

## 7. URL quick presets

- **Smaller / faster iteration:** `?rows=10&cols=10&numGames=40` (clamps: grid **4–32**, games **4–128**). Combines with `?pipeline=full_gpu_resident` or bench flags as needed.
- **Same defaults via preset:** `?preset=fast` or `?preset=interactive` fills missing `rows`/`cols`/`numGames` with **10/10/40**; any of those query keys still override per param.

---

*Last consolidated as a thread recap artifact; extend `plans_and_ideas.md` §8 changelog when landing major perf or runtime changes.*
