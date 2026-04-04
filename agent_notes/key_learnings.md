# Key learnings — AlphaPlague RL (dense)

Durable takeaways for **correct learning**, **stable PPO**, and **what actually moved the needle** vs throughput-only work. For Q&A and file map see [THREAD_RECAP.md](./THREAD_RECAP.md).

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

---

## 6. Quick verification when “it stopped learning”

1. Confirm **masks** on rollouts match **empty/legal** semantics for the state encoding you feed the net.  
2. Confirm **PPO** `oldLogProb` and **training** `newLogProb` use the **same** masking convention.  
3. Check **entropy** (collapse vs blow-up); adaptive coeff may need retuning if board size or valid-move rate changes.  
4. If using **league**, confirm **nonzero** `checkpointFraction` and that saves occur (`saveInterval`).  
5. If the worker **feels stuck**, inspect **`queueDepth`** / soft cap before blaming the optimizer.
