# Exploration log — what we tried (AlphaPlague / AlphaGOJS)

Longer, narrative companion to [key_learnings.md](./key_learnings.md) (dense). Not a changelog; for commit-granular history use `git log`.

---

## Performance and worker pipeline

- **GPU game engine readbacks:** Moved from “download whole board every tick” toward **partial gathers** (`tf.gather` on active rows), **small terminal summaries**, **tensor masked resets** — fewer bytes and syncs per step (`src/engine/gpu_game_engine.js`).
- **UI / worker:** **Sampled board snapshots** and merged partial updates so the main thread does not need full N×N arrays every frame for large N (`ui.js`, worker messages, proxy).
- **Entropy in stats:** Wired **worker → proxy → UI** so displayed entropy matches training reality (`gpu_trainer.js`, `create_gpu_worker_pipeline` option forwarding).
- **Full GPU resident vs phased:** Resident can **starve** when the main thread enqueues faster than the worker drains. **Soft queue cap** (fraction of `maxQueuedSteps`, stronger for resident) reduces “infinite backlog” stalls (`gpu_worker_trainer_proxy.js`).
- **Bench modes:** `bench:loop` with `sim_random` / `sim_forward` / full RL; optional **instrumentation** and **minimal UI** via URL flags only when set — no hot-path cost when absent (`benchmarks/loop_decomposition_benchmark.mjs`, `gpu_owner_runtime.js`).

---

## Policy path on the GPU worker (rollout)

- **Goal:** Avoid **CPU `extractStatesMasksCPU` + re-upload** on every tick when TF.js policy is already on GPU.
- **Approach:** **Gather** state rows on GPU → build **obs/mask** tensors → **batched forward** → **masked logits + multinomial** → read back **actions, log π, value** (and snapshot states where needed). **Fallback** to CPU extract + previous path on failure (`gpu_game_engine.js`, `gpu_owner_runtime.js`).
- **Checkpoint opponents:** **Batched TF** sampling with **actions-only readback**; **CPU fallback** if TF path throws (`checkpoint_pool.js`).

---

## PPO training loop (main thread / worker train)

- **Minibatch efficiency:** Build **`statesFull` + `masksFull` once**; per minibatch **`tf.gather`**; compute **`newLogProb`** via **`logSoftmax(maskedLogits)`** and **one-hot** gather instead of rebuilding large JS temporaries (`src/ppo.js`).
- **Historical note:** Earlier profiles used **more epochs × smaller minibatches** (stronger update per generation, slower `train()`); current defaults favor **fewer steps per `train()`** — see `plans_and_ideas.md` §6 and git history.

---

## Tooling and side experiments

- **Webpack 5** migration: dev server, prod build, **vendor split**, **worker chunk**, **`.wgsl` as raw source** for experiments (`webpack.config.js`, `webpack.prod.js`, `package.json`).
- **WebGPU plague env (experimental):** **`plague_env.wgsl`** (spread + apply + terminals) + CPU reference + **parity / throughput** benches; optional **`?webgpuEnv=1`** worker path vs TF tensor sim (`docs/WEBGPU_PLAGUE_GAME_SPEC.md`).
- **URL ergonomics:** **`preset=fast` / `preset=interactive`** fill missing **`rows` / `cols` / `numGames`** with **10 / 10 / 40** (`src/app.js`).
- **`spatial_lite`:** Default model in registry — separable conv + per-cell logits + value head (`spatial_lite_model.js`).

---

## What we did *not* pursue (yet)

- **Custom WebGPU policy forward** with TF.js-only training (hybrid stack complexity).
- **Orchestrator** parity with full **tensor gather** path (worker path went further first).
- **Full uint8 / fp16 training** in TF.js (platform limits).

---

## Observed outcome at “key_learnings” commit

After the above were in tree together, **training showed convergence** and **league Elo increased** in manual runs — documented as motivation in `key_learnings.md`, not as a reproducible benchmark unless captured in `bench:system:headless` JSON or pinned seeds.

---

- **2026-04-04:** `NODE_OPTIONS=--openssl-legacy-provider npx webpack --config webpack.config.js` OK; `npm run bench:loop` failed (Puppeteer: Chrome not found / sandbox cache).
- **2026-04-04 (follow-up):** **`GPUOrchestrator`** gained worker-style **`_buildActionBatchGpu`** + tensor **`_selectWithAlgorithm`** (`app.js` passes **`algoType`** for minification-safe kind). **`GPUGameEngine.resetSlots`** (`plague_walls`) avoids full-board **`dataSync`** when only some rows reset (sorted indices → **`tf.slice`** keep runs + fresh wall rows + **`tf.concat`**; full-matrix regen when all slots reset). Docs: **`key_learnings`**, **`THREAD_RECAP`**, this line.
