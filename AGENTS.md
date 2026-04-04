## Cursor Cloud specific instructions

**AlphaPlague** — a browser-based self-play RL plague/territory game using TensorFlow.js. Single `package.json`, no monorepo. No backend, no database, no external services.

### Running the dev server

```
source ~/.nvm/nvm.sh && nvm use 18   # optional; use a current LTS Node
npm install
npm run dev
```

Webpack **5** (`webpack.config.js`); production: **`npm run build`**. The dev server defaults to **`http://localhost:8080`** with HMR (see `webpack.config.js` if the port differs).

### Architecture

- `src/game.js` — Pure JS game logic (no TF.js dependency), 10×10 plague territory game
- `src/model.js` — TF.js policy network (256→128→100 dense, REINFORCE training with entropy bonus)
- `src/trainer.js` — Self-play trainer running 40 parallel games with batched inference
- `src/ui.js` — Dark-theme UI with training grid, stats, and human-vs-AI play mode
- `src/app.js` — Entry point
- `src/data.js`, `src/nn.js`, `src/draw.js` — Legacy/unused files from original prototype

### Key caveats

- **Node:** Use a current **LTS** (e.g. **18+**). If an older toolchain errors on OpenSSL, try `NODE_OPTIONS=--openssl-legacy-provider` for that command only.
- **No lockfile:** Both `yarn.lock` and `package-lock.json` are gitignored.
- **No tests or linter configured.**
- **TF.js CPU backend:** The cloud VM lacks WebGL. TF.js falls back to CPU automatically (warning in console is harmless). Game logic is pure JS so it works fine; only the NN model uses TF.js.

### Defaults (app bootstrap)

- **Model:** `spatial_lite`. **Algorithm:** PPO.
- **Runtime:** Tiers **A** and **C** (GPU worker) → **`single_gpu_phased`** (smooth; avoids full-GPU queue stalls). Tier **B** / fallback → `cpu_actors_gpu_learner`. Pick **Full GPU resident** in the UI for max throughput. Override with `?pipeline=…`.
- **Training / league:** `trainInterval` **30** games (phased, CPU, resident); checkpoint save every **15** gens, **30%** league games (`app.js` `CHECKPOINT_POOL_CONFIG`).

#### URL overrides (grid and parallel games)

Optional query params are parsed at startup (invalid values fall back to defaults): `rows` and `cols` clamped to **4–32** (default **20** each), `numGames` clamped to **4–128** (default **80**). They apply to the initial pipeline and UI; `?pipeline=…` and bench flags still merge as before. **`preset=fast`** or **`preset=interactive`** sets **10×10** grid and **40** games for any of `rows` / `cols` / `numGames` **not** present in the query (explicit `rows`, `cols`, or `numGames` always win). Example: `?preset=fast` is equivalent to the fast-iteration triple `?rows=10&cols=10&numGames=40` when those keys are omitted.

### Benchmarks & profiling (optional)

- **`npm run build`** then **`npm run bench:system:headless`** — end-to-end app throughput (Puppeteer opens `docs/index.html`). Install browser: `npx puppeteer browsers install chrome`.
- **`npm run bench:loop`** — GPU worker loop decomposition: `sim_random` (no policy/train) vs `sim_forward` (forward only) vs full RL; prints games/s and optional policy/physics ms per sim tick.
- **`npm run bench:webgpu:spread`** — native **WGSL plague spread** only: spreads/s and cell-updates/s vs CPU reference (needs WebGPU adapter; exits 0 with `skipped` if none). Not directly comparable to `bench:loop` games/s (that includes policy + full sim); use as an upper bound on physics-style compute. **`npm run bench:webgpu:parity`** — correctness check vs CPU. **`npm run bench:webgpu:node`** — generic WebGPU microbenches (Dawn Node).
- **URL flags** (dev/bench only; omit in normal use): `?pipeline=single_gpu_phased&benchLoop=sim_random|sim_forward&benchInstrument=1&benchMinimalUi=1`. When absent, there is no extra work on hot paths.
- **Deeper plans / defaults / roadmap:** `agent_notes/README.md`, **`agent_notes/key_learnings.md`** (dense takeaways), `agent_notes/exploration_log.md`, `agent_notes/plans_and_ideas.md`, recap `agent_notes/THREAD_RECAP.md`.
