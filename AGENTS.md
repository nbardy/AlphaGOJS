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

- **`docs/PLAGUE_GAME_RULES.md`** — Plague **game rules** (walls + classic), spread math, terminal/winner, NN encoding; handoff doc (canonical code: `src/games/plague_walls.js` / `plague_classic.js`).
- `src/game.js` — Pure JS game logic (no TF.js dependency), 10×10 plague territory game
- `src/model.js` — TF.js policy network (256→128→100 dense, REINFORCE training with entropy bonus)
- `src/trainer.js` — Self-play trainer running 40 parallel games with batched inference
- `src/ui.js` — Dark-theme UI with training grid, stats, and human-vs-AI play mode
- `src/app.js` — Entry point (single-model training)
- `src/app_league.js` — **`league.html`** entry: all registered architectures in parallel, unified league Elo (`src/league_pipeline.js`)
- `src/data.js`, `src/nn.js`, `src/draw.js` — Legacy/unused files from original prototype

### Key caveats

- **Node:** Use a current **LTS** (e.g. **18+**). If an older toolchain errors on OpenSSL, try `NODE_OPTIONS=--openssl-legacy-provider` for that command only.
- **No lockfile:** Both `yarn.lock` and `package-lock.json` are gitignored.
- **No tests or linter configured.**
- **TF.js CPU backend:** The cloud VM lacks WebGL. TF.js falls back to CPU automatically (warning in console is harmless). Game logic is pure JS so it works fine; only the NN model uses TF.js.

### Defaults (app bootstrap)

- **Model:** `spatial_lite`. **Algorithm:** PPO.
- **Runtime:** Tiers **A** and **C** (GPU worker) → **`single_gpu_phased`** (smooth; avoids full-GPU queue stalls). Tier **B** / fallback → `cpu_actors_gpu_learner`. Pick **Full GPU resident** in the UI for max throughput. Override with `?pipeline=…`.
- **Training / league (main app):** `trainInterval` **30** games; checkpoint save every **15** gens, **30%** vs-past-self games (`app.js` `CHECKPOINT_POOL_CONFIG`).
- **League page defaults:** higher `trainInterval` (**48** games per architecture before PPO, `LEAGUE_DEFAULT_TRAIN_INTERVAL` in `src/league_pipeline.js`), **38%** checkpoint-style games (`checkpointFraction` **0.38**), tighter replay and staggered multi-model trains in the GPU worker. Per-model PPO batch is still scaled by **1/N** architectures so aggregate samples per burst stay comparable to single-model training.

#### League page (`league.html`)

After **`npm run dev`** or **`npm run build`**, open **`/league.html`** (dev) or **`docs/league.html`** (static). The main **`index.html`** link in the UI points here; league links back to single-model training.

**Grid / preset (same rules as main app):** `rows`, `cols`, `numGames`, `preset=fast|interactive`, `pipeline=…`, and bench flags `benchInstrument`, `benchMinimalUi`, `webgpuEnv` behave like the main app.

**League-only tuning (optional query params):** invalid or absent values fall back to code defaults.

| Param | Range | Effect |
|--------|--------|--------|
| `trainInterval` | **8–200** | Games completed **per architecture** before a PPO `train()` (default **48**). Try **56–72** if you want more rollout before each optimizer step; lower for faster iteration. |
| `checkpointFraction` or `ckptFrac` | **0.05–0.95** | Fraction of new games that use a frozen checkpoint opponent (default **0.38**). Lower (~**0.25**) for more pure self-play; higher for stability vs older strategies. |
| `trainBatchSize` | **64–2048** | Base batch before per-architecture **÷N** scaling in the worker (default **512**). |
| `multiTrainStagger` | **`0` / `false`** | Disables staggered training (all ready architectures train in one job). Default is stagger **on**. |

Examples: `league.html?trainInterval=64&ckptFrac=0.3` — `league.html?preset=fast&trainInterval=40`.

#### URL overrides (grid; main and league)

Optional query params are parsed at startup (invalid values fall back to defaults): `rows` and `cols` clamped to **4–32** (default **20** each), `numGames` clamped to **4–128** (default **80**). They apply to the initial pipeline and UI; `?pipeline=…` and bench flags still merge as before. **`preset=fast`** or **`preset=interactive`** sets **10×10** grid and **40** games for any of `rows` / `cols` / `numGames` **not** present in the query (explicit `rows`, `cols`, or `numGames` always win). Example: `?preset=fast` is equivalent to the fast-iteration triple `?rows=10&cols=10&numGames=40` when those keys are omitted.

### Benchmarks & profiling (optional)

- **Index:** **`docs/BENCHMARKS.md`** — matrix of every bench, what layer it measures (full app vs WGSL kernel vs TF.js backends), and fair-comparison notes.
- **zsh paste:** run **one command per line**. Do not append **`(~…)`** after a command — zsh treats it as a glob qualifier and can print `unknown file attribute: ~`.
- **`npm run bench:all`** / **`npm run bench:all:smoke`** — sequential bundle: `build` (unless `--skip-build`), **`bench:webgpu:spread`**, **`bench:loop`**, **`bench:system:headless`**. Add **`--with-native-webgpu`** for **`bench:webgpu:parity`**. Default **`bench:all`** uses **two** pipelines and **`--inferenceRuns=24`** so the system step does not run 30+ minutes; use **`node benchmarks/run_all_benchmarks.mjs --full-system`** for all three pipelines + heavier inference (slow). See `benchmarks/run_all_benchmarks.mjs`.
- **`npm run build`** then **`npm run bench:system:headless`** — end-to-end app throughput (Puppeteer opens `docs/index.html`). Either install Puppeteer’s Chrome (**`npx puppeteer browsers install chrome`**) **or** point at your own binary: **`CHROME_PATH`** or **`PUPPETEER_EXECUTABLE_PATH`** (see **`getPuppeteerLaunchOptions`** in `benchmarks/puppeteer_bench_common.mjs`). If the system bench “hangs,” it is often still working: reduce **`--inferenceRuns`** or **`--pipelines`** (see header in `benchmarks/system_interface_benchmark.mjs`).
- **`npm run bench:loop`** — GPU worker loop decomposition: `sim_random` (no policy/train) vs `sim_forward` (forward only) vs full RL; prints games/s and optional policy/physics ms per sim tick.
- **`npm run bench:webgpu:spread`** — native **WGSL plague spread** only: spreads/s and cell-updates/s vs CPU reference (needs WebGPU adapter; exits 0 with `skipped` if none). GPU and CPU paths share the same warmup for a fairer ratio. Not directly comparable to `bench:loop` games/s. **`npm run bench:webgpu:parity`** — correctness check vs CPU. **`npm run bench:webgpu:node`** — generic WebGPU microbenches (Dawn Node).
- **URL flags** (dev/bench only; omit in normal use): `?pipeline=single_gpu_phased&benchLoop=sim_random|sim_forward&benchInstrument=1&benchMinimalUi=1&webgpuEnv=1` (WebGPU sim in the GPU worker when supported; falls back to TF `GPUGameEngine`). When absent, there is no extra work on hot paths.
- **Deeper plans / defaults / roadmap:** `agent_notes/README.md`, **`agent_notes/key_learnings.md`** (dense takeaways), `agent_notes/exploration_log.md`, `agent_notes/plans_and_ideas.md`, recap `agent_notes/THREAD_RECAP.md`.
