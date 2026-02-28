## Cursor Cloud specific instructions

**AlphaPlague** — a browser-based self-play RL plague/territory game using TensorFlow.js. Single `package.json`, no monorepo. No backend, no database, no external services.

### Running the dev server

```
source ~/.nvm/nvm.sh && nvm use 18
NODE_OPTIONS=--openssl-legacy-provider yarn run start:dev
```

The app serves at `http://localhost:8080` with HMR enabled.

### Architecture

- `src/game.js` — Pure JS game logic (no TF.js dependency), 10×10 plague territory game
- `src/model.js` — TF.js policy network (256→128→100 dense, REINFORCE training with entropy bonus)
- `src/trainer.js` — Self-play trainer running 40 parallel games with batched inference
- `src/ui.js` — Dark-theme UI with training grid, stats, and human-vs-AI play mode
- `src/app.js` — Entry point
- `src/data.js`, `src/nn.js`, `src/draw.js` — Legacy/unused files from original prototype

### Key caveats

- **Node 18 required.** Webpack 4 + webpack-cli 2 + Babel 6 are incompatible with Node 20+.
- **`NODE_OPTIONS=--openssl-legacy-provider`** is required for webpack on Node 18.
- **webpack version pinning:** After `yarn install`, run `npm install webpack@4.19.1 --no-save` to downgrade webpack to a version compatible with webpack-cli@2. The `^4.0.0` range resolves to 4.47.0 which crashes webpack-cli@2.
- **No lockfile:** Both `yarn.lock` and `package-lock.json` are gitignored.
- **No tests or linter configured.**
- **TF.js CPU backend:** The cloud VM lacks WebGL. TF.js falls back to CPU automatically (warning in console is harmless). Game logic is pure JS so it works fine; only the NN model uses TF.js.

### Production build & GitHub Pages

```
NODE_OPTIONS=--openssl-legacy-provider npx webpack --config webpack.prod.js
```

Outputs minified bundle to `docs/`. The `gh-pages` branch contains the built static files at root and is configured as the GitHub Pages source. Live at: https://nbardy.github.io/AlphaGOJS/

A GitHub Actions workflow (`.github/workflows/deploy.yml`) is set up to auto-rebuild on push to `master`, but requires Actions to be enabled on the repo.
