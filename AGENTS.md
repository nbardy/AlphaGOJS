## Cursor Cloud specific instructions

This is **ml-blog**, a client-side TensorFlow.js self-play ML demo (2D grid game rendered on HTML Canvas). Single `package.json`, no monorepo. No backend, no database, no external services.

### Running the dev server

```
source ~/.nvm/nvm.sh && nvm use 18
NODE_OPTIONS=--openssl-legacy-provider yarn run start:dev
```

The app serves at `http://localhost:8080` with HMR enabled.

### Key caveats

- **Node 18 required.** The project uses Webpack 4 + webpack-cli 2 + Babel 6, which are incompatible with Node 20+.
- **`NODE_OPTIONS=--openssl-legacy-provider`** is required when running webpack commands on Node 18 (MD4 hash function used by webpack 4 is not available in OpenSSL 3).
- **webpack version pinning:** After `yarn install`, run `npm install webpack@4.19.1 --no-save` to downgrade webpack to a version compatible with webpack-cli@2. The `^4.0.0` semver range in `package.json` resolves to 4.47.0 which has an incompatible JSON schema structure that crashes webpack-cli@2.
- **No lockfile:** Both `yarn.lock` and `package-lock.json` are gitignored, so dependency resolution may vary.
- **No tests or linter configured.** There are no test scripts or linting tools in this project.
- **WebGL required for full simulation.** The TensorFlow.js game simulation requires WebGL. In headless/cloud VMs without GPU, TF.js falls back to CPU but hits a dtype mismatch error in the game loop. The app structure (canvas, score text) still renders.
