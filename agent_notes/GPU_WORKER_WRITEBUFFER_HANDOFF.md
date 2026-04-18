# Handoff: GPU worker `writeBuffer` / league discrete models (2026-04-18)

**Status:** **Unresolved.** User still sees  
`GPU worker error: Failed to execute 'writeBuffer' on 'GPUQueue': Number of bytes to write is too large`  
after hard refresh, with `[tf] backend: webgpu (worker)` and stack through `gpu_owner_runtime.js` init → `gpu_owner.worker.js`.

This note records **everything tried in-thread**, hypotheses, and **next steps** for the next agent.

---

## Symptoms (console)

- `lockdown-install.js` SES message is normal (dependency).
- `Failed to execute 'writeBuffer' on 'GPUQueue': Number of bytes to write is too large`
- Stack: `gpu_worker_trainer_proxy.js` → worker `init` → `gpu_owner_runtime.js` (~289) → `READY` path after async init.
- `favicon.ico` 404 is unrelated.

---

## Issue A — League + discrete observation models (fixed in code)

**Error:** `Discrete-observation models (patch3_discrete) are not supported in league / multi-model mode.`  
**Cause:** `GPUOwnerRuntime` throws if any model has `expectsDiscreteInput` when `multiModel` is true (`src/nextgen/runtime/gpu_owner_runtime.js`). League builds `modelTypes` from **all** registry entries.

**What we did:**

- `src/model_registry.js` — `leagueMultiModel: false` on `patch3_discrete` and `patch3_token`.
- `export function listLeagueModelTypes()` — excludes those entries.
- `src/league_pipeline.js` — uses `listLeagueModelTypes()` instead of `listModelTypes()`.
- `src/app_league.js` — passes `listLeagueModelTypes` to the UI config.

**Status:** Should stop league from pulling discrete archs into the worker; **orthogonal** to the `writeBuffer` error on single-model `index.html`.

---

## Issue B — `writeBuffer` “too large” (still failing)

### Hypotheses (not all mutually exclusive)

1. **Per-call upload cap** — Some WebGPU stacks (often cited: strict WebKit/Safari) reject a **single** `queue.writeBuffer` over N bytes even when the destination `GPUBuffer` is valid.
2. **Uniform vs storage** — TF.js `makeUniforms` builds a **UNIFORM** buffer; default device **`maxUniformBufferBindingSize`** is often **64 KiB**. If TF builds a larger uniform block **without** requesting a higher adapter limit, validation may fail with a **misleading** “bytes too large” style message.
3. **Partial `requiredLimits`** — TF `base.js` requests several adapter limits but historically omitted `maxUniformBufferBindingSize`; requesting only **some** limits can leave others at **minimum** spec defaults.
4. **Wrong bundled file** — Multiple copies of TF WebGPU exist (`backend_webgpu.js`, `tf-backend-webgpu.js` bundle, `.fesm.js`, `.min.js`). Patches might not hit the path webpack actually executes in the worker chunk.
5. **Not TF at all** — Could be `WebGPUGameEngine.seedInitialBoardsIfNeeded` or another `writeBuffer` on the **shared** `GPUDevice` (less likely on default path without `webgpuEnv=1`).

### What we implemented

| Change | Location | Intent |
|--------|----------|--------|
| Chunked tensor upload + chunked uniform upload | `patches/@tensorflow+tfjs-backend-webgpu+4.22.0.patch` → `dist/backend_webgpu.js` | Replace one-shot `writeBuffer` in `uploadToGPU` and `makeUniforms`; helpers `queueWriteBufferChunked`, `queueWriteArrayBufferChunked`. |
| Chunk size sweep | Same patch (evolved) | Started **4 MiB** → **64 KiB** per chunk when 4 MiB still failed. |
| `maxUniformBufferBindingSize` | Patched `dist/base.js` + `dist/tf-backend-webgpu.node.js` | `requiredLimits['maxUniformBufferBindingSize'] = adapterLimits.maxUniformBufferBindingSize`. |
| Plague engine seed upload | `src/engine/webgpu_queue_write_chunked.js` + `webgpu_plague_game_engine.js` | Chunk initial board `writeBuffer` to `bufA`. |
| Global queue shim | `src/webgpu_queue_write_shim.js` imported **first** in `src/tf_backend_bootstrap.js` | Wrap `GPUQueue.prototype.writeBuffer` to split **any** large write into **2 KiB** chunks (256-byte–aligned step size). |
| `postinstall` | `patch-package` | Confirmed `npm install` applies `@tensorflow/tfjs-backend-webgpu@4.22.0` patch. |

### What did **not** fix the user-visible error

- Reducing patch chunk size (4 MiB → 64 KiB).
- Adding `maxUniformBufferBindingSize` to device `requiredLimits`.
- Chunking plague `seedInitialBoardsIfNeeded` (user often on default **TF** `GPUGameEngine`, not `WebGPUGameEngine`, unless `webgpuEnv=1`).
- Restarting dev server, `npm install`, hard refresh (user still reproduces).

**Interpretation:** Either (a) the failing `writeBuffer` bypasses our shim (unlikely if shim runs before TF backend import), (b) the failure is **not** `writeBuffer` byte count but another validation surfaced with the same message, (c) **cached/stale bundle** in the browser or a **different entry** (e.g. extension, Service Worker), or (d) **chunk size must go lower** (e.g. 256–512 B) or alignment rules differ (destination offset multiple of 256 — shim uses 2048-byte steps which preserve alignment if base offset is aligned).

---

## Files to read first (next agent)

- `src/tf_backend_bootstrap.js` — import order: shim → `tf` → `@tensorflow/tfjs-backend-webgpu`.
- `src/webgpu_queue_write_shim.js` — global `GPUQueue.prototype.writeBuffer` wrapper.
- `patches/@tensorflow+tfjs-backend-webgpu+4.22.0.patch` — full diff.
- `node_modules/@tensorflow/tfjs-backend-webgpu/dist/base.js` — `requiredLimits` block.
- `node_modules/@tensorflow/tfjs-backend-webgpu/dist/backend_webgpu.js` — `uploadToGPU`, `makeUniforms`.
- `src/nextgen/runtime/gpu_owner_runtime.js` — init order: `ensureBestTfBackendOnce`, model create, engine, `seedInitialBoardsIfNeeded`.

---

## Suggested next steps (todo)

1. **Confirm shim runs in worker** — Log once from `webgpu_queue_write_shim.js` and from first `orig.writeBuffer` path; ensure worker bundle includes `tf_backend_bootstrap` and shim import (no duplicate unshimmed TF path).
2. **Breakpoint / wrap at throw** — In DevTools, break on `GPUQueue.prototype.writeBuffer` and log `destination.size`, `destinationOffset`, `size` for the failing call.
3. **Try `CHUNK = 256` or `512`** in `webgpu_queue_write_shim.js` (must keep **destinationOffset + pos** aligned if spec requires 256-byte alignment for that buffer class).
4. **Temporary fallback** — In worker only, `tf.setBackend('cpu')` or force pipeline `cpu_actors_gpu_learner` to unblock training while GPU upload path is debugged.
5. **Browser matrix** — Repro on Chrome vs Safari vs Firefox; note exact versions (WebGPU differs).
6. **Search repo for other `writeBuffer`** — `rg writeBuffer src node_modules/@tensorflow` after patch.
7. **Upstream** — Compare with stock TF.js 4.22 `backend_webgpu.js` `uploadToGPU` / device limits; consider opening issue or bumping `@tensorflow/tfjs-backend-webgpu` if fixed upstream.

---

## League-related code references

- `src/league_pipeline.js` — `listLeagueModelTypes()`
- `src/model_registry.js` — `listLeagueModelTypes`, `leagueMultiModel`
- `src/nextgen/runtime/gpu_owner_runtime.js` — discrete check in multi-model init

---

## Session meta

- Dev server: `npm run dev` → `http://localhost:8080/`.
- User confirmed hard refresh; issue **persisted** after restarts and patch reinstall.

This document is the **canonical thread handoff** for the `writeBuffer` investigation as of the conversation that created it.
