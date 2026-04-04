# TODO and final ideal state — benchmarks, sim backends, packed WebGPU env

This document ties together **runtimes**, **benchmarks**, and the **target architecture** (2-bit packed `u32` boards, native WebGPU env, fair comparison vs today’s TF.js `GPUGameEngine`).

---

## 1. Current state (as of this doc)

### 1.1 Production training paths

| Path | Code | Board |
|------|------|--------|
| GPU worker (default / full resident) | `src/nextgen/runtime/gpu_owner_runtime.js` + `GPUGameEngine` | TF.js tensor |
| CPU actors | `SelfPlayTrainer` + `src/games/plague_*.js` | JS / Int8 |
| Main-thread GPU orchestrator | `GPUOrchestrator` + `GPUGameEngine` | TF.js tensor |

**Registry today** (`src/runtime/runtime_registry.js`) only exposes **`gpu_worker`** and **`cpu`**. The **`GPUOrchestrator`** branch in `createPipeline` is **not** reachable from the UI/runtime dropdown unless a new runtime is added.

### 1.2 Native WebGPU plague code (experimental)

| Piece | Role |
|-------|------|
| `src/engine/wgsl/plague_env.wgsl` | WGSL **env**: `spread_pass` + apply + terminals (unpacked `u32` per cell); spread-only benches use the same file |
| `src/engine/webgpu_plague_spread_engine.js` | Ping-pong buffers, uniform tick, host API (`spread_pass` only) |
| `src/engine/webgpu_plague_spread.js` | Webpack bundles `plague_env.wgsl` for spread micro-bench / browser |
| `src/engine/plague_spread_cpu.js` / `plague_spread_rng.js` | CPU reference + shared hash RNG for parity |

**Not integrated** into `gpu_owner_runtime` or `GPUOrchestrator`.

### 1.3 Benchmarks

| Family | Scripts / files | Purpose |
|--------|-----------------|--------|
| Full app | `bench:loop`, `bench:system:headless`, `bench:all` | Games/s, pipelines, sim_random / sim_forward / full RL |
| WGSL plague kernel | `bench:webgpu:spread`, `bench:webgpu:parity` | Spread-only throughput + correctness |
| Generic WebGPU | `bench:webgpu:node`, `bench:webgpu:headless`, `browser_webgpu_benchmark.html` | Driver / API microbenches |
| TF.js backends | `bench.html` / `docs/bench.html` | TF backend comparison, not game throughput |

Shared Puppeteer bits live in **`benchmarks/puppeteer_bench_common.mjs`**.

---

## 2. Final ideal state (goals)

### 2.1 Simulation

- **Canonical rules:** `src/games/plague_walls.js` (and classic variant) remain the **semantic reference**.
- **Packed storage:** **2 bits per cell** inside **`u32`** lanes (16 cells per `u32`), or equivalent dense packing, for **bandwidth and occupancy**.
- **Execution:** **WebGPU compute** passes for:
  - apply moves (P1/P2 onto empty cells, validated)
  - spread (same stochastic semantics as JS reference — per-neighbor RNG)
  - terminal detection + winner counts (walls excluded)
  - slot reset for finished games
- **RNG:** Either **stateless hash** per `(game, tick, cell, neighbor)` (as in current WGSL spread) or **explicit per-game PRNG state** in storage buffers; must match chosen **spec** and be **testable**.

### 2.2 Integration

- A **`WebGPUGameEngine`** (name TBD) implements the **same abstract contract** as `GPUGameEngine` today:
  - `applyActions`, `spread`, `resolveTerminals`, `resetSlots`
  - batched `numGames × rows × cols`
  - observation export for policy (float mask/state compatible with existing models, or a documented migration)
- **Selectable** via runtime registry or feature flag:
  - e.g. `gpu_worker_webgpu_packed` alongside current `single_gpu_phased` / `full_gpu_resident`.

### 2.3 Policy / training

- **Short term:** unpack packed board → existing **Float32** observation path (`getBoardForNN` semantics).
- **Long term (optional):** fused unpack + first linear op, or custom inference — only if profiling proves the bottleneck.

### 2.4 Benchmarks — fair “old vs new”

Run **the same outer loop** for both backends:

| Stage | What to compare |
|-------|-----------------|
| **Kernel** | TF `spread` vs WGSL spread (already partially covered; extend when moves exist) |
| **Env only** | `sim_random` games/s — **TF `GPUGameEngine` vs packed WebGPU env** |
| **Policy** | `sim_forward` games/s + policy ms/tick |
| **Full RL** | `full` games/s + train steps/s + readback bytes |

**Acceptance:** Packed WebGPU path shows **higher `sim_random` games/s** on target hardware, and **does not regress** `sim_forward` / `full` beyond an agreed tolerance (or explains overhead from readback/unpack).

---

## 3. TODO list (ordered)

### Phase A — Hygiene and clarity (done / ongoing)

- [x] Document benchmark families in **`docs/BENCHMARKS.md`**
- [x] Deduplicate Puppeteer Chrome flags + app ready + `docs/index.html` resolution → **`puppeteer_bench_common.mjs`**
- [x] Fair **CPU warmup** in **`bench:webgpu:spread`** (match GPU warmup iterations)
- [x] **`bench:all`** / **`bench:all:smoke`** aggregator (`benchmarks/run_all_benchmarks.mjs`)
- [ ] Keep **`AGENTS.md`** benchmark section in sync with `docs/BENCHMARKS.md` (short index + pointer)

### Phase B — Native env completeness (unpacked `u32` first)

- [x] WGSL + host: **apply moves** pass (`plague_env.wgsl` + `WebGPUGameEngine`)
- [x] WGSL + host: **terminals** (empty / P1 / P2 counts + winner)
- [x] **Reset** finished slots (**CPU layout + `writeBuffer`**, not a WGSL reset pass)
- [ ] **Parity tests** vs full `plague_walls.js` tick (golden boards / end-to-end TF vs WebGPU env)
- [ ] **Throughput bench** for “env tick” = apply + spread + terminal only (no policy, no TF)

### Phase C — Packing (2-bit in `u32`)

- [ ] Pack/unpack utilities in WGSL + JS
- [ ] Switch buffers to packed layout; keep **same host API**
- [ ] Extend parity + throughput benches for packed path
- [ ] Profile **upload/readback** vs TF tensor path

### Phase D — Runtime integration

- [x] **`WebGPUGameEngine`** matches **`GPUGameEngine`** surface used by the worker
- [x] Wired into **`gpu_owner_runtime`** via **`?webgpuEnv=1`** / **`useWebGPUGameEngine`**
- [x] **`bench:loop --webgpuEnv`** for TF vs WebGPU env URL flag (`bench:system` sweep optional)
- [ ] Decide **`GPUOrchestrator`**: **delete**, **register** as experimental runtime, or **document legacy** only

### Phase E — Hardening

- [ ] CI job: `npm run build` + `bench:all:smoke` (no strict native WebGPU parity unless runner has adapter)
- [ ] Optional nightly: `--with-native-webgpu` on hardware agents
- [ ] Version benchmark JSON outputs with **git sha** + **adapter** string for regression tracking

---

### Close-out plan (what’s left vs “north star”)

1. **Optional:** Add **`bench:system`** query flag for **`webgpuEnv=1`** (same as loop) so one script sweeps TF vs WebGPU env across pipelines.
2. **Env-only bench** (Phase B): Node or Puppeteer harness that runs **N ticks** of `WebGPUGameEngine` only (no worker / no TF policy) — isolates sim vs `bench:loop` full app.
3. **Phase C packing** when bandwidth/readback shows up in profiles.
4. **Phase E:** wire **`bench:all:smoke`** in CI; keep **`--with-native-webgpu`** for dedicated GPU runners only.
5. **Aggregator gap:** `bench:all` does not run **`bench:webgpu:node`** or **`bench:webgpu:headless`** — run those manually or extend `run_all_benchmarks.mjs` if you want one command for literally everything.

---

## 4. Non-goals (for this roadmap)

- Rewriting **PPO / training** in raw WGSL (see `ARCHITECTURES.md` Arch 7 — possible but out of scope).
- Promising **bit-exact** parity between **TF `GPUGameEngine` spread** and **JS `plague_walls.js`** without a dedicated “TF-compat” mode (today they differ by design).

---

## 5. Duplicate vs complementary benchmarks

| Pair | Relationship |
|------|----------------|
| `node_webgpu_benchmark` vs `browser_webgpu_benchmark` | **Complementary** — different runtimes/drivers |
| `bench:loop` vs `bench:system` | **Overlapping** but **different** — loop stresses sim modes; system sweeps pipelines + richer metrics |
| `bench:webgpu:spread` vs `bench:loop` | **Not comparable** — kernel vs full app |

---

## 6. One-line “north star”

**Ship a packed WebGPU plague environment behind the same worker interface as `GPUGameEngine`, and prove it in `sim_random` / `sim_forward` / full benchmarks against the current TF tensor path.**
