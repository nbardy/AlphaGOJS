# AlphaPlague — game rules, current code paths, and WebGPU / 2-bit direction

This document is for humans and for models helping reimplement the environment in fast **WebGPU (WGSL)** code. It summarizes **rules**, **where the logic lives today**, **how training drives the sim**, and a **target shape** (packed cell state, native buffers) without prescribing every API detail.

---

## 1. What the game is

**AlphaPlague** is a two-player, simultaneous-move **territory / “plague spread”** game on a rectangular grid (`rows` × `cols`, commonly up to 32×32 in the app). Players take **actions every tick** (in code: both players place each turn, then the board **spreads** once). The game ends when **no empty cells remain**. The winner is whoever controls more **non-wall** cells (P1 vs P2); ties are possible.

There are two registered variants:

| Variant ID        | Walls | Board cell semantics (logic) |
|-------------------|-------|------------------------------|
| `plague_classic`  | No    | `0` empty, `1` P1, `-1` P2   |
| `plague_walls`    | Yes   | above + `2` = immovable **wall** |

Default in the app is **`plague_walls`**.

---

## 2. Rules (canonical reference: JavaScript implementations)

### 2.1 Turn structure (one “tick”)

1. **Moves:** Each player chooses one **empty** cell index (flat `0 … rows*cols-1`). Illegal moves (non-empty, out of range) are rejected; training code is expected to mask only valid empties.
2. **Spread:** `spreadPlague()` runs **once** after both moves.
3. **Terminal:** If **no** cell is empty (`0`), the episode is over.
4. **Outcome:** `getWinner()` compares **counts** of P1 (`1`) vs P2 (`-1`). Walls do not count toward either side.

Human UI may also end early if there are no valid moves; the core trainers use `isGameOver()` (all non-empty).

### 2.2 Spread dynamics (CPU reference)

Only **empty** cells (`0`) can change during spread. Occupied cells (P1, P2, and walls) are copied forward unchanged.

For each empty cell, consider the **four orthogonal neighbors** (up, down, left, right). For **classic**:

- For each neighbor that is **P1 or P2** (`±1`), add `neighbor * U(0,1)` to a running sum (one independent uniform random draw **per neighbor**).
- New value: `clamp(trunc(sum * 2), -1, 1)` — so the cell becomes `-1`, `0`, or `1`.

For **walls**:

- Neighbors that are **walls** (`2`) **do not** contribute to the sum (they block influence for that edge).
- Same update rule for the new cell value.

**Reset:**

- **Classic:** all cells → `0`.
- **Walls:** clear board, generate random wall chains (scaled to area), then clear a **center 3×3** so the middle stays playable.

Full detail is in source (see §3).

### 2.3 Observation for the neural net (`getBoardForNN(player)`)

The policy sees a **length `rows*cols`** float vector:

- **Classic:** own cells `+1`, opponent `-1`, empty `0`.
- **Walls:** same, but **wall → `0.5`** (perspective-independent terrain channel in one flat vector).

Valid-move mask: `1` where cell is empty (`0`), else `0` (`getValidMovesMask`).

---

## 3. Key code today (where to read)

| Role | Path | Notes |
|------|------|--------|
| Game rules (walls) | `src/games/plague_walls.js` | `WALL = 2`, `_generateWalls`, `spreadPlague`, scoring |
| Game rules (no walls) | `src/games/plague_classic.js` | Same interface, no `2` |
| Game factory / UI renderer lookup | `src/games/registry.js`, `src/game.js` | `createGame(id, rows, cols)` |
| CPU self-play loop | `src/trainer.js` | Per-game `Game` instances; `tick()` applies both players, then `spreadPlague`, then `isGameOver` |
| Batched GPU path (TF.js tensors) | `src/engine/gpu_game_engine.js` | Many boards in one tensor; `applyActions`, `spread`, `resolveTerminals`, `resetSlots` |
| GPU tick orchestration | `src/orchestration/gpu_orchestrator.js` | Batches actions, calls engine, league / checkpoints |
| Legacy GPU trainer | `src/gpu_trainer.js` | Alternative batched path using same ideas |

**Interface expectations** for a “game object” used by trainers/UI: `reset`, `getValidMoves` / `getValidMovesMask`, `makeMove(player, index)`, `spreadPlague`, `isGameOver`, `getWinner`, `countCells`, `getBoardForNN`, plus `rows` / `cols` / `size`.

---

## 4. Current approach (architecture, not just rules)

- **Game logic is “pure JS”** in `src/games/*` — no TensorFlow dependency. This is the **semantic ground truth** for rules and wall behavior.
- **Training** runs many games in parallel. CPU pipelines hold an array of game objects. GPU-oriented pipelines use **`GPUGameEngine`**: board state as a **`[numGames, boardSize]`** float tensor, with moves applied via one-hot adds and spread implemented with **conv2d** + random uniforms (see `spread()` in `gpu_game_engine.js`).
- **Neural net** is TensorFlow.js (WebGL / WASM / etc., depending on environment); inference and learning are separate from the grid rules, but the engine must feed **consistent** `state` and `mask` tensors.

**Important for a WebGPU rewrite:** the **batched TF engine is not a byte-for-byte duplicate** of the JS reference:

- **Stochastic spread:** JS uses **four independent** random weights (one per neighbor). The TF `spread()` uses a **single** random factor per cell multiplied by the **sum** of neighbors — a different random process.
- **Walls:** JS uses a discrete **`2`** wall cell and special-cases spread. **`GPUGameEngine` stores only a continuum-style `{−1,0,1}` style occupancy in the main tensor** and does not encode walls the same way as `plague_walls.js`; `gameType` is present on the config but the spread/terminal path in `gpu_game_engine.js` does not implement wall terrain.

For a **new** WebGPU sim, decide explicitly:

1. **Spec target:** match **`plague_walls.js` / `plague_classic.js`** (recommended for policy compatibility), or match the current TF batched behavior (only if you intentionally want training/inference parity with today’s tensor engine).

---

## 5. Dream: WebGPU / WGSL + 2-bit cells + native buffers

### 5.1 Why 2 bits per cell

There are exactly **four logical cell kinds**: **empty**, **P1**, **P2**, **wall**. That fits **2 bits** (4 codes). Packing reduces bandwidth and allows **full-board updates in compute shaders** with minimal storage.

Example encoding (choose one convention and stick to it everywhere):

| Code (2b) | Meaning |
|-----------|---------|
| `0b00` | Empty |
| `0b01` | Player 1 |
| `0b10` | Player 2 |
| `0b11` | Wall |

Pack **16 cells per `u32`** (or 8 cells per `u16` if you prefer half the unpacking work per thread). Unpack in WGSL with shifts/masks, pack again after a tick.

### 5.2 What to put in GPU memory

Typical layout for **massively parallel self-play**:

- **Board buffer:** `numGames × ceil(boardSize * 2 / 32)` words, or a simpler `numGames × boardSize` **byte** buffer if you prioritize clarity over extreme packing first.
- **Side metadata:** per-game `done`, `winner`, `turn`, optional episode id — small **structured buffer** or separate **storage buffer**.
- **RNG:** WebGPU has **no built-in `random()`** in WGSL. Use **PCG / xoshiro** state per game (or per workgroup) in a storage buffer, advanced each spread / tick. Match the **same statistical rule** as your chosen spec (per-neighbor uniforms vs single draw — see §4).

### 5.3 Compute pipeline shape

1. **Apply moves:** one compute pass (or merge with spread) — write P1/P2 into empty cells; validate emptiness atomically or via deterministic slot assignment if each thread owns one game.
2. **Spread:** one pass over empty cells; read 4 neighbors (handle edges); apply wall blocking and RNG; write new cell states.
3. **Terminals:** parallel reduction or second pass — if any empty remains, continue; else set `done` and compute winner from population counts (walls excluded).

### 5.4 Bridge to the existing NN stack

Today the model consumes **float** observations (`getBoardForNN`). A WebGPU sim can either:

- **Unpack 2-bit → float** in a small shader or JS when building inference tensors, or
- **Train a model** that reads a compact packed buffer (harder with current TF.js, easier if inference also moves to WebGPU / custom WASM).

The **mask** remains: playable = empty cells in the packed representation.

### 5.5 Suggested migration order

1. Freeze a **single spec** (prefer JS `plague_walls.js` semantics).
2. Implement **WGSL spread + move + terminal** with **tests** against the JS engine on small boards (property or golden seeds).
3. Swap the batched engine behind the same **orchestrator** interface (`applyActions` / `spread` / `resolveTerminals` / `resetSlots` / state export for inference).
4. Optimize packing (2-bit → `u32` lanes) once correctness matches.

---

## 6. Summary

- **Rules:** simultaneous placement on empty cells, then **one stochastic spread** step; game ends when no empties; score by P1 vs P2 cells; **walls** block spread and do not score.
- **Truth in repo:** `src/games/plague_*.js`.
- **Current scale path:** `GPUGameEngine` + `gpu_orchestrator.js` using TF.js tensors — fast for many games, but **not identical** to JS spread RNG and **does not fully mirror wall terrain** in the tensor engine.
- **Future:** WGSL compute, **2-bit packed grids**, explicit RNG buffers, and a deliberate choice to match **JS rules** (recommended) for long-term parity between environment, UI, and policy.

---

*Generated for cross-model handoff; update this file if rules or engine parity targets change.*
