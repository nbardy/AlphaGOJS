# AlphaGOJS v2

Self-play reinforcement learning on a plague spread board game, powered by a single fused WebGPU kernel that does forward pass, backward pass, and AdamW optimizer update in one GPU dispatch.

## Performance

- **525 games/sec** training throughput (B=256, 24x24 board, Apple M4)
- **5-8x faster** than the TF.js v1 implementation
- **Zero ML framework dependencies** — pure WGSL + TypeScript

## Architecture

One monolithic WGSL kernel (`src/fused_ppo.wgsl`) with 4 entry points:

| Entry Point | Purpose |
|---|---|
| `init_boards` | Generate walls, clear board state |
| `rollout_step` | Forward pass + action sampling + game step + plague spread |
| `gae_scan` | Backward GAE advantage computation |
| `ppo_step` | Forward replay + PPO loss + analytical backprop + fused AdamW |

Neural network: patch-based ConvNet with skip connections, 2-bit packed board state (16 cells per u32), fp16 embeddings.

## Requirements

- Browser with WebGPU + `shader-f16` support (Chrome 121+, Edge 121+)
- [Bun](https://bun.sh) (recommended) or Node.js 20+

## Quick Start

```bash
bun install
bun run dev
# Open http://localhost:6974
```

## Project Structure

```
src/
  fused_ppo.wgsl      # The fused kernel (822 lines WGSL)
  gpu_harness.ts       # GPU buffer management + training loop
  worker.ts            # Web Worker wrapper (pause/resume protocol)
  main.ts              # Browser UI + chart rendering
  charts.ts            # Canvas chart renderer
  checkpoint_pool.ts   # Elo-rated checkpoint opponent pool
  idb_storage.ts       # IndexedDB persistence
```

## v1 Archive

The original TF.js implementation is preserved on the `v1-archive` tag.
