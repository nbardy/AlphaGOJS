# AlphaPlague — Architecture Plan

## Phase 1: Fix the RL (no infra changes)

**Goal**: Make the current single-threaded system actually learn well before scaling.

### 1.1 Add Value Head (A2C)

Change network from policy-only to policy+value:

```
Input [100] → Dense(256, relu) → Dense(128, relu) ─┬─ Dense(100, softmax) → policy
                                                     └─ Dense(1, tanh)      → value
```

Training changes:
- Value loss: `MSE(V(s), actual_return)`
- Advantage: `A = R - V(s)` (or GAE with λ=0.95)
- Policy loss: `-E[A · log π(a|s)]`
- Total loss: `policy_loss + 0.5 * value_loss - 0.01 * entropy`

This is A2C (Advantage Actor-Critic). Single biggest improvement we can make.

### 1.2 Margin-Based Reward

Replace binary `±1` with continuous:
```
reward = (my_cells - opp_cells) / total_cells  ∈ [-1, +1]
```

### 1.3 Discounted Returns + GAE

Compute proper discounted returns with γ=0.99 and GAE λ=0.95:
```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
A_t = Σ (γλ)^k · δ_{t+k}
```

Requires storing V(s) predictions during game play.

### 1.4 Snapshot ELO

- Save network weights every 100 generations
- Store as JSON blobs (small: ~300KB each)
- Every 500 generations, play round-robin tournament
- Compute ELO ratings, display curve in UI

### 1.5 Async Batch Management

Remove 300ms dead-slot wait. When a game finishes:
1. Harvest experiences immediately
2. Replace with fresh game in same tick
3. Keep batch always full at 40

## Phase 2: PPO / GRPO (algorithm upgrade)

### 2.1 PPO (Proximal Policy Optimization)

Replace A2C's unconstrained policy gradient with PPO's clipped objective:

```
ratio = π_new(a|s) / π_old(a|s)
L_clip = min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
```

Requires storing `π_old(a|s)` (the log-prob at time of action) alongside each experience. Multiple epochs over the same batch.

**Why PPO over REINFORCE**: PPO prevents catastrophic policy updates. With REINFORCE, one bad batch can destroy the policy. PPO clips the update magnitude.

### 2.2 GRPO (Group Relative Policy Optimization)

GRPO's key idea: for a given state, sample K actions. Compute rewards for each. Normalize rewards within the group (subtract mean, divide by std). No value network needed.

```
For state s:
  Sample K actions: a_1, ..., a_K from π(·|s)
  Get rewards: r_1, ..., r_K
  Normalize: A_i = (r_i - mean(r)) / std(r)
  Update: ∇ = Σ A_i · ∇log π(a_i|s)
```

**Applicability to AlphaPlague**: Partially. GRPO works best when you can get multiple completions for the same prompt/state. In a game, you could fork the game at each decision point and play out K different moves. But the stochastic plague spread means the same move leads to different outcomes, which adds noise. The group normalization would need to account for this.

**Verdict**: PPO is the safer upgrade path for games. GRPO is brilliant for LLM RLHF where you can score multiple completions deterministically.

### 2.3 Swappable Algorithm Interface

Design an interface so algorithms can be swapped:

```javascript
class RLAlgorithm {
  // Called each step with (state, validMask) → action, logProb
  selectAction(state, mask) {}
  
  // Called with completed episode data
  processEpisode(history, reward) {}
  
  // Called when enough data accumulated
  trainStep() {}
}

class REINFORCEAlgo extends RLAlgorithm { ... }
class A2CAlgo extends RLAlgorithm { ... }
class PPOAlgo extends RLAlgorithm { ... }
class GRPOAlgo extends RLAlgorithm { ... }
```

## Phase 3: WebGPU Acceleration

### 3.1 TF.js WebGPU Backend

Upgrade TF.js from 0.11.5 to 4.x. Swap backend:

```javascript
import '@tensorflow/tfjs-backend-webgpu';
await tf.setBackend('webgpu');
```

Expected improvements:
- Batch=40 inference: 7.7ms → ~1ms
- Training: 288ms → ~30ms

**Risk**: TF.js 4.x may have breaking API changes from 0.11.5. Will need to update model construction code. Training gradient ops may not all be supported on WebGPU — may need CPU fallback for training.

### 3.2 WebGPU Compute for Game Simulation

Move `spreadPlague()` to a WGSL compute shader. Store all game boards in a single `GPUBuffer`. One dispatch = one plague step for ALL games.

Board representation in GPU memory:
```
games_buffer: i32[NUM_GAMES * BOARD_SIZE]
  game 0: [cell_0, cell_1, ..., cell_99]
  game 1: [cell_0, cell_1, ..., cell_99]
  ...
```

Compute shader workgroups: one per game, 100 threads per workgroup (one per cell).

### 3.3 WebGPU Rendering

Replace Canvas 2D with a WebGPU render pipeline:
- Storage buffer → vertex shader → fragment shader
- One draw call renders all games as a grid of colored quads
- No CPU involvement per frame

## Phase 4: Web Worker Architecture

### 4.1 Thread Architecture

```
Main Thread          Worker: Simulation       Worker: NN/Training
─────────────        ──────────────────       ──────────────────
UI events            Game loop (pure JS)      Model inference
Stats display        Experience harvesting    Training loop
Canvas compositing   Board state updates      Weight snapshots

        ◄──── SharedArrayBuffer ────►
        Board states, actions, experiences
```

### 4.2 SharedArrayBuffer Protocol

```javascript
// Shared memory layout
const shared = new SharedArrayBuffer(
  NUM_GAMES * BOARD_SIZE * 4  // board states (float32 for NN)
  + NUM_GAMES * 4             // actions (int32)
  + NUM_GAMES * 4             // valid move counts (int32)
  + EXPERIENCE_BUFFER_SIZE    // ring buffer for experiences
);
```

Simulation worker writes board states → NN worker reads them, writes actions → Simulation worker reads actions, advances games. Lock-free via `Atomics.wait` / `Atomics.notify`.

### 4.3 Off-Policy Tolerance

With async workers, the NN inference and training will be slightly behind the game simulation. Games may be played with a policy that's 1-2 training steps old. This is fine for PPO (which handles off-policy data via importance sampling / clipping). For vanilla REINFORCE, staleness would be a problem.

This is why Phase 2 (PPO) should come before Phase 4 (Workers). PPO naturally handles the off-policy gap that async introduces.

## Phase 5: Population and ELO

### 5.1 Snapshot League

```
Every 100 generations:
  snapshots.push(model.getWeights())

Every 500 generations:
  For each pair (snapshot_i, snapshot_j):
    Play 50 games
    Record wins/losses
  Compute ELO ratings via maximum likelihood
  Display ELO chart in UI
```

### 5.2 Opponent Sampling

Instead of always self-play, sample opponent from:
- 80%: current policy (self-play)
- 15%: random historical snapshot (league play)
- 5%: uniform random (exploration / sanity check)

This prevents strategy cycling and provides diverse training signal.

### 5.3 Gatekeeper

New policy only becomes "champion" if it beats the previous champion >55% over 200 games. Prevents regression.

## Phase 6: Scale Up

### 6.1 Larger Boards

Test 15×15, 20×20, 30×30. Larger boards = longer games = more complex strategy. Adjust network size accordingly.

### 6.2 CNN Architecture

Replace dense layers with convolutional layers for spatial reasoning:
```
Input [10, 10, 1] → Conv2D(32, 3×3) → Conv2D(64, 3×3) → Flatten → Dense heads
```

CNNs capture spatial patterns (clusters, borders, encirclement) that dense layers must learn implicitly.

### 6.3 MCTS Integration

Add Monte Carlo Tree Search using the policy-value network:
- Policy head guides search (which moves to explore)
- Value head evaluates leaf positions
- MCTS selects the best move via tree statistics
- Train on MCTS-improved policy targets (not raw policy output)

This is the full AlphaZero pipeline. Most complex but most powerful.
