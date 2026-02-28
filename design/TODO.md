# AlphaPlague — TODO

## Immediate (Phase 1 — fix RL fundamentals)

- [ ] **Add value head to network** — second output `Dense(1, tanh)` predicting game outcome
- [ ] **Implement A2C** — advantage = return - V(s), value loss + policy loss + entropy
- [ ] **Margin-based reward** — `(my_cells - opp_cells) / total` instead of binary ±1
- [ ] **GAE (λ=0.95, γ=0.99)** — proper temporal credit assignment
- [ ] **Fix batch management** — remove 300ms dead slot wait, immediately replace finished games
- [ ] **Skip rendering at high speed** — only render every Nth frame when speed > 10x
- [ ] **Snapshot ELO** — save weights every 100 gens, play round-robin, display ELO chart

## Short-term (Phase 2 — algorithm upgrade)

- [ ] **Implement PPO** — clipped surrogate objective, multiple epochs per batch
- [ ] **Store log-probs with experiences** — needed for importance sampling ratio
- [ ] **Design swappable algorithm interface** — `RLAlgorithm` base class
- [ ] **Implement GRPO variant** — group sampling, relative advantage normalization
- [ ] **Add algorithm selector to UI** — dropdown: REINFORCE / A2C / PPO / GRPO
- [ ] **Add heuristic opponents** — center-first, greedy-spread, random (ELO baselines)
- [ ] **Win rate vs baselines display** — chart showing improvement over time

## Medium-term (Phase 3 — WebGPU acceleration)

- [ ] **Upgrade TF.js to 4.x** — breaking change migration from 0.11.5
- [ ] **Enable WebGPU backend** — `tf.setBackend('webgpu')` with CPU fallback
- [ ] **Benchmark WebGPU vs CPU** — measure inference and training speedup
- [ ] **WebGPU compute shader for plague spread** — WGSL shader, one dispatch for all games
- [ ] **WebGPU render pipeline** — replace Canvas 2D, one draw call for entire grid
- [ ] **ImageData rendering fallback** — for browsers without WebGPU

## Longer-term (Phase 4 — Web Workers)

- [ ] **Move game simulation to Worker** — pure JS, no DOM dependency
- [ ] **Move NN inference/training to Worker** — unblock main thread
- [ ] **SharedArrayBuffer protocol** — zero-copy shared state between workers
- [ ] **Atomics-based synchronization** — lock-free producer/consumer for experiences
- [ ] **OffscreenCanvas rendering** — render in worker, transfer to main thread
- [ ] **Handle off-policy staleness** — PPO clipping handles 1-2 step delay naturally

## Ambitious (Phase 5-6 — full AlphaZero pipeline)

- [ ] **Population-based training** — maintain K networks, cross-play ELO league
- [ ] **Opponent sampling** — 80% self-play, 15% historical, 5% random
- [ ] **Gatekeeper** — new policy must beat champion >55% to promote
- [ ] **CNN architecture** — Conv2D layers for spatial pattern recognition
- [ ] **MCTS integration** — tree search guided by policy+value network
- [ ] **Larger boards** — 15×15, 20×20, 30×30 with scaled networks
- [ ] **TrueSkill rating** — Bayesian skill rating with uncertainty tracking

## RL Algorithm Bank (swappable implementations)

| Algorithm | Type | Value Net? | Off-Policy? | Best For |
|---|---|---|---|---|
| **REINFORCE** | Policy gradient | No | No | Baseline / debugging |
| **A2C** | Actor-critic | Yes | No | Stable self-play training |
| **PPO** | Clipped actor-critic | Yes | Mildly | Async workers, robust |
| **GRPO** | Group-relative PG | No (group norm) | No | Multiple rollouts per state |
| **IMPALA** | Async actor-critic | Yes | Yes (V-trace) | Large-scale distributed |
| **APE-X** | Distributed DQN | Yes (Q) | Yes (replay) | Massively parallel actors |

## Research References

- **AlphaZero** (Silver et al., 2017) — MCTS + self-play + policy-value network
- **PPO** (Schulman et al., 2017) — clipped surrogate objective
- **GRPO** (DeepSeek, 2024) — group-relative advantage, no critic
- **IMPALA** (Espeholt et al., 2018) — V-trace correction for async actors
- **SPIRAL** (2025) — self-play on zero-sum games for reasoning
- **QZero** (2026) — model-free AlphaGo via entropy-regularized Q-learning
- **RGSC** (2026) — regret-guided search control, +77-89 ELO over AlphaZero
- **AReaL** (2025) — fully async RL with staleness-enhanced PPO, 2.77× speedup
