# PPO in AlphaPlague

This document explains **Proximal Policy Optimization (PPO)** in general and **how this project implements it**. The canonical implementation is [`src/ppo.js`](../src/ppo.js).

## Is our setup the right way?

**Yes.** The intended loop is:

1. **Rollout (inference)** — Forward the policy to choose actions. Store state, action, **log-probability under the policy at collection time** (`oldLogProb`), **value** \(V(s)\), mask, and trajectory metadata. No gradients.

2. **After each game** — Turn each player’s sub-trajectory into **advantages** and **returns**, then append transitions to the replay buffer.

3. **Optimization** — Sample a batch from the buffer, then for several **epochs** and **minibatches**: run a **fresh forward** through the **current** weights, build the PPO loss (clipped surrogate + value + entropy), and **backprop once** per minibatch.

You do **not** cache the backward pass from rollout: the training loss depends on **today’s** logits versus **stored** old log-probs (the importance ratio). Weights change between collection and updates (and across PPO epochs), so the extra forward during `train()` is required, not redundant.

---

## What PPO is (conceptually)

**Policy gradient** methods adjust the policy in the direction that increases expected return. Naive updates can be unstable: one bad large step can wreck the policy.

**PPO** keeps most of the simplicity of policy gradients but limits how much the policy can change per update:

- Define **probability ratio** \(r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}\) using the **current** policy \(\pi_\theta\) and the **behavior** policy at data collection \(\pi_{\theta_{\text{old}}}\) (we store \(\log \pi_{\theta_{\text{old}}}\) as `oldLogProb`).

- Use a **clipped surrogate** so increasing \(r_t\) only helps up to a cap (and decreasing \(r_t\) only hurts down to a floor), which discourages destructively large policy shifts while still allowing many gradient steps on the **same** rollout batch.

Typical PPO objective (per sample, then averaged):

\[
L^{\text{CLIP}}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta)\,\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat{A}_t \right) \right]
\]

We **maximize** that (implementation adds a negative sign and minimizes). A **critic** (value head) is trained to predict returns, and **entropy** is often added to encourage exploration.

---

## How we implement it

### Model outputs

The network outputs **policy logits** over board cells and a **scalar value** per state. **Illegal moves** are masked (logits shifted to \(-10^9\) before softmax) so training matches inference ([`src/action.js`](../src/action.js) on CPU rollouts; same masking in `train()`).

### Collection (`selectActions`)

- One forward per batch: `model.forward` → logits + value.
- CPU-side masked softmax, sampling, and **log-prob of the chosen action** (stable log-sum-exp over legal moves; see `logProbOfAction` in `action.js`).
- Returns `{ action, logProb, value }` per state. **Entropy** for adaptive regularization is estimated from the same CPU probs (`lastEntropy`).

When the GPU batched path is used ([`src/nextgen/runtime/gpu_owner_runtime.js`](../src/nextgen/runtime/gpu_owner_runtime.js)), the same quantities are produced on-GPU (multinomial sample, `logSoftmax` for the taken action, value head).

### End of game (`onGameFinished`)

- Split the trajectory by **player** (two-player game).
- **Rewards**: sparse terminal signal — win \(+1\), loss \(-1\), draw \(0\) on the **last** step of that player’s segment; intermediate steps \(0\).
- **GAE (Generalized Advantage Estimation)** backward in time:

  \[
  \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad
  A_t = \delta_t + \gamma\lambda\, A_{t+1}
  \]

  with \(V\) after the last step treated as \(0\) (terminal, no bootstrap beyond episode end).

- **Return** for the critic: \(\hat{R}_t = A_t + V(s_t)\).

- Each transition pushed to `buffer`: `state`, `action`, `oldLogProb`, `advantage`, `returnVal`, `mask`.

### Training (`train`)

1. **Splice** up to `batchSize` items from the front of the buffer (consumed, not infinite replay of the same batch unless refilled by new games).

2. **Normalize advantages** to zero mean, unit variance (minibatch-wide, over the full batch before epoch loop).

3. For **`epochs`** (default 2), with shuffled indices, for each **minibatch** (default size 128):

   - `model.model.predict` on minibatch states → logits + value.
   - Masked softmax, `newLogProbs` for stored actions.
   - **Ratio** \(r = \exp(\log \pi_\theta - \texttt{oldLogProb})\).
   - **Policy loss**: negative mean of `min(r * A, clip(r) * A)` with \(\varepsilon = 0.2\).
   - **Value loss**: MSE between predicted value and `returnVal`, scaled by `valueLossCoeff` (0.5).
   - **Entropy bonus**: subtract `entropyCoeff * entropy` (encourage spread over legal moves).

4. **Adam** optimizer (lr `3e-4`), one `minimize()` per minibatch.

5. **Adaptive entropy** (outside the tape): nudge `entropyCoeff` toward a **target entropy** derived from board size (~50% of max entropy at ~30% valid cells). Clamped to `[0.001, 0.1]`.

### Hyperparameters (defaults in `ppo.js`)

| Symbol / concept | Code | Default |
|------------------|------|--------|
| Clip \(\varepsilon\) | `epsilon` | 0.2 |
| Discount \(\gamma\) | `gamma` | 0.99 |
| GAE \(\lambda\) | `lambda` | 0.95 |
| PPO epochs \(K\) | `epochs` | 2 |
| Minibatch size | `minibatchSize` | 128 |
| Value loss scale | `valueLossCoeff` | 0.5 |
| Max buffer | `maxBufferSize` | 20000 |

Training is triggered when enough games have finished and the buffer is large enough (`shouldTrain` — see call sites in the trainer / orchestration).

### PPG extension

[`src/ppg.js`](../src/ppg.js) runs **standard PPO** first (`super.train`), then an optional auxiliary critic phase on a separate sample — details live in that file.

---

## File map

| Piece | Location |
|--------|----------|
| PPO class, buffer, GAE, `train()` | [`src/ppo.js`](../src/ppo.js) |
| Masked softmax / log-prob helpers | [`src/action.js`](../src/action.js) |
| Algorithm registration (UI “PPO”) | [`src/algo_registry.js`](../src/algo_registry.js) |
| GPU batched rollout (log prob + value) | [`src/nextgen/runtime/gpu_owner_runtime.js`](../src/nextgen/runtime/gpu_owner_runtime.js) |

---

## References

- Schulman et al., *Proximal Policy Optimization Algorithms* (arXiv:1707.06347).
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE).
