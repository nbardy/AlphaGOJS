// alphagojs_v2/src/fused_ppo.wgsl
// --------------------------------------------------------------------------
// PRO-GRADE FUSED PPO KERNEL
// Architecture: Patch-based ConvNet with Skip Connections and Pixel Shuffle.
// Optimizations: L1 Caching, Cooperative Chunking, Hogwild Internal-Loop.
// --------------------------------------------------------------------------
enable f16;

const K: u32 = 24u;
const D: u32 = 8u;
const WG: u32 = 64u;
const MAX_STEPS: u32 = 600u;

const N: u32 = K * K;
const P: u32 = K / 3u; // 8
const PATCH_CH: u32 = 9u * D; // 72
const WPB: u32 = (N + 15u) / 16u; // 36
const MASK_FILL: f32 = -3.4e38;
const EPS: f32 = 1.0e-12;

// Dense Weights
const W_CONV1: u32 = 0u;
const W_CONV2: u32 = W_CONV1 + 9u * D * D;
const W_FUSE: u32 = W_CONV2 + 9u * D * PATCH_CH;
const W_POLICY: u32 = W_FUSE + 2u * D * D;
const W_VALUE: u32 = W_POLICY + D;
const W_TOTAL: u32 = W_VALUE + D + 1u;
const W_OPP: u32 = W_TOTAL * 3u;

// Embed Weights
const E_CELL: u32 = 0u;
const E_PATCH: u32 = E_CELL + 4u * D;
const E_TOTAL: u32 = E_PATCH + 262144u * D; // 2097184
const E_OPP: u32 = E_TOTAL * 3u;

const DW2_SIZE: u32 = 9u * D * 9u * D;
const DW2_PER_THREAD: u32 = (DW2_SIZE + WG - 1u) / WG;

struct Params {
  batch_size: u32,
  step: u32,
  seed: u32,
  max_steps: u32,
  
  adam_step: u32,
  use_opponent: u32,
  _p1: u32, _p2: u32,
  
  lr: f32,
  beta1: f32,
  beta2: f32,
  eps_adam: f32,
  
  epsilon_clip: f32,
  c1_value: f32,
  c2_entropy: f32,
  gamma: f32,
  
  gae_lambda: f32,
  grad_clip: f32,
  weight_decay: f32,
  elo_scale: f32,
  
  territory_bonus: f32,
  wall_density: f32,
  _pad1: f32,
  _pad2: f32,
}

struct Transition {
  action: u32,
  log_prob: f16,
  value: f16,
  reward: f16,
  advantage: f16,
  value_target: f16,
  _pad: f16,
  board: array<u32, WPB>,
}

struct BoardState {
  packed: array<u32, WPB>,
  game_over: u32,
  player: u32,
  step_count: u32,
  _pad: u32,
  transitions: array<Transition, MAX_STEPS>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> boards: array<BoardState>;
@group(0) @binding(2) var<storage, read_write> embed_w: array<f16>;
@group(0) @binding(3) var<storage, read_write> dense_w: array<f32>;

var<workgroup> sh_board: array<u32, 36u>;
var<workgroup> sh_patch_state: array<u32, 64u>;
var<workgroup> sh_a: array<f32, P * P * D>;
var<workgroup> sh_b: array<f32, 576u>;
var<workgroup> sh_pool: array<f32, 64u>;
var<workgroup> sh_reduce_m: array<f32, 64u>;
var<workgroup> sh_reduce_s: array<f32, 64u>;
var<workgroup> sh_patch_delta2: array<f32, PATCH_CH>;
var<workgroup> sh_value: f32;
var<workgroup> sh_action: u32;
var<workgroup> sh_log_prob: f32;
var<workgroup> sh_delta_v: f32;
var<workgroup> sh_total_steps: u32;
var<workgroup> sh_bar_a1: array<f32, P * P * D>;

var<private> p_invert: bool = false;
var<private> w_offset: u32 = 0u;
var<private> e_offset: u32 = 0u;

fn pcg(s: u32) -> u32 {
  var st = s * 747796405u + 2891336453u;
  let w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rng_float(seed: u32, a: u32, b: u32) -> f32 {
  return f32(pcg(seed ^ pcg(a ^ pcg(b))) & 0x00FFFFFFu) / 16777216.0;
}

fn get_sh_cell_state(y: u32, x: u32) -> u32 {
  let n = y * K + x;
  let s = (sh_board[n >> 4u] >> ((n & 15u) << 1u)) & 3u;
  if (p_invert) {
    if (s == 1u) { return 2u; }
    if (s == 2u) { return 1u; }
  }
  return s;
}

fn set_cell(b: u32, cell: u32, state: u32) {
  let wi = cell >> 4u;
  let bo = (cell & 15u) << 1u;
  boards[b].packed[wi] = (boards[b].packed[wi] & ~(3u << bo)) | ((state & 3u) << bo);
}

// Weights
fn cell_e(state: u32, d: u32) -> f32 { return f32(embed_w[e_offset + E_CELL + state * D + d]); }
fn patch_e(row: u32, d: u32) -> f32 { return f32(embed_w[e_offset + E_PATCH + row * D + d]); }
fn c1w(ky: u32, kx: u32, c: u32, o: u32) -> f32 { return dense_w[w_offset + W_CONV1 + ((ky*3u+kx)*D+c)*D+o]; }
fn c2w(ky: u32, kx: u32, c: u32, o: u32) -> f32 { return dense_w[w_offset + W_CONV2 + ((ky*3u+kx)*D+c)*PATCH_CH+o]; }
fn fw(c: u32, o: u32) -> f32 { return dense_w[w_offset + W_FUSE + c * D + o]; }
fn pw(d: u32) -> f32 { return dense_w[w_offset + W_POLICY + d]; }
fn vw(d: u32) -> f32 { return dense_w[w_offset + W_VALUE + d]; }
fn vbias() -> f32 { return dense_w[w_offset + W_VALUE + D]; }

struct MS { m: f32, s: f32 }
fn combine(am: f32, as2: f32, bm: f32, bs: f32) -> MS {
  if (as2 == 0.0) { return MS(bm, bs); }
  if (bs == 0.0) { return MS(am, as2); }
  let m = max(am, bm);
  return MS(m, as2 * exp(am - m) + bs * exp(bm - m));
}

fn apply_adam_f32(offset: u32, grad: f32) {
  let adam_t = f32(params.adam_step + 1u);
  let inv_bias1 = 1.0 / (1.0 - pow(params.beta1, adam_t));
  let inv_bias2 = 1.0 / (1.0 - pow(params.beta2, adam_t));
  let m_idx = W_TOTAL + offset;
  let v_idx = W_TOTAL * 2u + offset;
  
  var w = dense_w[offset];
  w *= (1.0 - params.lr * params.weight_decay);
  
  let c_grad = clamp(grad, -params.grad_clip, params.grad_clip);
  let m = params.beta1 * dense_w[m_idx] + (1.0 - params.beta1) * c_grad;
  let v = params.beta2 * dense_w[v_idx] + (1.0 - params.beta2) * c_grad * c_grad;
  
  dense_w[m_idx] = m;
  dense_w[v_idx] = v;
  
  w -= params.lr * (m * inv_bias1) / (sqrt(max(v * inv_bias2, 0.0)) + params.eps_adam);
  dense_w[offset] = w;
}

fn apply_adam_f16(offset: u32, grad: f32) {
  let adam_t = f32(params.adam_step + 1u);
  let inv_bias1 = 1.0 / (1.0 - pow(params.beta1, adam_t));
  let inv_bias2 = 1.0 / (1.0 - pow(params.beta2, adam_t));
  let m_idx = E_TOTAL + offset;
  let v_idx = E_TOTAL * 2u + offset;
  
  var w = f32(embed_w[offset]);
  w *= (1.0 - params.lr * params.weight_decay);
  
  let c_grad = clamp(grad, -params.grad_clip, params.grad_clip);
  let m = params.beta1 * f32(embed_w[m_idx]) + (1.0 - params.beta1) * c_grad;
  let v = params.beta2 * f32(embed_w[v_idx]) + (1.0 - params.beta2) * c_grad * c_grad;
  
  embed_w[m_idx] = f16(m);
  embed_w[v_idx] = f16(max(v, 1e-4));
  
  w -= params.lr * (m * inv_bias1) / (sqrt(max(v * inv_bias2, 0.0)) + params.eps_adam);
  embed_w[offset] = f16(w);
}

fn sh_a_at(py: i32, px: i32, d: u32) -> f32 {
  if (py < 0 || px < 0 || py >= i32(P) || px >= i32(P)) { return 0.0; }
  return sh_a[u32(py) * P * D + u32(px) * D + d];
}

fn conv2_at_patch(py: i32, px: i32, o: u32) -> f32 {
  var acc: f32 = 0.0;
  for (var c: u32 = 0u; c < D; c++) {
    for (var ky: u32 = 0u; ky < 3u; ky++) {
      for (var kx: u32 = 0u; kx < 3u; kx++) {
        let y = py + 2*(i32(ky)-1);
        let x = px + 2*(i32(kx)-1);
        if (y >= 0 && y < i32(P) && x >= 0 && x < i32(P)) {
          acc += c2w(ky, kx, c, o) * sh_a[u32(y)*P*D + u32(x)*D + c];
        }
      }
    }
  }
  return max(acc, 0.0);
}

fn forward_pass(b: u32, lid: u32, is_live: bool) {
  let step = boards[b].step_count;

  // Load Board to L1 Cache
  if (!is_live && lid < WPB) {
    sh_board[lid] = boards[b].transitions[step].board[lid];
  } else if (is_live && lid < WPB) {
    sh_board[lid] = boards[b].packed[lid];
  }
  workgroupBarrier();

  // Patch Embedding
  if (lid < P * P) {
    let py = lid / P; let px = lid % P;
    var pi: u32 = 0u;
    for (var dy: u32 = 0u; dy < 3u; dy++) {
      for (var dx: u32 = 0u; dx < 3u; dx++) {
        pi |= (get_sh_cell_state(py*3u+dy, px*3u+dx) << (2u * (dy*3u+dx)));
      }
    }
    if (!is_live) { sh_patch_state[lid] = pi; }
    for (var d: u32 = 0u; d < D; d++) { sh_a[lid * D + d] = patch_e(pi, d); }
  }
  workgroupBarrier();

  // Conv1
  var conv1_reg: array<f32, D>;
  if (lid < P * P) {
    let py = i32(lid / P); let px = i32(lid % P);
    for (var o: u32 = 0u; o < D; o++) {
      var acc: f32 = 0.0;
      for (var c: u32 = 0u; c < D; c++) {
        for (var ky: u32 = 0u; ky < 3u; ky++) {
          for (var kx: u32 = 0u; kx < 3u; kx++) {
            let y = py + i32(ky) - 1; let x = px + i32(kx) - 1;
            if (y >= 0 && y < i32(P) && x >= 0 && x < i32(P)) {
              acc += c1w(ky, kx, c, o) * sh_a[u32(y)*P*D + u32(x)*D + c];
            }
          }
        }
      }
      conv1_reg[o] = max(acc, 0.0);
    }
  }
  workgroupBarrier();
  if (lid < P * P) {
    for (var o: u32 = 0u; o < D; o++) { sh_a[lid * D + o] = conv1_reg[o]; }
  }
  workgroupBarrier();

  // Conv2 + Pixel Shuffle + Fuse + Heads
  var lm: f32 = MASK_FILL; var ls: f32 = 0.0;
  var pool: array<f32, D>;
  for (var d: u32 = 0u; d < D; d++) { pool[d] = 0.0; }

  for (var cell: u32 = lid; cell < N; cell += WG) {
    let y = cell / K; let x = cell % K;
    let py = y / 3u; let px = x / 3u;
    let sub = (y % 3u) * 3u + (x % 3u);
    let state = get_sh_cell_state(y, x);

    var decoded: array<f32, D>;
    for (var d: u32 = 0u; d < D; d++) { decoded[d] = conv2_at_patch(i32(py), i32(px), sub * D + d); }

    var fused: array<f32, D>;
    for (var o: u32 = 0u; o < D; o++) {
      var acc: f32 = 0.0;
      for (var c: u32 = 0u; c < D; c++) {
        acc += fw(c, o) * decoded[c] + fw(D + c, o) * cell_e(state, c);
      }
      fused[o] = max(acc, 0.0);
      pool[o] += fused[o];
    }

    var logit: f32 = 0.0;
    for (var d: u32 = 0u; d < D; d++) { logit += pw(d) * fused[d]; }
    let valid = (state == 0u);
    let masked = select(MASK_FILL, logit, valid);
    sh_b[cell] = masked;
    if (valid) {
      let ms = combine(lm, ls, masked, 1.0);
      lm = ms.m; ls = ms.s;
    }
  }
  workgroupBarrier();

  // Reductions
  for (var d: u32 = 0u; d < D; d++) {
    sh_pool[lid] = pool[d];
    workgroupBarrier();
    for (var stride: u32 = WG >> 1u; stride > 0u; stride >>= 1u) {
      if (lid < stride) { sh_pool[lid] += sh_pool[lid + stride]; }
      workgroupBarrier();
    }
    if (lid == 0u) { pool[d] = sh_pool[0] / f32(N); }
    workgroupBarrier();
  }

  if (lid == 0u) {
    var v: f32 = vbias();
    for (var d: u32 = 0u; d < D; d++) { v += vw(d) * pool[d]; }
    sh_value = tanh(v);
  }

  sh_reduce_m[lid] = lm; sh_reduce_s[lid] = ls;
  workgroupBarrier();
  for (var stride: u32 = WG >> 1u; stride > 0u; stride >>= 1u) {
    if (lid < stride) {
      let ms = combine(sh_reduce_m[lid], sh_reduce_s[lid], sh_reduce_m[lid+stride], sh_reduce_s[lid+stride]);
      sh_reduce_m[lid] = ms.m; sh_reduce_s[lid] = ms.s;
    }
    workgroupBarrier();
  }
  let gm = sh_reduce_m[0]; let gs = max(sh_reduce_s[0], EPS);

  for (var cell: u32 = lid; cell < N; cell += WG) {
    let state = get_sh_cell_state(cell / K, cell % K);
    sh_b[cell] = select(0.0, exp(sh_b[cell] - gm) / gs, state == 0u);
  }
  workgroupBarrier();
}


// --------------------------------------------------------------------------
// Entry: ROLLOUT (Self-Play Generation)
// --------------------------------------------------------------------------
@compute @workgroup_size(WG)
fn rollout_step(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) loc: vec3<u32>) {
  let b = wg.x;
  if (b >= params.batch_size) { return; }
  let lid = loc.x;
  let is_live = boards[b].game_over == 0u;

  if (params.use_opponent == 1u && boards[b].player == 2u) {
    w_offset = W_OPP; e_offset = E_OPP;
  } else {
    w_offset = 0u; e_offset = 0u;
  }

  let step = boards[b].step_count;
  if (is_live) {
    for (var w: u32 = lid; w < WPB; w += WG) { boards[b].transitions[step].board[w] = boards[b].packed[w]; }
  }
  workgroupBarrier();

  forward_pass(b, lid, true);

  if (lid == 0u && is_live) {
    let rand = rng_float(params.seed, params.step, b);
    var cumsum: f32 = 0.0; var chosen: u32 = 0u; var chosen_p: f32 = 0.0;
    for (var i: u32 = 0u; i < N; i++) {
      let p = sh_b[i];
      if (p <= 0.0) { continue; }
      cumsum += p;
      if (rand < cumsum) { chosen = i; chosen_p = p; break; }
    }
    if (chosen_p == 0.0) {
      for (var i: u32 = 0u; i < N; i++) { if (sh_b[i] > 0.0) { chosen = i; chosen_p = sh_b[i]; break; } }
    }
    sh_action = chosen;
    sh_log_prob = log(max(chosen_p, EPS));

    boards[b].transitions[step].action = chosen;
    boards[b].transitions[step].log_prob = f16(sh_log_prob);
    boards[b].transitions[step].value = f16(sh_value);
    boards[b].transitions[step].reward = f16(0.0);
  }
  workgroupBarrier();

  // 1. Initialize sh_b with the old state
  for (var cell: u32 = lid; cell < N; cell += WG) {
    sh_b[cell] = f32(get_sh_cell_state(cell / K, cell % K));
  }
  workgroupBarrier();

  // 2. Network places its piece
  if (lid == 0u && is_live) {
    sh_b[sh_action] = f32(boards[b].player);
    boards[b].player = select(1u, 2u, boards[b].player == 1u);
  }
  workgroupBarrier();

  p_invert = false;
  // 3. Spread Plague (only on cells that are empty and NOT the new action)
  for (var cell: u32 = lid; cell < N; cell += WG) {
    let r = cell / K; let c = cell % K;
    let state = get_sh_cell_state(r, c);
    
    if (!is_live || state != 0u || cell == sh_action) { continue; }
    
    var sum: f32 = 0.0;
    let rb = params.seed ^ params.step ^ cell;

    if (r > 0u) { let n = get_sh_cell_state(r-1u, c); if (n == 1u) { sum += rng_float(rb,0u,b); } else if (n == 2u) { sum -= rng_float(rb,0u,b); } }
    if (r + 1u < K) { let n = get_sh_cell_state(r+1u, c); if (n == 1u) { sum += rng_float(rb,1u,b); } else if (n == 2u) { sum -= rng_float(rb,1u,b); } }
    if (c > 0u) { let n = get_sh_cell_state(r, c-1u); if (n == 1u) { sum += rng_float(rb,2u,b); } else if (n == 2u) { sum -= rng_float(rb,2u,b); } }
    if (c + 1u < K) { let n = get_sh_cell_state(r, c+1u); if (n == 1u) { sum += rng_float(rb,3u,b); } else if (n == 2u) { sum -= rng_float(rb,3u,b); } }

    let v = clamp(trunc(sum * 2.0), -1.0, 1.0);
    sh_b[cell] = f32(select(0u, 1u, v > 0.0) | select(0u, 2u, v < 0.0));
  }
  workgroupBarrier();

  // 4. Safe, race-free parallel repacking to VRAM
  if (is_live && lid < WPB) {
    var word: u32 = 0u;
    for (var i = 0u; i < 16u; i++) {
      let cell = lid * 16u + i;
      if (cell < N) {
        let state = u32(sh_b[cell]);
        word |= (state << (i * 2u));
      }
    }
    boards[b].packed[lid] = word;
  }
  workgroupBarrier();

  if (lid == 0u && is_live) {
    var has_empty: bool = false;
    let full_words = N >> 4u; let leftover = N & 15u;
    for (var w: u32 = 0u; w < full_words; w++) {
      let word = boards[b].packed[w];
      if (((word | (word >> 1u)) & 0x55555555u) != 0x55555555u) { has_empty = true; break; }
    }
    if (!has_empty && leftover > 0u) {
      let word = boards[b].packed[full_words];
      let mask = (1u << (leftover * 2u)) - 1u;
      let check_mask = 0x55555555u & mask;
      if ((((word & mask) | ((word & mask) >> 1u)) & check_mask) != check_mask) { has_empty = true; }
    }
    let ns = step + 1u;
    boards[b].step_count = ns;
    
    if (!has_empty || ns >= params.max_steps) {
      boards[b].game_over = 1u;
      var p1: u32 = 0u; var p2: u32 = 0u;
      for (var i: u32 = 0u; i < N; i++) {
        let s = get_sh_cell_state(i/K, i%K);
        if (s == 1u) { p1++; } else if (s == 2u) { p2++; }
      }
      
      let base_reward = select(select(-1.0, 1.0, p1 > p2), 0.0, p1 == p2);
      let total_cells = f32(p1 + p2);
      let p1_pct = select(0.5, f32(p1) / total_cells, total_cells > 0.0);
      let territory_diff = (p1_pct - 0.5) * 2.0;
      
      let scaled_reward = (base_reward * params.elo_scale) + (territory_diff * params.territory_bonus);
      boards[b].transitions[ns - 1u].reward = f16(scaled_reward);
    }
  }
}

// --------------------------------------------------------------------------
// Entry: GAE SCAN
// --------------------------------------------------------------------------
@compute @workgroup_size(WG)
fn gae_scan(@builtin(global_invocation_id) id: vec3<u32>) {
  let b = id.x;
  if (b >= params.batch_size) { return; }
  let T = boards[b].step_count;
  if (T == 0u) { return; }

  var gae: f32 = 0.0;
  for (var t1: u32 = T; t1 > 0u; t1--) {
    let t = t1 - 1u;
    let reward = f32(boards[b].transitions[t].reward);
    let value = f32(boards[b].transitions[t].value);
    let nv = select(f32(boards[b].transitions[t+1u].value), 0.0, t+1u >= T);
    let delta = reward + params.gamma * nv - value;
    gae = delta + params.gamma * params.gae_lambda * gae;
    boards[b].transitions[t].advantage = f16(gae);
    boards[b].transitions[t].value_target = f16(gae + value);
  }
}

// --------------------------------------------------------------------------
// Entry: FUSED PPO + ADAMW
// --------------------------------------------------------------------------
@compute @workgroup_size(WG)
fn ppo_step(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) loc: vec3<u32>) {
  let b = wg.x;
  if (b >= params.batch_size) { return; }
  let lid = loc.x;

  if (lid == 0u) { sh_total_steps = min(boards[b].step_count, params.max_steps); }
  workgroupBarrier();
  let total_steps = workgroupUniformLoad(&sh_total_steps);

  for (var step = 0u; step < total_steps; step++) {
    p_invert = ((step % 2u) != 0u);
    if (params.use_opponent == 1u && p_invert) {
      w_offset = W_OPP; e_offset = E_OPP;
    } else {
      w_offset = 0u; e_offset = 0u;
    }
    forward_pass(b, lid, false);

    if (lid == 0u) {
      let action = boards[b].transitions[step].action;
      let old_lp = f32(boards[b].transitions[step].log_prob);
      let adv = f32(boards[b].transitions[step].advantage);
      let v_tar = f32(boards[b].transitions[step].value_target);

      let ratio = exp(log(max(sh_b[action], EPS)) - old_lp);
      let surr1 = ratio * adv;
      let surr2 = clamp(ratio, 1.0 - params.epsilon_clip, 1.0 + params.epsilon_clip) * adv;
      let clip_loss = -min(surr1, surr2);
      let v_loss = params.c1_value * (sh_value - v_tar) * (sh_value - v_tar);

      var entropy: f32 = 0.0;
      for (var i: u32 = 0u; i < N; i++) {
        let p = sh_b[i];
        if (p > EPS) { entropy -= p * log(p); }
      }

      boards[b].transitions[0u].reward = f16(clip_loss + v_loss - params.c2_entropy * entropy);
      // Smuggle raw policy entropy H out for CPU readback (mirrors the loss path above).
      // _pad is an otherwise-unused f16 slot in Transition, so this changes no struct stride.
      boards[b].transitions[0u]._pad = f16(entropy);

      let is_p2_turn = (step % 2u) != 0u;
      let is_opponent_turn = params.use_opponent == 1u && is_p2_turn;

      var g_lp = 0.0;
      var d_v = 0.0;
      if (!is_opponent_turn) {
        let e = params.epsilon_clip;
        if ((adv >= 0.0 && ratio < 1.0 + e) || (adv < 0.0 && ratio > 1.0 - e)) {
          g_lp = -(adv * ratio) / f32(params.batch_size);
        }
        d_v = (2.0 * params.c1_value / f32(params.batch_size)) * (sh_value - v_tar) * (1.0 - sh_value * sh_value);
      }
      
      sh_delta_v = d_v;
      sh_pool[0] = entropy;
      sh_pool[1] = g_lp;
    }
    workgroupBarrier();

    let H_b = sh_pool[0];
    let sh_g_lp = sh_pool[1];
    let c2_B = params.c2_entropy / f32(params.batch_size);
    let action = boards[b].transitions[step].action;

    for (var i: u32 = lid; i < N; i += WG) {
      let p_i = sh_b[i];
      var delta_pi = 0.0;
      if (p_i > 0.0) {
        let indicator = select(0.0, 1.0, i == action);
        delta_pi = sh_g_lp * (indicator - p_i) + c2_B * p_i * (log(max(p_i, EPS)) + H_b);
      }
      sh_b[i] = delta_pi;
    }
    workgroupBarrier();

    var l_dWpi: array<f32, D>;
    var l_dWv: array<f32, D>;
    var l_dbv: f32 = 0.0;
    var l_dWf: array<f32, 128>;
    var l_dE_cell: array<f32, 32>;

    if (lid == 0u) { l_dbv = sh_delta_v; }

    var local_dW2: array<f32, DW2_PER_THREAD>;
    for (var i = 0u; i < DW2_PER_THREAD; i++) { local_dW2[i] = 0.0; }

    for (var i = 0u; i < 512u; i += WG) { sh_bar_a1[lid + i] = 0.0; }
    workgroupBarrier();

    for (var patch_idx = 0u; patch_idx < 64u; patch_idx++) {
      if (lid < 9u) {
        let sub = lid;
        let py = patch_idx / 8u; let px = patch_idx % 8u;
        let y = py * 3u + sub / 3u; let x = px * 3u + sub % 3u;
        let state = get_sh_cell_state(y, x);

        var decoded: array<f32, D>;
        for (var c = 0u; c < 8u; c++) { decoded[c] = conv2_at_patch(i32(py), i32(px), sub * 8u + c); }

        var af: array<f32, D>;
        for (var o = 0u; o < 8u; o++) {
          var acc = 0.0;
          for (var c = 0u; c < 8u; c++) { acc += fw(c, o) * decoded[c] + fw(8u + c, o) * cell_e(state, c); }
          af[o] = max(acc, 0.0);
        }

        let delta_pi_i = sh_b[y * K + x];
        let delta_v_N = sh_delta_v / f32(N);
        var delta_f: array<f32, D>;

        for (var o = 0u; o < 8u; o++) {
          l_dWpi[o] += delta_pi_i * af[o];
          l_dWv[o] += delta_v_N * af[o];
          
          let bar_af = delta_pi_i * pw(o) + delta_v_N * vw(o);
          delta_f[o] = select(0.0, bar_af, af[o] > 0.0);
          
          for (var c = 0u; c < 8u; c++) {
            l_dWf[c * 8u + o] += delta_f[o] * decoded[c];
            l_dWf[(8u + c) * 8u + o] += delta_f[o] * cell_e(state, c);
            l_dE_cell[state * 8u + c] += delta_f[o] * fw(8u + c, o);
          }
        }

        for (var d = 0u; d < 8u; d++) {
          var bar_decoded_d = 0.0;
          for (var o = 0u; o < 8u; o++) { bar_decoded_d += delta_f[o] * fw(d, o); }
          sh_patch_delta2[sub * 8u + d] = select(0.0, bar_decoded_d, decoded[d] > 0.0);
        }
      }
      workgroupBarrier();

      for (var i = 0u; i < DW2_PER_THREAD; i++) {
        let w_idx = lid + i * 64u;
        if (w_idx < DW2_SIZE) {
          let k = w_idx % 72u; let c = (w_idx / 72u) % 8u;
          let kx = (w_idx / 576u) % 3u; let ky = (w_idx / 1728u) % 3u;
          let py = patch_idx / 8u; let px = patch_idx % 8u;
          local_dW2[i] += sh_patch_delta2[k] * sh_a_at(i32(py) + 2*(i32(ky)-1), i32(px) + 2*(i32(kx)-1), c);
        }
      }

      for (var item = lid; item < 72u; item += WG) {
        let c = item % 8u; let v = (item / 8u) % 3u; let u = (item / 24u) % 3u;
        let py = i32(patch_idx / 8u) + 2 * (i32(u) - 1);
        let px = i32(patch_idx % 8u) + 2 * (i32(v) - 1);
        if (py >= 0 && py < 8 && px >= 0 && px < 8) {
          var sum = 0.0;
          for (var k = 0u; k < 72u; k++) { sum += sh_patch_delta2[k] * c2w(u, v, c, k); }
          sh_bar_a1[u32(py) * 64u + u32(px) * 8u + c] += sum;
        }
      }
      workgroupBarrier();
    }

    for (var chunk = 0u; chunk < 177u; chunk += 2u) {
      let count = min(2u, 177u - chunk);
      let c0 = chunk; let c1 = chunk + 1u;
      
      var val0 = 0.0;
      if (c0 < 8u) { val0 = l_dWpi[c0]; }
      else if (c0 < 16u) { val0 = l_dWv[c0 - 8u]; }
      else if (c0 == 16u) { val0 = l_dbv; }
      else if (c0 < 145u) { val0 = l_dWf[c0 - 17u]; }
      else { val0 = l_dE_cell[c0 - 145u]; }
      
      var val1 = 0.0;
      if (c1 < 8u) { val1 = l_dWpi[c1]; }
      else if (c1 < 16u) { val1 = l_dWv[c1 - 8u]; }
      else if (c1 == 16u) { val1 = l_dbv; }
      else if (c1 < 145u) { val1 = l_dWf[c1 - 17u]; }
      else { val1 = l_dE_cell[c1 - 145u]; }
      
      if (count > 0u) { sh_pool[lid] = val0; }
      if (count > 1u) { sh_reduce_m[lid] = val1; }
      workgroupBarrier();
      
      if (lid < count) {
        var sum = 0.0;
        if (lid == 0u) { for(var t=0u; t<64u; t++) { sum += sh_pool[t]; } }
        if (lid == 1u) { for(var t=0u; t<64u; t++) { sum += sh_reduce_m[t]; } }
        
        let w_idx = chunk + lid;
        if (w_idx < 8u) { apply_adam_f32(W_POLICY + w_idx, sum); }
        else if (w_idx < 16u) { apply_adam_f32(W_VALUE + (w_idx - 8u), sum); }
        else if (w_idx == 16u) { apply_adam_f32(W_VALUE + D, sum); }
        else if (w_idx < 145u) { apply_adam_f32(W_FUSE + (w_idx - 17u), sum); }
        else { apply_adam_f16(E_CELL + (w_idx - 145u), sum); }
      }
      workgroupBarrier();
    }

    for (var i = 0u; i < DW2_PER_THREAD; i++) {
      let w_idx = lid + i * 64u;
      if (w_idx < DW2_SIZE) { apply_adam_f32(W_CONV2 + w_idx, local_dW2[i]); }
    }
    workgroupBarrier();

    for (var w_idx = lid; w_idx < 576u; w_idx += WG) {
      let o = w_idx % 8u; let c = (w_idx / 8u) % 8u;
      let v = (w_idx / 64u) % 3u; let u = (w_idx / 192u) % 3u;
      var grad = 0.0;
      for (var p = 0; p < 8; p++) {
        for (var q = 0; q < 8; q++) {
          if (sh_a[(p * 8 + q) * 8 + i32(o)] > 0.0) {
            let py = p + i32(u) - 1; let px = q + i32(v) - 1;
            if (py >= 0 && py < 8 && px >= 0 && px < 8) {
              grad += sh_bar_a1[(p * 8 + q) * 8 + i32(o)] * patch_e(sh_patch_state[u32(py) * 8u + u32(px)], c);
            }
          }
        }
      }
      apply_adam_f32(W_CONV1 + w_idx, grad);
    }
    workgroupBarrier();

    var pi = select(0u, sh_patch_state[lid], lid < 64u);
    var l_bar_patch0: array<f32, D>;
    let patch_p = i32(lid / 8u); let patch_q = i32(lid % 8u);
    for (var c = 0u; c < 8u; c++) {
      var grad = 0.0;
      for (var u = 0u; u < 3u; u++) {
        for (var v = 0u; v < 3u; v++) {
          let p_out = patch_p - (i32(u) - 1); let q_out = patch_q - (i32(v) - 1);
          if (p_out >= 0 && p_out < 8 && q_out >= 0 && q_out < 8) {
            for (var o = 0u; o < 8u; o++) {
              if (sh_a[(p_out * 8 + q_out) * 8 + i32(o)] > 0.0) {
                grad += sh_bar_a1[(p_out * 8 + q_out) * 8 + i32(o)] * c1w(u, v, c, o);
              }
            }
          }
        }
      }
      l_bar_patch0[c] = grad;
    }

    sh_pool[lid] = bitcast<f32>(pi);
    for(var c = 0u; c < 8u; c++) { sh_b[lid * 8u + c] = l_bar_patch0[c]; }
    workgroupBarrier();

    var is_first = true;
    for (var t = 0u; t < lid; t++) { if (bitcast<u32>(sh_pool[t]) == pi) { is_first = false; break; } }
    if (is_first) {
      var sum_grad: array<f32, D>;
      for(var c = 0u; c < 8u; c++) { sum_grad[c] = 0.0; }
      for (var t = lid; t < 64u; t++) {
        if (bitcast<u32>(sh_pool[t]) == pi) {
          for(var c = 0u; c < 8u; c++) { sum_grad[c] += sh_b[t * 8u + c]; }
        }
      }
      for (var c = 0u; c < 8u; c++) { apply_adam_f16(E_PATCH + pi * D + c, sum_grad[c]); }
    }
    workgroupBarrier();
  }
}

// --------------------------------------------------------------------------
// Entry: INIT BOARDS (Wall Generation)
// --------------------------------------------------------------------------
@compute @workgroup_size(WG)
fn init_boards(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) loc: vec3<u32>) {
  let b = wg.x;
  if (b >= params.batch_size) { return; }
  let lid = loc.x;

  if (lid < WPB) { boards[b].packed[lid] = 0u; }
  workgroupBarrier();

  if (lid == 0u) {
    boards[b].game_over = 0u;
    boards[b].player = 1u;
    boards[b].step_count = 0u;

    // Generate random walls
    let area = f32(N);
    let base_chains = 5.0 + trunc(rng_float(params.seed, 100u, b) * 8.0);
    let num_chains = u32(round((base_chains * area) / 100.0 * params.wall_density));
    
    let DR = array<i32, 4>(0, 1, 0, -1);
    let DC = array<i32, 4>(1, 0, -1, 0);

    var rng_state = params.seed ^ b;
    
    for (var chain = 0u; chain < num_chains; chain++) {
      rng_state = pcg(rng_state);
      var r = i32(rng_float(rng_state, chain, 0u) * f32(K));
      var c = i32(rng_float(rng_state, chain, 1u) * f32(K));
      let length = 1u + u32(rng_float(rng_state, chain, 2u) * 4.0);
      var dir = u32(rng_float(rng_state, chain, 3u) * 4.0);

      for (var seg = 0u; seg < length; seg++) {
        if (r < 0 || r >= i32(K) || c < 0 || c >= i32(K)) { break; }
        
        // Wall state is 3
        let cell = u32(r) * K + u32(c);
        let wi = cell >> 4u;
        let bo = (cell & 15u) << 1u;
        boards[b].packed[wi] = (boards[b].packed[wi] & ~(3u << bo)) | (3u << bo);
        
        r += DR[dir];
        c += DC[dir];
        
        if (rng_float(rng_state, chain, 4u + seg) < 0.3) {
          let turn = select(3u, 1u, rng_float(rng_state, chain, 5u + seg) < 0.5);
          dir = (dir + turn) % 4u;
        }
      }
    }
    
    // Clear center 3x3 for fair start
    let mid = i32(K / 2u);
    for (var dr = -1; dr <= 1; dr++) {
      for (var dc = -1; dc <= 1; dc++) {
        let rr = mid + dr;
        let cc = mid + dc;
        if (rr >= 0 && rr < i32(K) && cc >= 0 && cc < i32(K)) {
          let cell = u32(rr) * K + u32(cc);
          let wi = cell >> 4u;
          let bo = (cell & 15u) << 1u;
          boards[b].packed[wi] = boards[b].packed[wi] & ~(3u << bo);
        }
      }
    }
  }
}
