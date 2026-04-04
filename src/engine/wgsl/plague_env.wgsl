// WebGPU plague env (unpacked u32 per cell: 0 empty, 1 P1, 2 P2, 3 wall).
// `spread_pass` is the canonical spread kernel — shared with spread-only benches / CPU parity
// (plague_spread_cpu.js). Apply + terminal passes support WebGPUGameEngine in the worker.

struct GameConfig {
  rows: u32,
  cols: u32,
  tick: u32,
  num_games: u32,
}

struct ApplyConfig {
  rows: u32,
  cols: u32,
  num_games: u32,
  player_cell: u32,
}

@group(0) @binding(0) var<uniform> spread_config: GameConfig;
@group(0) @binding(1) var<storage, read> spread_board_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> spread_board_out: array<u32>;

@group(1) @binding(0) var<uniform> apply_config: ApplyConfig;
@group(1) @binding(1) var<storage, read> apply_board_in: array<u32>;
@group(1) @binding(2) var<storage, read_write> apply_board_out: array<u32>;
@group(1) @binding(3) var<storage, read> slot_actions: array<i32>;
@group(1) @binding(4) var<storage, read> slot_active: array<u32>;

@group(2) @binding(0) var<uniform> term_config: GameConfig;
@group(2) @binding(1) var<storage, read> term_board: array<u32>;
@group(2) @binding(2) var<storage, read_write> term_counts: array<u32>;

const EMPTY: u32 = 0u;
const P1: u32 = 1u;
const P2: u32 = 2u;
const WALL: u32 = 3u;

fn hash_u32(x: u32) -> u32 {
  var v = x;
  v = v ^ (v >> 16u);
  v = v * 0x7feb352du;
  v = v ^ (v >> 15u);
  v = v * 0x846ca68bu;
  v = v ^ (v >> 16u);
  return v;
}

fn rand01(game_id: u32, tick: u32, local_idx: u32, dir: u32) -> f32 {
  let mix = (game_id * 0x9e3779b9u)
    ^ (tick * 0x85ebca6bu)
    ^ (local_idx * 0xc2b2ae35u)
    ^ (dir * 0x27d4eb2du);
  let h = hash_u32(mix);
  return f32(h & 0xffffffu) * (1.0 / f32(0x1000000u));
}

fn get_neighbor_spread(board_base: u32, r: i32, c: i32) -> u32 {
  if (r < 0 || r >= i32(spread_config.rows) || c < 0 || c >= i32(spread_config.cols)) {
    return WALL;
  }
  let idx = u32(r) * spread_config.cols + u32(c);
  return spread_board_in[board_base + idx];
}

@compute @workgroup_size(64)
fn spread_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let board_size = spread_config.rows * spread_config.cols;
  let total = board_size * spread_config.num_games;
  let g = gid.x;
  if (g >= total) {
    return;
  }

  let game_id = g / board_size;
  let local_idx = g - game_id * board_size;
  let base = game_id * board_size;

  let current = spread_board_in[base + local_idx];
  if (current != EMPTY) {
    spread_board_out[base + local_idx] = current;
    return;
  }

  let r = i32(local_idx / spread_config.cols);
  let c = i32(local_idx % spread_config.cols);

  var sum: f32 = 0.0;
  let n0 = get_neighbor_spread(base, r - 1, c);
  if (n0 == P1) { sum = sum + rand01(game_id, spread_config.tick, local_idx, 0u); }
  else if (n0 == P2) { sum = sum - rand01(game_id, spread_config.tick, local_idx, 0u); }

  let n1 = get_neighbor_spread(base, r + 1, c);
  if (n1 == P1) { sum = sum + rand01(game_id, spread_config.tick, local_idx, 1u); }
  else if (n1 == P2) { sum = sum - rand01(game_id, spread_config.tick, local_idx, 1u); }

  let n2 = get_neighbor_spread(base, r, c - 1);
  if (n2 == P1) { sum = sum + rand01(game_id, spread_config.tick, local_idx, 2u); }
  else if (n2 == P2) { sum = sum - rand01(game_id, spread_config.tick, local_idx, 2u); }

  let n3 = get_neighbor_spread(base, r, c + 1);
  if (n3 == P1) { sum = sum + rand01(game_id, spread_config.tick, local_idx, 3u); }
  else if (n3 == P2) { sum = sum - rand01(game_id, spread_config.tick, local_idx, 3u); }

  let t = trunc(sum * 2.0);
  let cl = clamp(t, -1.0, 1.0);
  var out_v = EMPTY;
  if (cl > 0.0) { out_v = P1; }
  else if (cl < 0.0) { out_v = P2; }

  spread_board_out[base + local_idx] = out_v;
}

@compute @workgroup_size(64)
fn apply_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let board_size = apply_config.rows * apply_config.cols;
  let total = board_size * apply_config.num_games;
  let g = gid.x;
  if (g >= total) {
    return;
  }

  let game_id = g / board_size;
  let local_idx = g - game_id * board_size;
  let cur = apply_board_in[g];

  if (slot_active[game_id] == 0u) {
    apply_board_out[g] = cur;
    return;
  }

  let act = slot_actions[game_id];
  if (i32(local_idx) != act) {
    apply_board_out[g] = cur;
    return;
  }

  if (cur != EMPTY) {
    apply_board_out[g] = cur;
    return;
  }

  apply_board_out[g] = apply_config.player_cell;
}

@compute @workgroup_size(64)
fn terminal_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x;
  if (g >= term_config.num_games) {
    return;
  }

  let board_size = term_config.rows * term_config.cols;
  let base = g * board_size;
  var empty_c: u32 = 0u;
  var p1_c: u32 = 0u;
  var p2_c: u32 = 0u;

  for (var i = 0u; i < board_size; i++) {
    let c = term_board[base + i];
    if (c == EMPTY) {
      empty_c = empty_c + 1u;
    } else if (c == P1) {
      p1_c = p1_c + 1u;
    } else if (c == P2) {
      p2_c = p2_c + 1u;
    }
  }

  let o = g * 3u;
  term_counts[o + 0u] = empty_c;
  term_counts[o + 1u] = p1_c;
  term_counts[o + 2u] = p2_c;
}
