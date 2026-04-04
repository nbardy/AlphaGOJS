// Plague spread — one compute pass, ping-pong buffers (read board_in, write board_out).
// Semantics: match src/games/plague_walls.js spread (orthogonal neighbors, per-neighbor U(0,1),
// walls block, edges block). Cell packing (unpacked u32 lane per cell):
//   0 = empty, 1 = P1, 2 = P2, 3 = wall
//
// RNG: stateless mix hash on (game_id, tick, local_cell_index, dir_index) so each neighbor
// draw is independent without atomics (see docs/WEBGPU_PLAGUE_GAME_SPEC.md).

struct GameConfig {
  rows: u32,
  cols: u32,
  tick: u32,
  num_games: u32,
}

@group(0) @binding(0) var<uniform> config: GameConfig;
@group(0) @binding(1) var<storage, read> board_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> board_out: array<u32>;

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

// ~ U(0, 1) using top 24 bits (stable in f32)
fn rand01(game_id: u32, tick: u32, local_idx: u32, dir: u32) -> f32 {
  // WGSL: * and ^ cannot mix without explicit parentheses.
  let mix = (game_id * 0x9e3779b9u)
    ^ (tick * 0x85ebca6bu)
    ^ (local_idx * 0xc2b2ae35u)
    ^ (dir * 0x27d4eb2du);
  let h = hash_u32(mix);
  return f32(h & 0xffffffu) * (1.0 / f32(0x1000000u));
}

fn get_neighbor(board_base: u32, r: i32, c: i32) -> u32 {
  if (r < 0 || r >= i32(config.rows) || c < 0 || c >= i32(config.cols)) {
    return WALL;
  }
  let idx = u32(r) * config.cols + u32(c);
  return board_in[board_base + idx];
}

@compute @workgroup_size(64)
fn spread_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let board_size = config.rows * config.cols;
  let total = board_size * config.num_games;
  let g = gid.x;
  if (g >= total) {
    return;
  }

  let game_id = g / board_size;
  let local_idx = g - game_id * board_size;
  let base = game_id * board_size;

  let current = board_in[base + local_idx];
  if (current != EMPTY) {
    board_out[base + local_idx] = current;
    return;
  }

  let r = i32(local_idx / config.cols);
  let c = i32(local_idx % config.cols);

  var sum: f32 = 0.0;
  // Order matches plague_walls.js: up, down, left, right → dir 0..3
  let n0 = get_neighbor(base, r - 1, c);
  if (n0 == P1) { sum = sum + rand01(game_id, config.tick, local_idx, 0u); }
  else if (n0 == P2) { sum = sum - rand01(game_id, config.tick, local_idx, 0u); }

  let n1 = get_neighbor(base, r + 1, c);
  if (n1 == P1) { sum = sum + rand01(game_id, config.tick, local_idx, 1u); }
  else if (n1 == P2) { sum = sum - rand01(game_id, config.tick, local_idx, 1u); }

  let n2 = get_neighbor(base, r, c - 1);
  if (n2 == P1) { sum = sum + rand01(game_id, config.tick, local_idx, 2u); }
  else if (n2 == P2) { sum = sum - rand01(game_id, config.tick, local_idx, 2u); }

  let n3 = get_neighbor(base, r, c + 1);
  if (n3 == P1) { sum = sum + rand01(game_id, config.tick, local_idx, 3u); }
  else if (n3 == P2) { sum = sum - rand01(game_id, config.tick, local_idx, 3u); }

  let t = trunc(sum * 2.0);
  let cl = clamp(t, -1.0, 1.0);
  var out_v = EMPTY;
  if (cl > 0.0) { out_v = P1; }
  else if (cl < 0.0) { out_v = P2; }

  board_out[base + local_idx] = out_v;
}
