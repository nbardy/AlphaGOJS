// 2-bit packed spread: 16 cells per u32, flat row-major cell index i → word i/16, shift (i%16)*2.
// Stochastic rules + RNG match plague_env.wgsl `spread_pass` (hash_u32 + rand01).

struct GameConfig {
  rows: u32,
  cols: u32,
  tick: u32,
  num_games: u32,
}

@group(0) @binding(0) var<uniform> cfg: GameConfig;
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

fn rand01(game_id: u32, tick: u32, local_idx: u32, dir: u32) -> f32 {
  let mix = (game_id * 0x9e3779b9u)
    ^ (tick * 0x85ebca6bu)
    ^ (local_idx * 0xc2b2ae35u)
    ^ (dir * 0x27d4eb2du);
  let h = hash_u32(mix);
  return f32(h & 0xffffffu) * (1.0 / f32(0x1000000u));
}

fn board_size() -> u32 {
  return cfg.rows * cfg.cols;
}

fn words_per_game() -> u32 {
  let bs = board_size();
  return (bs + 15u) / 16u;
}

fn read_code(words_base: u32, li: u32, bs: u32) -> u32 {
  if (li >= bs) {
    return EMPTY;
  }
  let wi = words_base + li / 16u;
  let sh = (li % 16u) * 2u;
  return (board_in[wi] >> sh) & 3u;
}

@compute @workgroup_size(64)
fn spread_packed_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let bs = board_size();
  let wpg = words_per_game();
  let total_words = wpg * cfg.num_games;
  let gw = gid.x;
  if (gw >= total_words) {
    return;
  }

  let game_id = gw / wpg;
  let word_in_game = gw % wpg;
  let words_base = game_id * wpg;

  let old_word = board_in[words_base + word_in_game];
  let combined = (old_word | (old_word >> 1u)) & 0x55555555u;
  if (combined == 0x55555555u) {
    board_out[words_base + word_in_game] = old_word;
    return;
  }

  var new_word: u32 = 0u;
  for (var i = 0u; i < 16u; i++) {
    let li = word_in_game * 16u + i;
    if (li >= bs) {
      new_word = new_word | (old_word & (3u << (i * 2u)));
      continue;
    }

    let code = (old_word >> (i * 2u)) & 3u;
    if (code != EMPTY) {
      new_word = new_word | (code << (i * 2u));
      continue;
    }

    let r = li / cfg.cols;
    let c = li % cfg.cols;
    var sum: f32 = 0.0;

    if (r > 0u) {
      let n = read_code(words_base, li - cfg.cols, bs);
      if (n == P1) { sum = sum + rand01(game_id, cfg.tick, li, 0u); }
      else if (n == P2) { sum = sum - rand01(game_id, cfg.tick, li, 0u); }
    }

    if (r + 1u < cfg.rows) {
      let n = read_code(words_base, li + cfg.cols, bs);
      if (n == P1) { sum = sum + rand01(game_id, cfg.tick, li, 1u); }
      else if (n == P2) { sum = sum - rand01(game_id, cfg.tick, li, 1u); }
    }

    if (c > 0u) {
      let n = read_code(words_base, li - 1u, bs);
      if (n == P1) { sum = sum + rand01(game_id, cfg.tick, li, 2u); }
      else if (n == P2) { sum = sum - rand01(game_id, cfg.tick, li, 2u); }
    }

    if (c + 1u < cfg.cols) {
      let n = read_code(words_base, li + 1u, bs);
      if (n == P1) { sum = sum + rand01(game_id, cfg.tick, li, 3u); }
      else if (n == P2) { sum = sum - rand01(game_id, cfg.tick, li, 3u); }
    }

    let t = trunc(sum * 2.0);
    let cl = clamp(t, -1.0, 1.0);
    var out_c = EMPTY;
    if (cl > 0.0) { out_c = P1; }
    else if (cl < 0.0) { out_c = P2; }

    new_word = new_word | (out_c << (i * 2u));
  }

  board_out[words_base + word_in_game] = new_word;
}
