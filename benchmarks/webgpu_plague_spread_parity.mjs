#!/usr/bin/env node
/**
 * Verifies GPU spread matches CPU reference (same hash RNG as plague_env.wgsl spread_pass).
 * Requires WebGPU (e.g. `webgpu` Dawn package): npm run bench:webgpu:parity
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { initWebGPUDevice } from './webgpu_node_helpers.mjs';
import { WebGPUPlagueSpreadEngine } from '../src/engine/webgpu_plague_spread_engine.js';
import { spreadPackedAllGames } from '../src/engine/plague_spread_cpu.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WGSL_PATH = join(__dirname, '../src/engine/wgsl/plague_env.wgsl');

function assertEqual(a, b, label) {
  if (a.length !== b.length) throw new Error(label + ': length ' + a.length + ' vs ' + b.length);
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      throw new Error(label + ': mismatch at ' + i + ' gpu=' + a[i] + ' cpu=' + b[i]);
    }
  }
}

function makePacked(rows, cols, fill) {
  const n = rows * cols;
  const out = new Uint32Array(n);
  out.fill(fill);
  return out;
}

async function main() {
  const wgsl = readFileSync(WGSL_PATH, 'utf8');
  const { ok, device } = await initWebGPUDevice();
  if (!ok || !device) {
    throw new Error(
      'No WebGPU adapter (needs GPU / Dawn). Try: npm run bench:webgpu:node env, or a WebGPU-capable browser.'
    );
  }

  const rows = 5;
  const cols = 5;
  const numGames = 3;
  const boardSize = rows * cols;

  const engine = new WebGPUPlagueSpreadEngine(device, { rows, cols, numGames }, wgsl);

  try {
    const packed = new Uint32Array(boardSize * numGames);
    packed.set(makePacked(rows, cols, 0), 0);
    packed.set(makePacked(rows, cols, 1), boardSize);
    for (let i = 0; i < boardSize; i++) packed[boardSize * 2 + i] = i % 4;

    for (let round = 0; round < 5; round++) {
      engine.uploadPacked(packed);
      const tick = round;
      const want = spreadPackedAllGames(packed, rows, cols, numGames, tick);
      await engine.spreadAndSync();
      const got = await engine.downloadPacked();
      assertEqual(got, want, 'round ' + round);
      packed.set(got);
    }

    console.log('webgpu_plague_spread_parity: OK (5 rounds, ' + numGames + ' games, ' + rows + 'x' + cols + ')');
  } finally {
    engine.dispose();
  }
}

main().catch((e) => {
  console.error(e.message || e);
  process.exitCode = 1;
});
