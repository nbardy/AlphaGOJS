#!/usr/bin/env node
/**
 * Verifies **2-bit packed** WGSL spread vs CPU reference (same hash RNG as unpacked spread_pass).
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { initWebGPUDevice } from './webgpu_node_helpers.mjs';
import { WebGPUPlagueSpreadPackedEngine } from '../src/engine/webgpu_plague_spread_packed_engine.js';
import { packAllGamesUint32To2Bit } from '../src/engine/plague_spread_pack_2bit.js';
import { spreadPacked2BitWordsInOut } from '../src/engine/plague_spread_cpu.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const WGSL_PATH = join(__dirname, '../src/engine/wgsl/plague_spread_packed.wgsl');

function assertEqual(a, b, label) {
  if (a.length !== b.length) throw new Error(label + ': length ' + a.length + ' vs ' + b.length);
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      throw new Error(label + ': mismatch at word ' + i + ' gpu=' + a[i] + ' cpu=' + b[i]);
    }
  }
}

function makeUnpackedGame(rows, cols, fill) {
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

  const engine = new WebGPUPlagueSpreadPackedEngine(device, { rows, cols, numGames }, wgsl);

  try {
    const flat = new Uint32Array(boardSize * numGames);
    flat.set(makeUnpackedGame(rows, cols, 0), 0);
    flat.set(makeUnpackedGame(rows, cols, 1), boardSize);
    for (let i = 0; i < boardSize; i++) {
      flat[boardSize * 2 + i] = i % 4;
    }

    let packed = packAllGamesUint32To2Bit(flat, rows, cols, numGames);
    const wantBuf = new Uint32Array(packed.length);

    for (let round = 0; round < 5; round++) {
      const tick = round;
      engine.uploadPackedWords(packed);
      spreadPacked2BitWordsInOut(packed, wantBuf, rows, cols, numGames, tick);
      await engine.spreadAndSync();
      const got = await engine.downloadPackedWords();
      assertEqual(got, wantBuf, 'round ' + round);
      packed = Uint32Array.from(got);
    }

    console.log(
      'webgpu_plague_spread_parity_packed: OK (5 rounds, ' + numGames + ' games, ' + rows + 'x' + cols + ')'
    );
  } finally {
    engine.dispose();
  }
}

main().catch((e) => {
  console.error(e.message || e);
  process.exitCode = 1;
});
