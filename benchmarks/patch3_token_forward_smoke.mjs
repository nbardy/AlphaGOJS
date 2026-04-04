#!/usr/bin/env node
/**
 * TF.js smoke + forward throughput for `patch3_token` (non-overlapping 3×3 patch tokens).
 * Same protocol as patch3_discrete_forward_smoke.mjs; input width is hp*wp*9, not boardSize.
 *
 *   npm run bench:patch3-token-smoke
 *   node benchmarks/patch3_token_forward_smoke.mjs --rows=20 --cols=20 --batch=64 --iters=400
 */
import * as tf from '@tensorflow/tfjs';
import { Patch3TokenDiscreteModel } from '../src/patch3_token_discrete_model.js';
import { statesRowsToModelInputTensor } from '../src/action.js';

function clampInt(v, fallback, min, max) {
  const n = Number.parseInt(v, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function parseArgs(argv) {
  const out = { rows: 10, cols: 10, batch: 32, iters: 500, warmup: 30, game: true };
  for (const arg of argv) {
    if (!arg.startsWith('--')) continue;
    const [key, value = ''] = arg.slice(2).split('=', 2);
    if (key === 'rows') out.rows = clampInt(value, out.rows, 4, 32);
    else if (key === 'cols') out.cols = clampInt(value, out.cols, 4, 32);
    else if (key === 'batch') out.batch = clampInt(value, out.batch, 1, 256);
    else if (key === 'iters') out.iters = clampInt(value, out.iters, 1, 100000);
    else if (key === 'warmup') out.warmup = clampInt(value, out.warmup, 0, 10000);
    else if (key === 'game') out.game = value !== 'false' && value !== '0';
  }
  return out;
}

function assertFinitePolicy(policy, boardSize, label) {
  const d = policy.dataSync();
  for (let i = 0; i < Math.min(d.length, boardSize * 4); i++) {
    if (!Number.isFinite(d[i])) throw new Error(label + ': non-finite logit at ' + i);
  }
}

function randomRawBoard(boardSize) {
  const a = new Int32Array(boardSize);
  for (let i = 0; i < boardSize; i++) a[i] = (Math.random() * 4) | 0;
  return a;
}

async function main() {
  const cfg = parseArgs(process.argv.slice(2));

  await tf.setBackend('cpu');
  await tf.ready();

  const model = new Patch3TokenDiscreteModel(cfg.rows, cfg.cols);
  const boardSize = model.boardSize;

  console.log(
    `[patch3-token-smoke] cpu | board ${cfg.rows}×${cfg.cols} (${boardSize} cells) | hp×wp ${model.hp}×${model.wp} patchInputLen=${model.patchInputLen} jointVocab=${model.jointVocab}`
  );

  const batchRows = [];
  for (let b = 0; b < cfg.batch; b++) batchRows.push(randomRawBoard(boardSize));

  for (let w = 0; w < cfg.warmup; w++) {
    const t = statesRowsToModelInputTensor(model, batchRows, cfg.batch);
    const o = model.forward(t);
    assertFinitePolicy(o.policy, boardSize, 'warmup');
    o.policy.dispose();
    o.value.dispose();
    t.dispose();
  }

  const t0 = performance.now();
  for (let i = 0; i < cfg.iters; i++) {
    for (let b = 0; b < cfg.batch; b++) {
      batchRows[b] = randomRawBoard(boardSize);
    }
    const t = statesRowsToModelInputTensor(model, batchRows, cfg.batch);
    const o = model.forward(t);
    o.policy.dispose();
    o.value.dispose();
    t.dispose();
  }
  const elapsed = (performance.now() - t0) / 1000;
  const forwards = cfg.iters * cfg.batch;
  const fps = forwards / elapsed;
  const msPer = (elapsed * 1000) / forwards;
  console.log(
    `[patch3-token-smoke] batched forward: ${forwards} passes (batch=${cfg.batch} × iters=${cfg.iters}) in ${elapsed.toFixed(3)}s → ${fps.toFixed(1)} forwards/s (${msPer.toFixed(3)} ms/forward)`
  );

  if (cfg.game) {
    const codes = new Int32Array(boardSize);
    codes.fill(0);
    let steps = 0;
    const maxSteps = boardSize + 50;
    while (steps < maxSteps) {
      const empties = [];
      for (let i = 0; i < boardSize; i++) if (codes[i] === 0) empties.push(i);
      if (empties.length === 0) break;

      const t1 = statesRowsToModelInputTensor(model, [codes], 1);
      const o1 = model.forward(t1);
      assertFinitePolicy(o1.policy, boardSize, 'game step ' + steps);
      o1.policy.dispose();
      o1.value.dispose();
      t1.dispose();

      const pick = empties[(Math.random() * empties.length) | 0];
      codes[pick] = 1;
      steps++;
    }
    if (steps !== boardSize) {
      throw new Error('expected ' + boardSize + ' place steps, got ' + steps);
    }
    console.log('[patch3-token-smoke] synthetic fill-game: %d forward passes OK', steps);
  }

  console.log('[patch3-token-smoke] done.');
}

main().catch((e) => {
  console.error('[patch3-token-smoke] FAIL:', e && e.message ? e.message : e);
  process.exit(1);
});
