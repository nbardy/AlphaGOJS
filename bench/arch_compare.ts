// Equal-generations architecture comparison: train D=4/8/16 for the SAME number of
// generations, then measure (a) wall-clock per generation and (b) learned quality.
//
// Quality is measured two ways:
//   - within-arch signals: loss, policy entropy, self-play Elo (NOTE: Elo is a SEPARATE
//     self-play ladder per arch → NOT directly comparable across archs).
//   - cross-arch HEAD-TO-HEAD: play the final trained weights of each arch against each
//     other (eval_harness.evalArchitectures, CPU JS inference). THIS is the apples-to-
//     apples expressiveness test — does a bigger D actually play better after equal training?
//
//   Usage:  GENS=50 EVAL_GAMES=20 bun bench/arch_compare.ts

import { setupGlobals } from "bun-webgpu";
import { installFakeIndexedDB } from "./idb_stub";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const { GPUTrainer } = await import("../src/gpu_harness");
const { LEAGUE_ARCHS } = await import("../src/arch_config");
const { evalArchitectures } = await import("../src/eval_harness");

const GENS = Number(process.env.GENS ?? 50);
const EVAL_GAMES = Number(process.env.EVAL_GAMES ?? 20);
const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);

interface Trained {
  arch: any;
  totalMs: number;
  msPerGen: number;
  lossFirst: number; lossLast: number;
  entFirst: number; entLast: number;
  elo: number;
  anchorWinRates: number[];
  weights: { dense: Float32Array; embed: Float16Array };
}

const trained: Trained[] = [];

for (const arch of LEAGUE_ARCHS) {
  installFakeIndexedDB(); // fresh empty checkpoint DB per arch (no cross-arch leakage)
  const trainer = new GPUTrainer(arch);
  let last: any = {};
  trainer.onStats = (s: any) => { last = s; };
  await trainer.init();

  const perGenMs: number[] = [];
  const losses: number[] = [];
  const ents: number[] = [];
  const anchorWinRates: number[] = [];

  console.log(`\n[${arch.name} D=${arch.D}] training ${GENS} generations...`);
  const t0 = performance.now();
  for (let g = 0; g < GENS; g++) {
    const a = performance.now();
    await trainer.runStep();
    perGenMs.push(performance.now() - a);
    losses.push(last.loss);
    ents.push(last.entropy);
    if (last.winRate >= 0) anchorWinRates.push(last.winRate);
  }
  const totalMs = performance.now() - t0;

  trained.push({
    arch,
    totalMs,
    msPerGen: mean(perGenMs),
    lossFirst: losses[0], lossLast: losses[losses.length - 1],
    entFirst: ents[0], entLast: ents[ents.length - 1],
    elo: last.elo,
    anchorWinRates,
    weights: await trainer.readWeights(),
  });
  console.log(`  done in ${(totalMs / 1000).toFixed(1)}s (${mean(perGenMs).toFixed(0)} ms/gen)`);
}

console.log(`\n================  EQUAL-GENERATIONS COMPARISON (GENS=${GENS})  ================`);
console.log(`arch      D   total(s)  ms/gen  games/s  loss(first→last)   entropy(first→last)  selfElo  anchorWR(last)`);
for (const t of trained) {
  const gps = (256 / t.msPerGen) * 1000;
  const awr = t.anchorWinRates.length ? t.anchorWinRates[t.anchorWinRates.length - 1].toFixed(2) : "n/a";
  console.log(
    `${t.arch.name.padEnd(9)} ${String(t.arch.D).padEnd(3)} ` +
    `${(t.totalMs / 1000).toFixed(1).padStart(7)} ${t.msPerGen.toFixed(0).padStart(7)} ${gps.toFixed(0).padStart(8)}  ` +
    `${t.lossFirst?.toFixed(3)}→${t.lossLast?.toFixed(3)}`.padEnd(18) +
    `   ${t.entFirst?.toFixed(2)}→${t.entLast?.toFixed(2)}`.padEnd(20) +
    `  ${t.elo?.toFixed(0).padStart(5)}   ${awr}`
  );
}

console.log(`\n================  CROSS-ARCH HEAD-TO-HEAD (${EVAL_GAMES} games each, final weights)  ================`);
console.log(`These ARE comparable: each arch's trained policy plays the others (CPU inference, sides swapped for fairness).`);
const baseSeed = 0xC0FFEE;
for (let i = 0; i < trained.length; i++) {
  for (let j = i + 1; j < trained.length; j++) {
    const A = trained[i], B = trained[j];
    const r = evalArchitectures(A.arch, A.weights, B.arch, B.weights, EVAL_GAMES, baseSeed);
    const decisive = r.winsA + r.winsB;
    const aPct = decisive ? ((r.winsA / decisive) * 100).toFixed(0) : "50";
    console.log(`  ${A.arch.name}(D=${A.arch.D}) vs ${B.arch.name}(D=${B.arch.D}): ${r.winsA}-${r.winsB} (draws ${r.draws})  →  ${A.arch.name} wins ${aPct}% of decisive`);
  }
}

console.log(`\nSUMMARY_JSON ${JSON.stringify(trained.map(t => ({
  arch: t.arch.name, D: t.arch.D,
  msPerGen: +t.msPerGen.toFixed(0),
  gamesPerSec: +((256 / t.msPerGen) * 1000).toFixed(1),
  lossLast: +(t.lossLast ?? 0).toFixed(3),
  entLast: +(t.entLast ?? 0).toFixed(3),
  selfElo: +(t.elo ?? 0).toFixed(0),
})))}`);

process.exit(0);
