// Equal-WALL-CLOCK config validation. The earlier comparison fixed generations, which
// is unfair (D=4 is far cheaper per gen). The decision-relevant question is: for a FIXED
// COMPUTE BUDGET, which config learns the best policy? So we give each config the same
// seconds, let it run as many generations as it can, then play a LARGER head-to-head
// (more games → tighter statistics) with the final weights.
//
//   Usage:  TIME_BUDGET=45 EVAL_GAMES=60 bun bench/validate_config.ts
//
// Also records the entropy/loss/anchor-win-rate trajectory so we can see whether the
// fast config's policy stays healthy (no late collapse) over a long run.

import { setupGlobals } from "bun-webgpu";
import { installFakeIndexedDB } from "./idb_stub";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const { GPUTrainer, CONFIG } = await import("../src/gpu_harness");
const { LEAGUE_ARCHS } = await import("../src/arch_config");
const { evalArchitectures } = await import("../src/eval_harness");

const TIME_BUDGET = Number(process.env.TIME_BUDGET ?? 45) * 1000; // ms per config
const EVAL_GAMES = Number(process.env.EVAL_GAMES ?? 60);

const CONFIGS = [
  { name: "D8_ep3", archName: "standard", epochs: 3 }, // current default
  { name: "D4_ep3", archName: "compact",  epochs: 3 },
  { name: "D4_ep1", archName: "compact",  epochs: 1 }, // the ~14.5x candidate
];

interface Trained { name: string; archName: string; arch: any; gens: number; entLast: number; lossLast: number; anchorWR: number; entTraj: number[]; weights: { dense: Float32Array; embed: Float16Array }; }
const trained: Trained[] = [];

for (const cfg of CONFIGS) {
  const arch = LEAGUE_ARCHS.find((a: any) => a.name === cfg.archName)!;
  CONFIG.ppoEpochs = cfg.epochs;
  CONFIG.numBoards = 256;
  installFakeIndexedDB();
  const trainer = new GPUTrainer(arch);
  let last: any = {};
  trainer.onStats = (s: any) => { last = s; };
  await trainer.init();

  console.log(`\n[${cfg.name}] training for ${TIME_BUDGET / 1000}s (D=${arch.D}, ppoEpochs=${cfg.epochs})...`);
  const entTraj: number[] = [];
  const anchorWRs: number[] = [];
  let gens = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < TIME_BUDGET) {
    await trainer.runStep();
    gens++;
    if (gens % 25 === 0) entTraj.push(+(last.entropy ?? 0).toFixed(2));
    if (last.winRate >= 0) anchorWRs.push(last.winRate);
  }
  const anchorWR = anchorWRs.length ? anchorWRs[anchorWRs.length - 1] : -1;
  console.log(`  ${gens} generations | entropy=${(last.entropy ?? 0).toFixed(2)} loss=${(last.loss ?? 0).toFixed(3)} anchorWR=${anchorWR.toFixed(2)}`);
  trained.push({ name: cfg.name, archName: cfg.archName, arch, gens, entLast: last.entropy, lossLast: last.loss, anchorWR, entTraj, weights: await trainer.readWeights() });
}

console.log(`\n================  EQUAL WALL-CLOCK (${TIME_BUDGET / 1000}s each)  ================`);
console.log(`config    D   gens   entLast  lossLast  anchorWR   entropy-trajectory(every 25 gens)`);
for (const t of trained) {
  console.log(`${t.name.padEnd(9)} ${String(t.arch.D).padEnd(3)} ${String(t.gens).padStart(5)}  ${(t.entLast ?? 0).toFixed(2).padStart(7)}  ${(t.lossLast ?? 0).toFixed(3).padStart(8)}  ${(t.anchorWR ?? 0).toFixed(2).padStart(7)}   [${t.entTraj.join(", ")}]`);
}

console.log(`\n================  HEAD-TO-HEAD (${EVAL_GAMES} games/pair, final weights)  ================`);
// For n games, a 50/50 true rate has std ~ 0.5/sqrt(n); flag when a result clears ~2 sigma.
const sigma = 0.5 / Math.sqrt(EVAL_GAMES);
console.log(`(2-sigma significance band for ${EVAL_GAMES} games: a win rate must exceed ${(50 + 200 * sigma).toFixed(0)}% to be > noise)`);
const seed = 0xBADC0DE;
for (let i = 0; i < trained.length; i++) {
  for (let j = i + 1; j < trained.length; j++) {
    const A = trained[i], B = trained[j];
    const r = evalArchitectures(A.arch, A.weights, B.arch, B.weights, EVAL_GAMES, seed);
    const dec = r.winsA + r.winsB;
    const aPct = dec ? (r.winsA / dec) * 100 : 50;
    const sig = Math.abs(aPct - 50) > 200 * sigma ? "  *significant*" : "  (within noise)";
    console.log(`  ${A.name} vs ${B.name}: ${r.winsA}-${r.winsB} (draws ${r.draws})  →  ${A.name} ${aPct.toFixed(0)}%${sig}`);
  }
}

console.log(`\nVALIDATE_JSON ${JSON.stringify(trained.map((t) => ({ name: t.name, D: t.arch.D, gens: t.gens, entLast: +(t.entLast ?? 0).toFixed(2), lossLast: +(t.lossLast ?? 0).toFixed(3), anchorWR: +(t.anchorWR ?? 0).toFixed(2) })))}`);

process.exit(0);
