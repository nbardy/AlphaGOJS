// Convergence check: train one config for many generations and log the trajectory of the
// signals that actually indicate learning. The honest convergence metric is win-rate vs a
// FIXED anchor (a stationary reference) — self-play Elo is relative (moving yardstick) and
// not comparable across runs. We log anchor win-rate, entropy, loss, and (relative) Elo.
//
//   Usage:  ARCH=compact PPO_EPOCHS=1 GENS=300 bun bench/convergence.ts
//
// The kernel is Hogwild (nondeterministic at B>1), so two runs won't be bit-identical — but
// the TREND (does anchor win-rate climb and plateau? does entropy settle?) should reproduce.

import { setupGlobals } from "bun-webgpu";
import { installFakeIndexedDB } from "./idb_stub";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };
installFakeIndexedDB();

const { GPUTrainer, CONFIG } = await import("../src/gpu_harness");
const { LEAGUE_ARCHS } = await import("../src/arch_config");

const ARCH = process.env.ARCH ?? "compact";
const GENS = Number(process.env.GENS ?? 300);
if (process.env.PPO_EPOCHS) CONFIG.ppoEpochs = Number(process.env.PPO_EPOCHS);
const arch = LEAGUE_ARCHS.find((a: any) => a.name === ARCH) ?? LEAGUE_ARCHS[0];

const trainer = new GPUTrainer(arch);
let last: any = {};
trainer.onStats = (s: any) => { last = s; };
await trainer.init();

console.log(`\n=== convergence: ${arch.name} D=${arch.D} ppoEpochs=${CONFIG.ppoEpochs} B=${CONFIG.numBoards}, ${GENS} gens ===`);
console.log(`gen   elo    loss    entropy   anchorWR(last)`);

const anchorWRs: number[] = [];
let lastWR = -1;
for (let g = 1; g <= GENS; g++) {
  await trainer.runStep();
  if (last.winRate >= 0) { anchorWRs.push(last.winRate); lastWR = last.winRate; }
  if (g % 25 === 0 || g === 1) {
    console.log(`${String(g).padStart(4)}  ${(last.elo ?? 0).toFixed(0).padStart(5)}  ${(last.loss ?? 0).toFixed(3)}  ${(last.entropy ?? 0).toFixed(3).padStart(7)}   ${lastWR >= 0 ? lastWR.toFixed(2) : "--"}`);
  }
}

// Convergence verdict: compare the mean anchor win-rate of the first vs last third of evals.
const third = Math.max(1, Math.floor(anchorWRs.length / 3));
const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
const early = mean(anchorWRs.slice(0, third));
const lateArr = anchorWRs.slice(-third);
const late = mean(lateArr);
const lateStd = Math.sqrt(mean(lateArr.map((x) => (x - late) ** 2)));
console.log(`\n--- anchor win-rate (vs fixed early checkpoint) ---`);
console.log(`  evals recorded: ${anchorWRs.length}`);
console.log(`  early third mean: ${early.toFixed(2)}   late third mean: ${late.toFixed(2)} ± ${lateStd.toFixed(2)}`);
console.log(`  trend: ${late > early + 0.05 ? "CLIMBING (learning)" : late < early - 0.05 ? "FALLING (regressing!)" : "FLAT (plateaued or not moving)"}`);
console.log(`\nCONV_JSON ${JSON.stringify({ arch: arch.name, D: arch.D, ep: CONFIG.ppoEpochs, gens: GENS, eloFinal: +(last.elo ?? 0).toFixed(0), lossFinal: +(last.loss ?? 0).toFixed(3), entropyFinal: +(last.entropy ?? 0).toFixed(3), anchorWR_early: +early.toFixed(2), anchorWR_late: +late.toFixed(2), anchorWR_lateStd: +lateStd.toFixed(2) })}`);
process.exit(0);
