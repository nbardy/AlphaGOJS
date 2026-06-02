// Flaggable tuning sweep: runs the real GPUTrainer across a matrix of the SAFE
// (non-kernel-numeric) levers — D (arch) × batch × ppoEpochs — and prints a
// throughput + training-sanity table. Use it to find the speed/quality knee for a
// given target before reaching for the riskier kernel flags (vec4 / fp16 accumulate).
//
//   Usage:  bun bench/sweep.ts
//   Env:    ARCHS=compact,standard  BATCHES=256,512  EPOCHS=1,3  STEPS=8 WARMUP=2
//
// NOTE: run this when the GPU is otherwise idle — a second concurrent WebGPU process
// shares the device and will skew the wall-clock timings.

import { setupGlobals } from "bun-webgpu";
import { installFakeIndexedDB } from "./idb_stub";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const { GPUTrainer, CONFIG } = await import("../src/gpu_harness");
const { LEAGUE_ARCHS } = await import("../src/arch_config");

const parseList = (s: string | undefined, def: string) => (s ?? def).split(",").map((x) => x.trim());
const ARCHS = parseList(process.env.ARCHS, "compact,standard");
const BATCHES = parseList(process.env.BATCHES, "256").map(Number);
const EPOCHS = parseList(process.env.EPOCHS, "1,3").map(Number);
const STEPS = Number(process.env.STEPS ?? 8);
const WARMUP = Number(process.env.WARMUP ?? 2);
const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);

interface Row { arch: string; D: number; B: number; epochs: number; gamesPerSec: number; boardStepsPerSec: number; msPerStep: number; entLast: number; lossLast: number; }
const rows: Row[] = [];

for (const archName of ARCHS) {
  const arch = LEAGUE_ARCHS.find((a: any) => a.name === archName);
  if (!arch) { console.log(`(skip unknown arch ${archName})`); continue; }
  for (const B of BATCHES) {
    for (const epochs of EPOCHS) {
      CONFIG.numBoards = B;
      CONFIG.ppoEpochs = epochs;
      installFakeIndexedDB();
      const trainer = new GPUTrainer(arch);
      let last: any = {};
      trainer.onStats = (s: any) => { last = s; };
      await trainer.init();
      for (let i = 0; i < WARMUP; i++) await trainer.runStep();
      const ms: number[] = [];
      for (let i = 0; i < STEPS; i++) {
        const a = performance.now();
        await trainer.runStep();
        ms.push(performance.now() - a);
      }
      const msPerStep = mean(ms);
      const gamesPerSec = (B / msPerStep) * 1000;
      const boardStepsPerSec = gamesPerSec * (last.avgStepsPerGame ?? 22);
      rows.push({ arch: arch.name, D: arch.D, B, epochs, gamesPerSec, boardStepsPerSec, msPerStep, entLast: last.entropy, lossLast: last.loss });
      console.log(`  ${arch.name}(D=${arch.D}) B=${B} epochs=${epochs}: ${gamesPerSec.toFixed(0)} g/s, ${Math.round(boardStepsPerSec)} board-steps/s, ent=${(last.entropy ?? 0).toFixed(2)}, loss=${(last.loss ?? 0).toFixed(3)}`);
    }
  }
}

console.log(`\n================  SWEEP MATRIX (STEPS=${STEPS}, B=batch, ep=ppoEpochs)  ================`);
console.log(`arch       D   B     ep   games/s  board-steps/s  ms/step  entLast  lossLast`);
for (const r of rows) {
  console.log(
    `${r.arch.padEnd(10)} ${String(r.D).padEnd(3)} ${String(r.B).padEnd(5)} ${String(r.epochs).padEnd(4)} ` +
    `${r.gamesPerSec.toFixed(0).padStart(7)}  ${String(Math.round(r.boardStepsPerSec)).padStart(13)}  ${r.msPerStep.toFixed(0).padStart(7)}  ` +
    `${(r.entLast ?? 0).toFixed(2).padStart(7)}  ${(r.lossLast ?? 0).toFixed(3).padStart(8)}`
  );
}
console.log(`\nSWEEP_JSON ${JSON.stringify(rows.map((r) => ({ ...r, gamesPerSec: +r.gamesPerSec.toFixed(1), boardStepsPerSec: Math.round(r.boardStepsPerSec), msPerStep: +r.msPerStep.toFixed(0), entLast: +(r.entLast ?? 0).toFixed(2), lossLast: +(r.lossLast ?? 0).toFixed(3) })))}`);

process.exit(0);
