// Headless throughput + sanity bench for the REAL browser GPUTrainer, run under Bun
// via bun-webgpu. This exercises the exact same src/gpu_harness.ts code path the
// browser runs (rollout + GAE + PPO epochs + readbacks + Elo eval) — only the
// browser-only edges are stubbed: navigator.gpu/Float16Array come from bun-webgpu's
// setupGlobals(), IndexedDB from our in-memory stub, and the canvas/worker are
// simply never used (onBoard is left unset so captureBoard early-returns).
//
//   Usage:  bun bench/headless.ts
//   Env:    B=512 STEPS=20 WARMUP=3 ARCH=standard bun bench/headless.ts
//           ARCH ∈ compact(D=4) | standard(D=8) | wide(D=16)
//
// Output is a single JSON-ish summary block so before/after runs are easy to diff.

import { setupGlobals } from "bun-webgpu";
import { installFakeIndexedDB } from "./idb_stub";

setupGlobals();                 // defines navigator.gpu, Float16Array, GPU* constants
installFakeIndexedDB();

// Belt-and-suspenders: ensure the two constant namespaces the harness uses exist
// (setupGlobals provides them, but ??= is a no-op if so and a safety net if not).
(globalThis as any).GPUBufferUsage ??= {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32,
  UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512,
};
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const { GPUTrainer, CONFIG } = await import("../src/gpu_harness");
const { LEAGUE_ARCHS } = await import("../src/arch_config");

const STEPS = Number(process.env.STEPS ?? 20);
const WARMUP = Number(process.env.WARMUP ?? 3);
const ARCH_NAME = process.env.ARCH ?? "standard";
const arch = LEAGUE_ARCHS.find((a: any) => a.name === ARCH_NAME) ?? LEAGUE_ARCHS[1];

// Optional batch-size override so we can A/B the batch lever WITHOUT editing source.
if (process.env.B) CONFIG.numBoards = Number(process.env.B);

console.log(`\n=== AlphaGOJS headless bench ===`);
console.log(`arch=${arch.name} D=${arch.D} | B=${CONFIG.numBoards} | maxSteps=${CONFIG.maxSteps} | ppoEpochs=${CONFIG.ppoEpochs}`);
console.log(`warmup=${WARMUP} timed=${STEPS}\n`);

const trainer = new GPUTrainer(arch);
let last: any = {};
trainer.onStats = (s: any) => { last = s; };
// onBoard intentionally left unset → captureBoard() early-returns (no canvas needed).

const tInit0 = performance.now();
await trainer.init();
const initMs = performance.now() - tInit0;
console.log(`init+compile: ${initMs.toFixed(0)} ms`);

for (let i = 0; i < WARMUP; i++) await trainer.runStep();

const perStepMs: number[] = [];
const entropies: number[] = [];
const losses: number[] = [];
const avgStepsArr: number[] = [];
const winRates: number[] = [];

const t0 = performance.now();
for (let i = 0; i < STEPS; i++) {
  const a = performance.now();
  await trainer.runStep();
  perStepMs.push(performance.now() - a);
  entropies.push(last.entropy);
  losses.push(last.loss);
  avgStepsArr.push(last.avgStepsPerGame);
  if (last.winRate >= 0) winRates.push(last.winRate);
}
const totalMs = performance.now() - t0;

const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };

// games/sec uses the same definition the dashboard reports: boards completed per second.
const gamesPerSec = (STEPS * CONFIG.numBoards) / (totalMs / 1000);
// board-environment-steps/sec (the FUSED_js unit): boards × avg game length per second.
const avgGameLen = mean(avgStepsArr);
const boardStepsPerSec = gamesPerSec * avgGameLen;

console.log(`\n--- THROUGHPUT ---`);
console.log(`per-runStep ms:   mean=${mean(perStepMs).toFixed(1)}  median=${median(perStepMs).toFixed(1)}  min=${Math.min(...perStepMs).toFixed(1)}  max=${Math.max(...perStepMs).toFixed(1)}`);
console.log(`games/sec:        ${gamesPerSec.toFixed(1)}   (B=${CONFIG.numBoards} / runStep wallclock)`);
console.log(`avg steps/game:   ${avgGameLen.toFixed(1)}`);
console.log(`board-steps/sec:  ${boardStepsPerSec.toFixed(0)}   (= games/sec × avg steps/game; comparable to FUSED_js steps/sec)`);

console.log(`\n--- TRAINING SANITY (first → last of ${STEPS} timed steps) ---`);
console.log(`loss:     ${losses[0]?.toFixed(4)} → ${losses[losses.length - 1]?.toFixed(4)}`);
console.log(`entropy:  ${entropies[0]?.toFixed(4)} → ${entropies[entropies.length - 1]?.toFixed(4)}   (uniform-policy max = ln(${CONFIG.numBoards >= 0 ? 24 * 24 : 0}) ≈ ${Math.log(24 * 24).toFixed(3)})`);
console.log(`elo:      ${last.elo?.toFixed(1)}`);
console.log(`win-rate evals recorded: ${winRates.length}  values: [${winRates.map((w) => w.toFixed(2)).join(", ")}]`);

// One-line machine-greppable summary.
console.log(`\nSUMMARY ${JSON.stringify({
  arch: arch.name, D: arch.D, B: CONFIG.numBoards,
  gamesPerSec: +gamesPerSec.toFixed(1),
  boardStepsPerSec: Math.round(boardStepsPerSec),
  perStepMsMean: +mean(perStepMs).toFixed(1),
  avgStepsPerGame: +avgGameLen.toFixed(1),
  entropyFirst: +(entropies[0] ?? 0).toFixed(3),
  entropyLast: +(entropies[entropies.length - 1] ?? 0).toFixed(3),
  lossLast: +(losses[losses.length - 1] ?? 0).toFixed(4),
  initMs: Math.round(initMs),
})}`);

process.exit(0);
