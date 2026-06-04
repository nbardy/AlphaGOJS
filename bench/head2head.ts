// Head-to-head with POSITION SPLIT. Trains agents in-process (the bench's IndexedDB is
// in-memory/ephemeral, so checkpoints aren't persisted — we capture final weights via
// readWeights() and keep them in memory), then plays them against each other via CPU JS
// inference, tracking wins separately for first-player (P1) vs second-player (P2) so we can
// separate the FIRST-MOVE ADVANTAGE from relative SKILL.
//
//   Usage:  GENS=120 GAMES=40 bun bench/head2head.ts
//
// Reports per pair: A's win% as P1 vs as P2 (the gap = first-move advantage), A's overall
// skill, and the global P1 win-rate (pure first-mover advantage across all games).

import { setupGlobals } from "bun-webgpu";
import { installFakeIndexedDB } from "./idb_stub";

setupGlobals();
(globalThis as any).GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
(globalThis as any).GPUMapMode ??= { READ: 1, WRITE: 2 };

const { GPUTrainer, CONFIG } = await import("../src/gpu_harness");
const { LEAGUE_ARCHS, computeWeightLayout } = await import("../src/arch_config");
const { inferAction } = await import("../src/js_inference");
const jg: any = await import("../src/js_game");

const GENS = Number(process.env.GENS ?? 120);
const GAMES = Number(process.env.GAMES ?? 40); // games per direction (×2 = per pair)

interface Agent { name: string; D: number; weights: any; layout: any; }

async function train(archName: string, epochs: number, gens: number, name: string): Promise<Agent> {
  installFakeIndexedDB();
  const arch = LEAGUE_ARCHS.find((a: any) => a.name === archName)!;
  CONFIG.ppoEpochs = epochs;
  const t = new GPUTrainer(arch);
  t.onStats = () => {};
  await t.init();
  for (let g = 0; g < gens; g++) await t.runStep();
  console.log(`  trained ${name} (D=${arch.D}, ep=${epochs}, ${gens} gens)`);
  return { name, D: arch.D, weights: await t.readWeights(), layout: computeWeightLayout(arch.D) };
}

// One game; p1 moves first. Mirrors eval_harness.playGame.
function playGame(p1: Agent, p2: Agent, seed: number): "p1" | "p2" | "draw" {
  const board = jg.initBoard(seed, 0.3);
  let cur = 1, step = 0;
  while (step < 600) {
    if (jg.isTerminal(board)) break;
    const slot = cur === 1 ? p1 : p2;
    const stepSeed = jg.pcg((seed ^ jg.pcg(step >>> 0)) >>> 0);
    const action = inferAction(slot.weights, slot.layout, board, cur, stepSeed);
    jg.setCellState(board, action, cur);
    jg.plagueSpread(board, seed, step, action, 0);
    cur = cur === 1 ? 2 : 1; step++;
  }
  const { p1: t1, p2: t2 } = jg.countTerritory(board);
  return t1 > t2 ? "p1" : t2 > t1 ? "p2" : "draw";
}

console.log(`\n=== head-to-head (GENS=${GENS}, GAMES=${GAMES}/direction) ===`);
const agents: Agent[] = [
  await train("compact", 1, GENS, "trained-A"),
  await train("compact", 1, GENS, "trained-B"),
  await train("compact", 1, 0, "untrained"), // random-init skill floor
];

let globalP1Wins = 0, globalDecisive = 0;
console.log(`\npair                         A-as-P1   A-as-P2   A overall   (first-move gap)`);
for (let i = 0; i < agents.length; i++) {
  for (let j = i + 1; j < agents.length; j++) {
    const A = agents[i], B = agents[j];
    let aP1w = 0, aP1d = 0, aP2w = 0, aP2d = 0;
    for (let g = 0; g < GAMES; g++) {
      const seed = jg.pcg((0x51A1 ^ jg.pcg(g >>> 0)) >>> 0);
      let r = playGame(A, B, seed);            // A is P1
      if (r === "p1") { aP1w++; globalP1Wins++; globalDecisive++; } else if (r === "p2") { globalDecisive++; }
      if (r !== "draw") aP1d++;
      r = playGame(B, A, seed);                // A is P2 (B is P1)
      if (r === "p2") { aP2w++; } else if (r === "p1") { globalP1Wins++; }
      if (r !== "draw") { aP2d++; globalDecisive++; }
    }
    const aP1 = aP1d ? aP1w / aP1d : 0.5;
    const aP2 = aP2d ? aP2w / aP2d : 0.5;
    const overall = (aP1w + aP2w) / Math.max(1, aP1d + aP2d);
    const label = `${A.name} vs ${B.name}`;
    console.log(`${label.padEnd(28)} ${(aP1 * 100).toFixed(0).padStart(5)}%   ${(aP2 * 100).toFixed(0).padStart(5)}%   ${(overall * 100).toFixed(0).padStart(7)}%   (${((aP1 - aP2) * 100).toFixed(0)} pts)`);
  }
}
const p1Adv = globalDecisive ? globalP1Wins / globalDecisive : 0.5;
console.log(`\nFIRST-MOVE ADVANTAGE (P1 win-rate across ALL ${globalDecisive} decisive games): ${(p1Adv * 100).toFixed(1)}%`);
console.log(`  (50% = no advantage; >50% = first player is favored)`);
process.exit(0);
