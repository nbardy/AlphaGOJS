// eval_harness.ts — Cross-architecture tournament eval using CPU inference.
// Plays games between two different architectures (e.g. D=4 vs D=16)
// by running each side's forward pass in JavaScript.

import { type ArchConfig, computeWeightLayout } from './arch_config';
import { type ArchWeights, inferAction } from './js_inference';
import {
  K, N, WPB,
  initBoard, getCellState, setCellState,
  plagueSpread, isTerminal, countTerritory,
  pcg,
} from './js_game';

const MAX_STEPS = 600;

export interface EvalResult {
  archA: string;
  archB: string;
  winsA: number;
  winsB: number;
  draws: number;
  games: number;
}

// --- Single game between two configs ---

interface PlayerSlot {
  config: ArchConfig;
  weights: ArchWeights;
}

function playGame(
  p1: PlayerSlot,
  p2: PlayerSlot,
  gameSeed: number,
): 'p1' | 'p2' | 'draw' {
  const layoutP1 = computeWeightLayout(p1.config.D);
  const layoutP2 = computeWeightLayout(p2.config.D);

  // Wall density default matches typical training params (~0.3).
  // Using 0.3 as a reasonable default; callers can parameterize if needed.
  const board = initBoard(gameSeed, 0.3);
  let currentPlayer = 1; // P1 goes first
  let stepCount = 0;

  while (stepCount < MAX_STEPS) {
    if (isTerminal(board)) break;

    // Select which architecture is playing
    const activeSlot = currentPlayer === 1 ? p1 : p2;
    const activeLayout = currentPlayer === 1 ? layoutP1 : layoutP2;

    // Derive a deterministic per-step seed from the game seed and step count
    const stepSeed = pcg((gameSeed ^ pcg(stepCount >>> 0)) >>> 0);

    const action = inferAction(
      activeSlot.weights,
      activeLayout,
      board,
      currentPlayer,
      stepSeed,
    );

    // Place the piece
    setCellState(board, action, currentPlayer);

    // Plague spread (uses game seed, step, action cell, batchIdx=0)
    plagueSpread(board, gameSeed, stepCount, action, 0);

    // Alternate turns
    currentPlayer = currentPlayer === 1 ? 2 : 1;
    stepCount++;
  }

  // Count territory to determine winner
  const { p1: t1, p2: t2 } = countTerritory(board);
  if (t1 > t2) return 'p1';
  if (t2 > t1) return 'p2';
  return 'draw';
}

// --- Tournament: play numGames between two architectures ---
// Half the games have archA as P1, half have archB as P1 (for fairness).

export function evalArchitectures(
  configA: ArchConfig, weightsA: ArchWeights,
  configB: ArchConfig, weightsB: ArchWeights,
  numGames: number,
  baseSeed: number,
): EvalResult {
  let winsA = 0;
  let winsB = 0;
  let draws = 0;

  const halfGames = Math.floor(numGames / 2);

  for (let i = 0; i < numGames; i++) {
    const gameSeed = pcg((baseSeed ^ pcg(i >>> 0)) >>> 0);

    // First half: archA is P1, archB is P2
    // Second half: archB is P1, archA is P2
    const aIsP1 = i < halfGames;

    const p1Slot: PlayerSlot = aIsP1
      ? { config: configA, weights: weightsA }
      : { config: configB, weights: weightsB };

    const p2Slot: PlayerSlot = aIsP1
      ? { config: configB, weights: weightsB }
      : { config: configA, weights: weightsA };

    const result = playGame(p1Slot, p2Slot, gameSeed);

    if (result === 'p1') {
      if (aIsP1) winsA++; else winsB++;
    } else if (result === 'p2') {
      if (aIsP1) winsB++; else winsA++;
    } else {
      draws++;
    }
  }

  return {
    archA: configA.name,
    archB: configB.name,
    winsA,
    winsB,
    draws,
    games: numGames,
  };
}
