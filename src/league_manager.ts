/**
 * League Manager — coordinates multi-architecture training and tournament.
 *
 * Round-robin training: each `runStep()` trains one architecture, then advances
 * to the next. Periodic cross-architecture evaluation updates Elo ratings.
 *
 * Design note: GPUTrainer instances are injected via `trainers` map after GPU
 * init (the worker owns device lifecycle). The manager never touches the GPU
 * directly — it delegates to trainer.runStep() and trainer.readWeights().
 */

import { type ArchConfig, LEAGUE_ARCHS, computeWeightLayout } from './arch_config';

// ── Domain types ──

/** Per-architecture mutable state tracked by the league. */
export interface ArchState {
  readonly config: ArchConfig;
  elo: number;
  generation: number;
  totalGames: number;
  latestLoss: number;
  latestGamesPerSec: number;
}

/** Snapshot of a single cross-architecture evaluation. */
export interface EvalResultEntry {
  readonly archA: string;
  readonly archB: string;
  readonly winsA: number;
  readonly winsB: number;
  readonly draws: number;
  readonly rollout: number;  // league-wide rollout counter when this eval happened
}

/** Full league state snapshot for the UI. */
export interface LeagueStats {
  readonly archs: readonly ArchState[];
  readonly evalHistory: readonly EvalResultEntry[];
  readonly currentTraining: string;
  readonly totalRollouts: number;
}

/** Weights read back from a GPUTrainer (dense f32 + embed f16). */
export interface ArchWeights {
  readonly dense: Float32Array;
  readonly embed: Float16Array;
}

/** Minimal trainer interface the league manager requires from GPUTrainer. */
export interface TrainerHandle {
  runStep(): Promise<void>;
  readWeights(): Promise<ArchWeights>;
  rollout: number;
  onStats?: (stats: TrainerStats) => void;
  onBoard?: (board: Uint32Array) => void;
}

/** Stats emitted by a single trainer step. */
export interface TrainerStats {
  loss: number;
  trainedGamesPerSec: number;
  rollout: number;
  games: number;
  [key: string]: unknown;
}

// ── Elo computation ──

/**
 * Standard Elo update for a pair of players after one match result.
 * scoreA: 1 = A wins, 0 = A loses, 0.5 = draw.
 */
function eloUpdate(
  ratingA: number,
  ratingB: number,
  scoreA: number,
  K: number,
): { a: number; b: number } {
  const expectedA = 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
  return {
    a: ratingA + K * (scoreA - expectedA),
    b: ratingB + K * ((1 - scoreA) - (1 - expectedA)),
  };
}

// ── League Manager ──

/** Number of boards per trainer step (matches CONFIG.numBoards in gpu_harness). */
const BOARDS_PER_STEP = 256;

export class LeagueManager {
  readonly archs: ArchState[];
  readonly evalHistory: EvalResultEntry[] = [];
  readonly trainers: Map<string, TrainerHandle> = new Map();

  currentArchIdx = 0;
  totalRollouts = 0;

  /** Evaluate every N league-wide rollouts. */
  evalInterval = 10;
  /** Number of games per cross-architecture evaluation matchup. */
  evalGamesPerPair = 20;
  /** Elo K-factor for cross-architecture evaluation. */
  eloK = 32;

  constructor(configs: ArchConfig[] = LEAGUE_ARCHS) {
    this.archs = configs.map(c => ({
      config: c,
      elo: 1000,
      generation: 0,
      totalGames: 0,
      latestLoss: 0,
      latestGamesPerSec: 0,
    }));
  }

  /**
   * Run one league step: train the current architecture, advance the round-robin,
   * and periodically trigger cross-architecture evaluation.
   *
   * Called each iteration from the worker loop.
   */
  async runStep(): Promise<LeagueStats> {
    const arch = this.archs[this.currentArchIdx];
    const trainer = this.trainers.get(arch.config.name);
    if (!trainer) {
      throw new Error(`No trainer registered for architecture "${arch.config.name}"`);
    }

    await trainer.runStep();

    arch.generation++;
    if (trainer.rollout > 0) {
      arch.totalGames += BOARDS_PER_STEP;
    }

    this.totalRollouts++;

    // Advance round-robin to next architecture
    this.currentArchIdx = (this.currentArchIdx + 1) % this.archs.length;

    // Periodic cross-architecture evaluation
    if (this.totalRollouts % this.evalInterval === 0) {
      await this.runEvalRound();
    }

    return this.getStats();
  }

  /**
   * Run one evaluation round: pick a random pair of architectures,
   * read their weights, and play games to update Elo.
   *
   * Uses dynamic import to avoid circular dependency with eval_harness
   * (which itself imports arch_config types).
   */
  async runEvalRound(): Promise<void> {
    if (this.archs.length < 2) return;

    // Pick a random pair (Fisher-Yates single draw for two distinct indices)
    const i = Math.floor(Math.random() * this.archs.length);
    let j = Math.floor(Math.random() * (this.archs.length - 1));
    if (j >= i) j++;

    const archA = this.archs[i];
    const archB = this.archs[j];

    const trainerA = this.trainers.get(archA.config.name);
    const trainerB = this.trainers.get(archB.config.name);
    if (!trainerA || !trainerB) return;

    const [weightsA, weightsB] = await Promise.all([
      trainerA.readWeights(),
      trainerB.readWeights(),
    ]);

    // Dynamic import: eval_harness is created by another agent and may not
    // exist yet at compile time. The league worker still initializes fine.
    const { evalArchitectures } = await import('./eval_harness');

    const result = evalArchitectures(
      archA.config,
      weightsA,
      archB.config,
      weightsB,
      this.evalGamesPerPair,
      this.totalRollouts * 1000,  // seed derived from league rollout count
    );

    // Compute aggregate score and update Elo
    const scoreA = (result.winsA + result.draws * 0.5) / result.games;
    const updated = eloUpdate(archA.elo, archB.elo, scoreA, this.eloK);
    archA.elo = updated.a;
    archB.elo = updated.b;

    this.evalHistory.push({
      archA: archA.config.name,
      archB: archB.config.name,
      winsA: result.winsA,
      winsB: result.winsB,
      draws: result.draws,
      rollout: this.totalRollouts,
    });
  }

  /** Return a snapshot of the full league state for the UI. */
  getStats(): LeagueStats {
    return {
      archs: this.archs,
      evalHistory: this.evalHistory,
      currentTraining: this.archs[this.currentArchIdx].config.name,
      totalRollouts: this.totalRollouts,
    };
  }
}
