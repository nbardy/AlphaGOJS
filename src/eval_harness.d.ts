/**
 * Type declarations for eval_harness (implementation created by another agent).
 *
 * This stub allows league_manager.ts to type-check the dynamic import of
 * evalArchitectures() before the implementation file lands.
 */

import type { ArchConfig } from './arch_config';
import type { ArchWeights } from './league_manager';

export interface EvalResult {
  archA: string;
  archB: string;
  winsA: number;
  winsB: number;
  draws: number;
  games: number;
}

export function evalArchitectures(
  configA: ArchConfig,
  weightsA: ArchWeights,
  configB: ArchConfig,
  weightsB: ArchWeights,
  numGames: number,
  baseSeed: number,
): EvalResult;
