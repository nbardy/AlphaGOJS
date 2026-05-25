/**
 * League Worker — web worker entry point for multi-architecture training.
 *
 * Separate from worker.ts (single-architecture mode). Initializes one GPUTrainer
 * per architecture in LEAGUE_ARCHS, wires stats callbacks, and runs the
 * LeagueManager round-robin loop.
 *
 * Message protocol:
 *   Main -> Worker:  START_LEAGUE, PAUSE, RESUME
 *   Worker -> Main:  LOG, ERROR, LEAGUE_STATS, BOARD
 */

import { LeagueManager, type TrainerStats } from './league_manager';
import { GPUTrainer } from './gpu_harness';
import { LEAGUE_ARCHS } from './arch_config';

let manager: LeagueManager;
let paused = false;

self.onmessage = async (e: MessageEvent) => {
  const { type } = e.data;

  if (type === 'START_LEAGUE') {
    await startLeague();
  } else if (type === 'PAUSE') {
    paused = true;
    self.postMessage({ type: 'LOG', msg: 'League paused' });
  } else if (type === 'RESUME') {
    paused = false;
    self.postMessage({ type: 'LOG', msg: 'League resumed' });
  }
};

async function startLeague(): Promise<void> {
  try {
    manager = new LeagueManager(LEAGUE_ARCHS);

    // Initialize one GPUTrainer per architecture.
    // Each trainer gets its own GPU pipeline parameterized by ArchConfig.D.
    for (const arch of LEAGUE_ARCHS) {
      self.postMessage({ type: 'LOG', msg: `Initializing ${arch.name} (D=${arch.D})...` });

      // GPUTrainer constructor accepts ArchConfig (another agent is adding this).
      const trainer = new GPUTrainer(arch as any);

      trainer.onStats = (stats: TrainerStats) => {
        const archState = manager.archs.find(a => a.config.name === arch.name);
        if (archState) {
          archState.latestLoss = stats.loss;
          archState.latestGamesPerSec = stats.trainedGamesPerSec;
        }
      };

      trainer.onBoard = (board: Uint32Array) => {
        self.postMessage({ type: 'BOARD', board, arch: arch.name });
      };

      await trainer.init();
      manager.trainers.set(arch.name, trainer);
      self.postMessage({ type: 'LOG', msg: `${arch.name} ready` });
    }

    self.postMessage({ type: 'LOG', msg: 'League initialized — starting training' });

    // Main training loop: round-robin across architectures indefinitely
    while (true) {
      if (paused) {
        await new Promise(resolve => setTimeout(resolve, 100));
        continue;
      }

      const stats = await manager.runStep();
      self.postMessage({ type: 'LEAGUE_STATS', stats });
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'ERROR', message });
  }
}
