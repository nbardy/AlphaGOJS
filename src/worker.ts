import { GPUTrainer } from './gpu_harness';
import { LEAGUE_ARCHS, type ArchConfig } from './arch_config';

let trainer: GPUTrainer;
let paused = false;

self.onmessage = async (e) => {
  if (e.data.type === 'START') {
    self.postMessage({ type: 'LOG', msg: "Worker received START" });
    try {
      // Accept arch config from message, default to 'standard' (D=8)
      const archConfig: ArchConfig = e.data.archConfig ?? LEAGUE_ARCHS[1];
      trainer = new GPUTrainer(archConfig);
      trainer.onStats = (stats) => self.postMessage({ type: 'STATS', stats });
      trainer.onBoard = (board) => self.postMessage({ type: 'BOARD', board });

      self.postMessage({ type: 'LOG', msg: "Initializing trainer..." });
      await trainer.init();
      self.postMessage({ type: 'LOG', msg: "GPUTrainer initialized" });

      while (true) {
        if (paused) {
          await new Promise(resolve => setTimeout(resolve, 100));
          continue;
        }
        await trainer.runStep();
      }
    } catch (err: any) {
      self.postMessage({ type: 'ERROR', message: err.message || String(err) });
    }
  } else if (e.data.type === 'PAUSE') {
    paused = true;
    self.postMessage({ type: 'LOG', msg: "Training paused" });
  } else if (e.data.type === 'RESUME') {
    paused = false;
    self.postMessage({ type: 'LOG', msg: "Training resumed" });
  } else if (e.data.type === 'GET_WEIGHTS') {
    // Live-weights fallback for human-vs-model play when no checkpoint exists yet.
    // Reads the current GPU weights and ships them to the main thread.
    try {
      const { dense, embed } = await trainer.readWeights();
      self.postMessage({ type: 'WEIGHTS', dense, embed, D: trainer.archConfig.D });
    } catch (err: any) {
      self.postMessage({ type: 'ERROR', message: err.message || String(err) });
    }
  }
};
