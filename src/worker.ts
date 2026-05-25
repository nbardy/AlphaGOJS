import { GPUTrainer } from './gpu_harness';

let trainer: GPUTrainer;
let paused = false;

self.onmessage = async (e) => {
  if (e.data.type === 'START') {
    self.postMessage({ type: 'LOG', msg: "Worker received START" });
    try {
      trainer = new GPUTrainer();
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
  }
};
