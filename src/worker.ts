import { GPUTrainer } from './gpu_harness';

console.log("Worker executing top level!");

let trainer: GPUTrainer;

self.onmessage = async (e) => {
  if (e.data.type === 'START') {
    self.postMessage({ type: 'LOG', msg: "Worker received START message" });
    try {
      trainer = new GPUTrainer();
      trainer.onStats = (stats) => self.postMessage({ type: 'STATS', stats });
      trainer.onBoard = (board) => self.postMessage({ type: 'BOARD', board });
      
      self.postMessage({ type: 'LOG', msg: "Worker initializing trainer..." });
      await trainer.init();
      self.postMessage({ type: 'LOG', msg: "Worker initialized GPUTrainer successfully" });
      
      while (true) {
        await trainer.runStep();
      }
    } catch (err: any) {
      self.postMessage({ type: 'ERROR', message: err.message || err.toString() });
    }
  }
};
