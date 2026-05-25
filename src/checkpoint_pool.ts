import { IDBStorage } from './idb_storage';

export interface Checkpoint {
  id: number;
  embed: Float16Array;
  dense: Float32Array;
  elo: number;
  generation: number;
}

export class CheckpointPool {
  checkpoints: Checkpoint[] = [];
  currentElo = 1000;
  eloK = 32;
  nextId = 1;
  storage = new IDBStorage();

  async init() {
    await this.storage.init();
    const loaded = await this.storage.loadAllCheckpoints();
    if (loaded && loaded.length > 0) {
      this.checkpoints = loaded;
      const latest = loaded[loaded.length - 1];
      if (latest) {
        this.currentElo = latest.elo;
        this.nextId = latest.id + 1;
        console.log(`Loaded ${loaded.length} checkpoints from IDB. Current Elo: ${this.currentElo}`);
      }
    }
  }

  async saveCheckpoint(embed: Float16Array, dense: Float32Array, generation: number) {
    const ckpt: Checkpoint = {
      id: this.nextId++,
      embed: new Float16Array(embed),
      dense: new Float32Array(dense),
      elo: this.currentElo,
      generation
    };
    
    this.checkpoints.push(ckpt);
    await this.storage.saveCheckpoint(ckpt);
    
    if (this.checkpoints.length > 20) {
      const oldest = this.checkpoints.shift();
      if (oldest) await this.storage.deleteCheckpoint(oldest.id);
    }
  }

  sampleOpponent(): Checkpoint | null {
    if (this.checkpoints.length === 0) return null;
    // heavily bias towards recent checkpoints
    const idx = Math.floor(Math.random() * this.checkpoints.length);
    return this.checkpoints[idx];
  }

  updateElo(opponentElo: number, currentWon: boolean, draw: boolean) {
    const expected = 1 / (1 + Math.pow(10, (this.currentElo - opponentElo) / 400));
    const actual = draw ? 0.5 : (currentWon ? 1 : 0);
    this.currentElo += this.eloK * (actual - expected);
    return this.currentElo;
  }
}
