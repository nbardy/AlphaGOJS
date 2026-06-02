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

  // Fixed reference opponent for the smooth win-rate progress curve. Set once to the
  // EARLIEST checkpoint and never reassigned, so it survives eviction from
  // `checkpoints` (which keeps only the last 20). Win-rate vs this fixed yardstick
  // climbs monotonically like Elo, unlike win-rate vs a randomly sampled opponent.
  anchor: Checkpoint | null = null;

  async init() {
    await this.storage.init();
    const loaded = await this.storage.loadAllCheckpoints();
    if (loaded && loaded.length > 0) {
      this.checkpoints = loaded;
      this.anchor = loaded[0] ?? null; // earliest persisted checkpoint = the anchor
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
    if (!this.anchor) this.anchor = ckpt; // first checkpoint becomes the fixed anchor
    await this.storage.saveCheckpoint(ckpt);

    if (this.checkpoints.length > 20) {
      const oldest = this.checkpoints.shift();
      if (oldest) await this.storage.deleteCheckpoint(oldest.id);
    }
  }

  sampleOpponent(): Checkpoint | null {
    if (this.checkpoints.length === 0) return null;
    // Exponential recency bias: u^0.5 maps uniform [0,1] toward higher indices (recent checkpoints)
    const u = Math.random();
    const biased = Math.floor(Math.pow(u, 0.5) * this.checkpoints.length);
    const idx = Math.min(biased, this.checkpoints.length - 1);
    return this.checkpoints[idx];
  }

  updateElo(opponentElo: number, currentWon: boolean, draw: boolean) {
    // Standard Elo: expected_A = 1 / (1 + 10^((R_B - R_A) / 400))
    const expected = 1 / (1 + Math.pow(10, (opponentElo - this.currentElo) / 400));
    const actual = draw ? 0.5 : (currentWon ? 1 : 0);
    this.currentElo += this.eloK * (actual - expected);
    return this.currentElo;
  }
}
