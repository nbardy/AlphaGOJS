// play_vs_model.ts — Human-vs-AI game controller.
//
// Lets a human play a full game against the BEST trained model, reusing the exact
// game engine + JS inference path that eval_harness uses (so the human plays against
// the same rules the network was trained on). The only difference from
// eval_harness.playGame is: on the human's turn we wait for a canvas click instead
// of calling inferAction.
//
// AI weights source (simplest path that works, in priority order):
//   1. Highest-Elo checkpoint persisted in IndexedDB (CheckpointPool / IDBStorage).
//      Same origin as the page, so the main thread can read the same DB the worker
//      writes. The checkpoint's architecture D is inferred from dense.length because
//      Checkpoint does not store D (each D has a unique denseWeightCount — see
//      computeWeightLayout). This avoids any worker round-trip.
//   2. Fall back to live GPU weights via a getLiveWeights() callback (a postMessage
//      round-trip to the worker's readWeights()) when no checkpoint has been saved yet.
//
// One clean path per concern: weight resolution is its own step (returns a canonical
// {weights, layout, source}); the game loop never re-checks "where did weights come
// from" — it just plays.

import { computeWeightLayout, type WeightLayout } from './arch_config';
import { type ArchWeights, inferAction } from './js_inference';
import {
  K, N,
  initBoard, getCellState, setCellState,
  plagueSpread, isTerminal, countTerritory,
  pcg,
} from './js_game';
import { CheckpointPool, type Checkpoint } from './checkpoint_pool';

// --- Weight source resolution ---------------------------------------------

export interface ResolvedModel {
  weights: ArchWeights;
  layout: WeightLayout;
  source: string; // human-readable provenance, e.g. "checkpoint #7 (Elo 1083)" or "live weights"
}

/** Reverse-map a checkpoint's dense weight array length to its architecture D.
 *  Each D produces a unique denseWeightCount, so the mapping is exact. Tries the
 *  common league sizes first, then a small brute-force sweep for safety. */
function inferDFromDenseLength(denseLength: number): number | null {
  for (let D = 1; D <= 64; D++) {
    if (computeWeightLayout(D).denseWeightCount === denseLength) return D;
  }
  return null;
}

export type LiveWeightsGetter = () => Promise<{ dense: Float32Array; embed: Float16Array; D: number }>;

/** Resolve the best model: highest-Elo checkpoint from IDB, else live GPU weights. */
export async function resolveBestModel(getLiveWeights: LiveWeightsGetter): Promise<ResolvedModel> {
  const pool = new CheckpointPool();
  await pool.init();

  // Pick the highest-Elo checkpoint whose D we can recover.
  let best: Checkpoint | null = null;
  for (const ckpt of pool.checkpoints) {
    if (inferDFromDenseLength(ckpt.dense.length) === null) continue; // skip unrecognized
    if (!best || ckpt.elo > best.elo) best = ckpt;
  }

  if (best) {
    const D = inferDFromDenseLength(best.dense.length)!;
    return {
      weights: { dense: best.dense, embed: best.embed },
      layout: computeWeightLayout(D),
      source: `checkpoint #${best.id} (Elo ${best.elo.toFixed(0)}, D=${D})`,
    };
  }

  // No saved checkpoint yet — fall back to current live GPU weights.
  const live = await getLiveWeights();
  return {
    weights: { dense: live.dense, embed: live.embed },
    layout: computeWeightLayout(live.D),
    source: `live weights (D=${live.D})`,
  };
}

// --- Game outcome ----------------------------------------------------------

export type Outcome =
  | { kind: 'human' }
  | { kind: 'ai' }
  | { kind: 'draw' };

function outcomeFromTerritory(humanPlayer: number, board: Uint32Array): Outcome {
  const { p1, p2 } = countTerritory(board);
  const humanTerritory = humanPlayer === 1 ? p1 : p2;
  const aiTerritory = humanPlayer === 1 ? p2 : p1;
  if (humanTerritory > aiTerritory) return { kind: 'human' };
  if (aiTerritory > humanTerritory) return { kind: 'ai' };
  return { kind: 'draw' };
}

// --- Controller ------------------------------------------------------------

export interface PlayCallbacks {
  /** Called after every board mutation so the host can redraw. */
  onRender: (board: Uint32Array) => void;
  /** Status / prompt text for the human ("Your turn", "AI thinking…", winner, …). */
  onStatus: (text: string) => void;
  /** Called once when the game terminates, with the final outcome. */
  onGameOver: (outcome: Outcome) => void;
}

export class PlayController {
  private board: Uint32Array;
  private model: ResolvedModel | null = null;
  private getLiveWeights: LiveWeightsGetter;
  private callbacks: PlayCallbacks;

  private humanPlayer: number; // 1 = blue (moves first), 2 = red
  private aiPlayer: number;
  private currentPlayer = 1;
  private stepCount = 0;
  private gameSeed: number;
  private over = false;
  private acceptingInput = false; // true only while it's the human's turn

  /** humanPlayer: 1 (human goes first) or 2 (AI goes first). */
  constructor(callbacks: PlayCallbacks, getLiveWeights: LiveWeightsGetter, humanPlayer = 1) {
    this.callbacks = callbacks;
    this.getLiveWeights = getLiveWeights;
    this.humanPlayer = humanPlayer;
    this.aiPlayer = humanPlayer === 1 ? 2 : 1;
    this.gameSeed = (Date.now() ^ (Math.random() * 0xffffffff)) >>> 0;
    this.board = initBoard(this.gameSeed, 0.3);
  }

  /** Initialize the model (best checkpoint or live) and start the game. */
  async start(): Promise<void> {
    this.callbacks.onStatus('Loading best model…');
    this.model = await resolveBestModel(this.getLiveWeights);
    this.over = false;
    this.currentPlayer = 1;
    this.stepCount = 0;
    this.gameSeed = (Date.now() ^ (Math.random() * 0xffffffff)) >>> 0;
    this.board = initBoard(this.gameSeed, 0.3);
    this.callbacks.onRender(this.board);
    await this.advance();
  }

  modelSource(): string {
    return this.model?.source ?? 'unknown';
  }

  /** Map a click at canvas pixel (px,py) to a cell index, given the canvas size. */
  private cellFromClick(px: number, py: number, canvasW: number, canvasH: number): number {
    const cellW = canvasW / K;
    const cellH = canvasH / K;
    const x = Math.min(K - 1, Math.max(0, Math.floor(px / cellW)));
    const y = Math.min(K - 1, Math.max(0, Math.floor(py / cellH)));
    return y * K + x;
  }

  /** A move is legal iff the target cell is empty (state 0). Mirrors the engine:
   *  inferAction masks all non-empty cells, so empty == legal for both sides. */
  private isLegal(cell: number): boolean {
    return getCellState(this.board, cell) === 0;
  }

  /** Handle a raw canvas click. No-op unless it's the human's turn and legal. */
  handleClick(px: number, py: number, canvasW: number, canvasH: number): void {
    if (!this.acceptingInput || this.over) return;
    const cell = this.cellFromClick(px, py, canvasW, canvasH);
    if (!this.isLegal(cell)) {
      this.callbacks.onStatus('That cell is taken — pick an empty (dark) square.');
      return;
    }
    this.acceptingInput = false;
    this.applyMove(cell, this.humanPlayer);
    // Hand control to the engine to play the AI turn(s) and re-prompt the human.
    void this.advance();
  }

  /** Apply a single move: place piece + plague spread + alternate turn.
   *  Mirrors eval_harness.playGame exactly. */
  private applyMove(cell: number, player: number): void {
    setCellState(this.board, cell, player);
    plagueSpread(this.board, this.gameSeed, this.stepCount, cell, 0);
    this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
    this.stepCount++;
    this.callbacks.onRender(this.board);
  }

  /** Drive the game forward: settle terminal, play AI turns, then prompt human.
   *  Returns when control is handed back to the human (or the game ends). */
  private async advance(): Promise<void> {
    if (this.checkTerminal()) return;

    // Play AI turns until it's the human's turn (or the game ends).
    while (this.currentPlayer === this.aiPlayer) {
      this.callbacks.onStatus('AI thinking…');
      // Yield so the "AI thinking…" status + last render paint before the blocking
      // synchronous forward pass (inferAction is CPU-bound and not trivially fast).
      await new Promise<void>((r) => requestAnimationFrame(() => r()));

      const action = this.aiSelectAction();
      this.applyMove(action, this.aiPlayer);
      if (this.checkTerminal()) return;
    }

    // Human's turn.
    this.acceptingInput = true;
    this.callbacks.onStatus(
      `Your turn (${this.humanPlayer === 1 ? 'blue' : 'red'}) — click an empty square.`,
    );
  }

  /** Run the best model's forward pass for the AI's move. */
  private aiSelectAction(): number {
    const model = this.model!;
    // Derive a per-step seed the same way eval_harness does.
    const stepSeed = pcg((this.gameSeed ^ pcg(this.stepCount >>> 0)) >>> 0);
    return inferAction(model.weights, model.layout, this.board, this.aiPlayer, stepSeed);
  }

  /** If the board is terminal, finalize and report the outcome. Returns true if over. */
  private checkTerminal(): boolean {
    if (this.over) return true;
    if (!isTerminal(this.board)) return false;
    this.over = true;
    this.acceptingInput = false;
    const outcome = outcomeFromTerritory(this.humanPlayer, this.board);
    const { p1, p2 } = countTerritory(this.board);
    const label =
      outcome.kind === 'human' ? 'You win!' :
      outcome.kind === 'ai' ? 'AI wins.' :
      "It's a draw.";
    this.callbacks.onStatus(`${label}  (blue ${p1} — red ${p2})`);
    this.callbacks.onGameOver(outcome);
    return true;
  }

  /** Render the current board immediately (used when re-entering play mode). */
  renderNow(): void {
    this.callbacks.onRender(this.board);
  }

  /** Stop accepting input (used when exiting play mode). */
  dispose(): void {
    this.acceptingInput = false;
    this.over = true;
  }
}
