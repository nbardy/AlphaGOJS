/**
 * Browser / webpack entry: bundles WGSL via asset/source rule.
 * For Node, import webgpu_plague_spread_engine.js and pass readFileSync(wgslPath).
 */

import plagueSpreadWGSL from './wgsl/plague_spread.wgsl';
import { WebGPUPlagueSpreadEngine as Core } from './webgpu_plague_spread_engine.js';

export {
  CELL_EMPTY,
  CELL_P1,
  CELL_P2,
  CELL_WALL,
  encodePlagueWallsBoardToPacked,
  decodePackedToPlagueWallsInt8,
  requestPlagueSpreadDevice
} from './webgpu_plague_spread_engine.js';

export class WebGPUPlagueSpreadEngine extends Core {
  constructor(device, config) {
    super(device, config, plagueSpreadWGSL);
  }
}
