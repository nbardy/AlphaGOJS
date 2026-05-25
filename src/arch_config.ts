export interface ArchConfig {
  name: string;
  D: number;
  color: string;
}

export interface WeightLayout {
  D: number;
  K: number;
  P: number;
  PATCH_CH: number;
  denseWeightCount: number;
  embedWeightCount: number;
}

export function computeWeightLayout(D: number): WeightLayout {
  const K = 24;
  const P = K / 3;
  const PATCH_CH = 9 * D;

  const W_CONV1 = 0;
  const W_CONV2 = W_CONV1 + 9 * D * D;
  const W_FUSE  = W_CONV2 + 9 * D * PATCH_CH;
  const W_POLICY = W_FUSE + 2 * D * D;
  const W_VALUE = W_POLICY + D;
  const W_TOTAL = W_VALUE + D + 1;

  const E_CELL  = 0;
  const E_PATCH = E_CELL + 4 * D;
  const E_TOTAL = E_PATCH + 262144 * D;

  return {
    D, K, P, PATCH_CH,
    denseWeightCount: W_TOTAL,
    embedWeightCount: E_TOTAL,
  };
}

export const LEAGUE_ARCHS: ArchConfig[] = [
  { name: 'compact',  D: 4,  color: '#44ff44' },
  { name: 'standard', D: 8,  color: '#4488ff' },
  { name: 'wide',     D: 16, color: '#ff4444' },
];
