// Shared Elo step for two ratings (A = P1 / focal player, B = opponent).
// scoreA: 1 win, 0 loss, 0.5 draw.

export function eloUpdatePair(ra, rb, scoreA, k) {
  if (k === undefined || k === null) k = 32;
  var expA = 1 / (1 + Math.pow(10, (rb - ra) / 400));
  var expB = 1 - expA;
  var scoreB = scoreA === 0.5 ? 0.5 : 1 - scoreA;
  return {
    a: ra + k * (scoreA - expA),
    b: rb + k * (scoreB - expB)
  };
}
