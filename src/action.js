// Shared action utilities for RL algorithms.
// Pure functions: no model, no optimizer, no state.

/**
 * Flatten an array of Float32Array states into a single Float32Array
 * suitable for tf.tensor2d construction.
 */
export function flattenStates(states, boardSize) {
  var n = states.length;
  var flat = new Float32Array(n * boardSize);
  for (var i = 0; i < n; i++) {
    flat.set(states[i], i * boardSize);
  }
  return flat;
}

/**
 * Apply mask to raw logits and return normalized probabilities.
 * Uses numerically stable softmax: subtract max before exp.
 * Falls back to uniform over valid moves if all masked logits are -Inf.
 */
export function maskedSoftmax(logits, mask) {
  var n = logits.length;
  var probs = new Float32Array(n);

  // Find max of valid logits for numerical stability
  var maxVal = -Infinity;
  for (var i = 0; i < n; i++) {
    if (mask[i] > 0 && logits[i] > maxVal) maxVal = logits[i];
  }

  // If no valid moves, return zeros
  if (maxVal === -Infinity) return probs;

  // Compute exp(logit - max) for valid moves
  var sum = 0;
  for (var i = 0; i < n; i++) {
    if (mask[i] > 0) {
      probs[i] = Math.exp(logits[i] - maxVal);
      sum += probs[i];
    }
  }

  // Normalize
  if (sum > 1e-8) {
    for (var i = 0; i < n; i++) probs[i] /= sum;
  } else {
    // Fallback: uniform over valid moves
    var validCount = 0;
    for (var i = 0; i < n; i++) if (mask[i] > 0) validCount++;
    if (validCount > 0) {
      for (var i = 0; i < n; i++) probs[i] = mask[i] > 0 ? 1.0 / validCount : 0;
    }
  }

  return probs;
}

/**
 * Sample an action index from a probability distribution.
 */
export function sampleFromProbs(probs) {
  var r = Math.random();
  var cumSum = 0;
  for (var i = 0; i < probs.length; i++) {
    cumSum += probs[i];
    if (r < cumSum) return i;
  }
  // Fallback: return last valid index
  for (var i = probs.length - 1; i >= 0; i--) {
    if (probs[i] > 0) return i;
  }
  return 0;
}

/**
 * Compute log probability of a specific action given logits and mask.
 * Uses the same masked softmax internally.
 */
export function logProbOfAction(logits, mask, action) {
  var probs = maskedSoftmax(logits, mask);
  return Math.log(probs[action] + 1e-8);
}
