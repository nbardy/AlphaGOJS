import { Reinforce } from './reinforce';
import { PPO } from './ppo';
import { PPG } from './ppg';
import { SAC } from './sac';
import { StochasticMuZero } from './stochastic_muzero';

// Algorithm registry: single source of truth for UI labels and constructors.
// Add new algorithms here only.
var ALGO_TYPES = [
  { id: 'ppo', label: 'PPO', create: function (model) { return new PPO(model); } },
  { id: 'ppg', label: 'PPG', create: function (model) { return new PPG(model); } },
  { id: 'sac', label: 'Discrete SAC', create: function (model) { return new SAC(model); } },
  { id: 'muzero', label: 'Stochastic MuZero', create: function (model) { return new StochasticMuZero(model); } },
  { id: 'reinforce', label: 'REINFORCE', create: function (model) { return new Reinforce(model); } }
];

export function listAlgorithmTypes() {
  var out = [];
  for (var i = 0; i < ALGO_TYPES.length; i++) {
    out.push({ id: ALGO_TYPES[i].id, label: ALGO_TYPES[i].label });
  }
  return out;
}

export function createAlgorithm(type, model) {
  for (var i = 0; i < ALGO_TYPES.length; i++) {
    if (ALGO_TYPES[i].id === type) {
      return ALGO_TYPES[i].create(model);
    }
  }
  throw new Error('Unknown algorithm type: ' + type);
}
