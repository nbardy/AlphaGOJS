// Runtime registry controls high-level execution topology selection.
// A runtime maps to an underlying pipeline kind plus tuned options.

// Order: default/smooth GPU mode first; full resident for max throughput (can stall / feel janky).
var RUNTIME_TYPES = [
  {
    id: 'single_gpu_phased',
    label: 'Single GPU phased (default)',
    pipelineKind: 'gpu_worker',
    options: {
      pipelineTypeOverride: 'single_gpu_phased',
      trainBatchSize: 512,
      trainInterval: 30,
      snapshotEveryTicks: 2,
      maxTickBatch: 4,
      maxQueuedSteps: 512,
      pauseTicksWhenTraining: true,
      trainInFlightQueueCap: 0,
      uiSnapshotMaxGames: 48
    }
  },
  {
    id: 'full_gpu_resident',
    label: 'Full GPU resident (max throughput)',
    pipelineKind: 'gpu_worker',
    options: {
      pipelineTypeOverride: 'full_gpu_resident',
      trainBatchSize: 512,
      trainInterval: 30,
      snapshotEveryTicks: 1,
      maxTickBatch: 16,
      maxQueuedSteps: 4096,
      queueSoftCapFraction: 0.75,
      pauseTicksWhenTraining: false,
      trainInFlightQueueCap: 64,
      uiSnapshotMaxGames: 48
    }
  },
  {
    id: 'cpu_actors_gpu_learner',
    label: 'CPU Actors + GPU Learner',
    pipelineKind: 'cpu',
    options: {
      pipelineTypeOverride: 'cpu_actors_gpu_learner',
      trainBatchSize: 512,
      trainInterval: 30
    }
  },
  // Legacy aliases kept for backwards compatibility with existing scripts.
  { id: 'cpu', label: 'CPU (legacy)', aliasOf: 'cpu_actors_gpu_learner' },
  { id: 'gpu', label: 'GPU (legacy)', aliasOf: 'full_gpu_resident' },
  { id: 'gpu_worker', label: 'GPU Worker (legacy)', aliasOf: 'single_gpu_phased' }
];

function findRuntimeSpec(id) {
  for (var i = 0; i < RUNTIME_TYPES.length; i++) {
    if (RUNTIME_TYPES[i].id === id) return RUNTIME_TYPES[i];
  }
  return null;
}

function shallowClone(obj) {
  if (!obj) return {};
  var out = {};
  var keys = Object.keys(obj);
  for (var i = 0; i < keys.length; i++) out[keys[i]] = obj[keys[i]];
  return out;
}

export function resolveRuntimeSpec(runtimeType) {
  var id = runtimeType || 'single_gpu_phased';
  var spec = findRuntimeSpec(id) || findRuntimeSpec('single_gpu_phased');
  var seen = {};
  while (spec && spec.aliasOf) {
    if (seen[spec.id]) break;
    seen[spec.id] = 1;
    spec = findRuntimeSpec(spec.aliasOf) || findRuntimeSpec('single_gpu_phased');
  }
  return {
    id: spec.id,
    label: spec.label,
    pipelineKind: spec.pipelineKind,
    options: shallowClone(spec.options)
  };
}

export function listRuntimeTypes() {
  var out = [];
  for (var i = 0; i < RUNTIME_TYPES.length; i++) {
    out.push({ id: RUNTIME_TYPES[i].id, label: RUNTIME_TYPES[i].label });
  }
  return out;
}
