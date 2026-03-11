// Message types shared by UI thread and GPU owner worker.

export var MSG = {
  INIT: 'init',
  READY: 'ready',
  TICK: 'tick',
  TICK_RESULT: 'tick_result',
  RESTART: 'restart',
  DISPOSE: 'dispose',
  ERROR: 'error',
  INFER_ACTION: 'infer_action',
  INFER_ACTION_RESULT: 'infer_action_result'
};

export function makeInit(config) {
  return { type: MSG.INIT, config: config };
}

export function makeTick(steps) {
  return { type: MSG.TICK, steps: steps };
}

export function makeRestart(config) {
  return { type: MSG.RESTART, config: config };
}

export function makeInferAction(requestId, state, mask) {
  return {
    type: MSG.INFER_ACTION,
    requestId: requestId,
    state: state,
    mask: mask
  };
}

