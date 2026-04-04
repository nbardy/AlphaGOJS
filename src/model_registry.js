import { DenseModel } from './dense_model';
import { SpatialModel } from './spatial_model';
import { SpatialLiteModel } from './spatial_lite_model';
import { SpatialPyramidGradModel } from './spatial_pyramid_grad_model';

// Model registry: single source of truth for UI labels and constructors.
// Add new model types here only.
var MODEL_TYPES = [
  {
    id: 'spatial_lite',
    label: 'Spatial lite (sep conv)',
    create: function (rows, cols) {
      return new SpatialLiteModel(rows, cols);
    }
  },
  {
    id: 'pyramid_grad',
    label: 'Pyramid + dilated + neighbor stem',
    create: function (rows, cols) {
      return new SpatialPyramidGradModel(rows, cols);
    }
  },
  { id: 'dense', label: 'Dense', create: function (rows, cols) { return new DenseModel(rows, cols); } },
  { id: 'spatial', label: 'Spatial (deep res, slow)', create: function (rows, cols) { return new SpatialModel(rows, cols); } }
];

export function listModelTypes() {
  var out = [];
  for (var i = 0; i < MODEL_TYPES.length; i++) {
    out.push({ id: MODEL_TYPES[i].id, label: MODEL_TYPES[i].label });
  }
  return out;
}

export function createModel(type, rows, cols) {
  for (var i = 0; i < MODEL_TYPES.length; i++) {
    if (MODEL_TYPES[i].id === type) {
      return MODEL_TYPES[i].create(rows, cols);
    }
  }
  throw new Error('Unknown model type: ' + type);
}
