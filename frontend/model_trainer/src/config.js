// Mirrors backend/app/schemas.py. Kept in sync by hand (small enough that
// a shared package would be overkill for this project). These are real
// scikit-learn datasets -- the worker actually loads and trains on them,
// nothing here is a placeholder.

export const DATASETS = [
  { value: "iris", label: "Iris", input: 4, output: 3, taskType: "classification" },
  { value: "wine", label: "Wine", input: 13, output: 3, taskType: "classification" },
  { value: "breast_cancer", label: "Breast Cancer Wisconsin", input: 30, output: 2, taskType: "classification" },
  { value: "digits", label: "Digits (8x8)", input: 64, output: 10, taskType: "classification" },
  { value: "california_housing", label: "California Housing", input: 8, output: 1, taskType: "regression" },
];

export const ACTIVATIONS = [
  { value: "relu", label: "ReLU" },
  { value: "sigmoid", label: "Sigmoid" },
  { value: "tanh", label: "Tanh" },
  { value: "leaky_relu", label: "Leaky ReLU" },
];

export const OUTPUT_ACTIVATIONS = [
  { value: "softmax", label: "Softmax" },
  { value: "sigmoid", label: "Sigmoid" },
  { value: "linear", label: "Linear" },
];

export const LOSS_FUNCTIONS = [
  { value: "cross_entropy", label: "Cross-Entropy" },
  { value: "mse", label: "Mean Squared Error" },
  { value: "mae", label: "Mean Absolute Error" },
];

export const MAX_HIDDEN_LAYERS = 6;
export const MAX_NEURONS_PER_LAYER = 100;
export const MAX_EPOCHS = 20;
export const MIN_DROPOUT = 0.0;
export const MAX_DROPOUT = 0.9;

// Pre-filled defaults -- genuinely editable now, not fixed server-side.
export const DEFAULT_LEARNING_RATE = 0.001;
export const DEFAULT_BATCH_SIZE = 32;
export const MIN_LEARNING_RATE = 0.00001;
export const MAX_LEARNING_RATE = 1.0;
export const MIN_BATCH_SIZE = 4;
export const MAX_BATCH_SIZE = 128;

// Diagram truncates any layer wider than this to first-3 / ... / last-3.
export const MAX_VISIBLE_NEURONS = 8;

export function datasetByValue(value) {
  return DATASETS.find((d) => d.value === value) || DATASETS[0];
}

// cross_entropy needs discrete class labels -- not valid for a regression
// dataset like california_housing. Mirrors the backend's model_validator.
export function lossFunctionsFor(datasetValue) {
  const ds = datasetByValue(datasetValue);
  if (ds.taskType === "regression") {
    return LOSS_FUNCTIONS.filter((l) => l.value !== "cross_entropy");
  }
  return LOSS_FUNCTIONS;
}

export function defaultHiddenLayer() {
  return { neurons: 32, dropout: 0.2 };
}
