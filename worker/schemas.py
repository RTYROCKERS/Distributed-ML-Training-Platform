"""
Request/response models for the training API.

This file is intentionally duplicated (byte-for-byte) at worker/schemas.py.
Backend and worker are deployed as two independent services, so rather than
publish an internal shared package, we keep one small file in sync in both
places. See the README for why.
"""
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class Dataset(str, Enum):
    iris = "iris"
    wine = "wine"
    breast_cancer = "breast_cancer"
    digits = "digits"
    california_housing = "california_housing"


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"


# Real dataset metadata. input/output dims are the actual feature count and
# number of classes (or 1, for regression) returned by scikit-learn -- the
# worker loads these datasets for real via sklearn.datasets, there is no
# placeholder here. task_type drives which loss functions are valid and how
# the trained model's output layer and final metric are interpreted.
DATASET_DIMS = {
    Dataset.iris: {"input": 4, "output": 3, "label": "Iris", "task_type": TaskType.classification},
    Dataset.wine: {"input": 13, "output": 3, "label": "Wine", "task_type": TaskType.classification},
    Dataset.breast_cancer: {"input": 30, "output": 2, "label": "Breast Cancer Wisconsin", "task_type": TaskType.classification},
    Dataset.digits: {"input": 64, "output": 10, "label": "Digits (8x8)", "task_type": TaskType.classification},
    Dataset.california_housing: {"input": 8, "output": 1, "label": "California Housing", "task_type": TaskType.regression},
}


class Activation(str, Enum):
    relu = "relu"
    sigmoid = "sigmoid"
    tanh = "tanh"
    leaky_relu = "leaky_relu"


class OutputActivation(str, Enum):
    softmax = "softmax"
    sigmoid = "sigmoid"
    linear = "linear"


class LossFunction(str, Enum):
    cross_entropy = "cross_entropy"
    mse = "mse"
    mae = "mae"


MAX_HIDDEN_LAYERS = 6
MAX_NEURONS_PER_LAYER = 100
MAX_EPOCHS = 20
MIN_DROPOUT = 0.0
MAX_DROPOUT = 0.9

# Hyperparameters are now genuinely user-controlled (see README) -- these
# are just the defaults pre-filled in the form and the bounds enforced by
# the validators below, not fixed constants baked into training.
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1.0
MIN_BATCH_SIZE = 4
MAX_BATCH_SIZE = 128


class TrainRequest(BaseModel):
    job_name: str = Field(..., min_length=1, max_length=80)
    dataset: Dataset
    hidden_layers: List[int] = Field(..., min_length=1)
    activation: Activation
    output_activation: OutputActivation
    loss_function: LossFunction
    dropout: List[float] = Field(..., min_length=1)
    epochs: int = Field(..., ge=1, le=MAX_EPOCHS)
    learning_rate: float = Field(DEFAULT_LEARNING_RATE, ge=MIN_LEARNING_RATE, le=MAX_LEARNING_RATE)
    batch_size: int = Field(DEFAULT_BATCH_SIZE, ge=MIN_BATCH_SIZE, le=MAX_BATCH_SIZE)

    @field_validator("hidden_layers")
    @classmethod
    def check_hidden_layers(cls, v: List[int]) -> List[int]:
        if len(v) > MAX_HIDDEN_LAYERS:
            raise ValueError(
                f"hidden_layers supports at most {MAX_HIDDEN_LAYERS} layers, got {len(v)}"
            )
        for n in v:
            if n < 1 or n > MAX_NEURONS_PER_LAYER:
                raise ValueError(
                    f"each hidden layer must have between 1 and {MAX_NEURONS_PER_LAYER} "
                    f"neurons, got {n}"
                )
        return v

    @field_validator("dropout")
    @classmethod
    def check_dropout_range(cls, v: List[float]) -> List[float]:
        for d in v:
            if d < MIN_DROPOUT or d > MAX_DROPOUT:
                raise ValueError(
                    f"dropout rates must be between {MIN_DROPOUT} and {MAX_DROPOUT}, got {d}"
                )
        return v

    @model_validator(mode="after")
    def check_dropout_matches_layers(self) -> "TrainRequest":
        if len(self.dropout) != len(self.hidden_layers):
            raise ValueError(
                "dropout must have exactly one entry per hidden layer: got "
                f"{len(self.hidden_layers)} hidden_layers but {len(self.dropout)} dropout values"
            )
        return self

    @model_validator(mode="after")
    def check_loss_matches_task_type(self) -> "TrainRequest":
        task_type = DATASET_DIMS[self.dataset]["task_type"]
        if task_type == TaskType.regression and self.loss_function == LossFunction.cross_entropy:
            raise ValueError(
                f"{self.dataset.value} is a regression dataset -- cross_entropy needs class "
                "labels, not a continuous target. Use mse or mae instead."
            )
        return self


class TrainResponse(BaseModel):
    job_id: str
    status: str
    node: str
