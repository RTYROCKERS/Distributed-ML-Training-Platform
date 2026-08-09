"""
Real dataset loading via scikit-learn. Nothing here is simulated: every
dataset is actually loaded into memory, split into train/validation, and
feature-scaled before training.

Five standard, small, well-known datasets are supported -- picked because
they load instantly from scikit-learn (four ship with the library itself;
california_housing downloads once via sklearn's fetcher and is cached
locally afterwards), covering both classification and regression so the
loss_function / output_activation choices in the form are meaningful.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from schemas import Dataset, TaskType

_LOADERS = {
    Dataset.iris: load_iris,
    Dataset.wine: load_wine,
    Dataset.breast_cancer: load_breast_cancer,
    Dataset.digits: load_digits,
    Dataset.california_housing: fetch_california_housing,
}


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    n_features: int
    n_outputs: int
    task_type: TaskType


def load_dataset(dataset: Dataset, task_type: TaskType, val_size: float = 0.2, seed: int = 42) -> DatasetSplit:
    loader = _LOADERS[dataset]
    bunch = loader()
    X = bunch.data.astype(np.float32)
    y = bunch.target

    stratify = y if task_type == TaskType.classification else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=stratify
    )

    # Fit scaling on train only, apply to both -- standard practice to
    # avoid leaking validation-set statistics into training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    if task_type == TaskType.classification:
        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)
        n_outputs = int(len(np.unique(y)))
    else:
        y_train = y_train.astype(np.float32).reshape(-1, 1)
        y_val = y_val.astype(np.float32).reshape(-1, 1)
        n_outputs = 1

    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        n_features=X.shape[1],
        n_outputs=n_outputs,
        task_type=task_type,
    )
