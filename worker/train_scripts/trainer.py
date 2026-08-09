"""
Real training engine. Every job that reaches here actually trains a real
PyTorch model on a real scikit-learn dataset: real forward passes, real
backward passes, real optimizer steps. There is no fake curve-generation
anywhere in this module.

This runs inside a single Celery task (see ../train.py). Any of the
worker processes consuming the shared Celery/Redis queue can pick up any
job -- that's the "distributed execution across multiple worker
processes" part of the system. Within one job, training itself currently
runs single-process; see docs/ddp-integration.md for the planned next
step of parallelizing a single job's training across multiple local
processes with PyTorch DistributedDataParallel.
"""
import io
import json

import numpy as np
import torch
from torch import nn

from job_store import append_epoch, publish_update, save_model, update_job_status
from redis_client import get_redis
from schemas import TaskType
from train_scripts.datasets import load_dataset
from train_scripts.model import ConfigurableMLP

r = get_redis()


def _make_loss_fn(loss_function: str, task_type: TaskType):
    if loss_function == "cross_entropy":
        # Expects raw logits + integer class labels -- softmax is baked
        # into nn.CrossEntropyLoss itself, so we deliberately do NOT apply
        # the model's own output_activation before this loss.
        base = nn.CrossEntropyLoss()

        def loss_fn(logits, target, model):
            return base(logits, target)

        return loss_fn

    # mse / mae: apply the model's configured output activation first,
    # then compare against the target (one-hot for classification,
    # the raw continuous value for regression).
    base = nn.MSELoss() if loss_function == "mse" else nn.L1Loss()

    def loss_fn(logits, target, model):
        pred = model.apply_output_activation(logits)
        if task_type == TaskType.classification:
            target = nn.functional.one_hot(target, num_classes=pred.shape[-1]).float()
        return base(pred, target)

    return loss_fn


def _iterate_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    n = X.shape[0]
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


def run_training(job: dict) -> dict:
    job_id = job["job_id"]
    epochs = int(job.get("epochs", 10))
    learning_rate = float(job.get("learning_rate", 0.001))
    batch_size = int(job.get("batch_size", 32))
    activation = job.get("activation")
    output_activation = job.get("output_activation")
    loss_function = job.get("loss_function")
    task_type = TaskType(job.get("task_type", "classification"))

    hidden_layers = job.get("hidden_layers", [])
    if isinstance(hidden_layers, str):
        hidden_layers = json.loads(hidden_layers) if hidden_layers else []

    dropout = job.get("dropout", [])
    if isinstance(dropout, str):
        dropout = json.loads(dropout) if dropout else []

    update_job_status(r, job_id, status="running")
    publish_update(r, job_id, {"type": "status", "status": "running"})

    try:
        split = load_dataset(job["dataset"], task_type)

        model = ConfigurableMLP(
            n_features=split.n_features,
            n_outputs=split.n_outputs,
            hidden_layers=hidden_layers,
            dropout=dropout,
            activation=activation,
            output_activation=output_activation,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = _make_loss_fn(loss_function, task_type)

        X_val_t = torch.from_numpy(split.X_val)
        y_val_t = torch.from_numpy(split.y_val)

        final_train_loss = final_val_loss = final_metric = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            batch_losses = []
            for X_batch, y_batch in _iterate_batches(split.X_train, split.y_train, batch_size, shuffle=True):
                X_t = torch.from_numpy(X_batch)
                y_t = torch.from_numpy(y_batch)

                optimizer.zero_grad()
                logits = model(X_t)
                loss = loss_fn(logits, y_t, model)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_loss = float(np.mean(batch_losses))

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = float(loss_fn(val_logits, y_val_t, model).item())

                if task_type == TaskType.classification:
                    preds = torch.argmax(val_logits, dim=-1)
                    metric = float((preds == y_val_t).float().mean().item())
                else:
                    preds = model.apply_output_activation(val_logits)
                    ss_res = float(((y_val_t - preds) ** 2).sum().item())
                    ss_tot = float(((y_val_t - y_val_t.mean()) ** 2).sum().item())
                    metric = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    metric = max(0.0, min(metric, 1.0))  # clamp for display

            final_train_loss, final_val_loss, final_metric = train_loss, val_loss, metric

            epoch_data = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(metric, 4),
            }
            append_epoch(r, job_id, epoch_data)
            publish_update(r, job_id, {"type": "epoch", "data": epoch_data})

        final_accuracy = round(final_metric, 4)
        final_loss = round(final_val_loss, 4)

        # Save the trained weights (state_dict, not the whole pickled model
        # object -- safer to load elsewhere) as model.pt bytes, into Redis
        # under a short-TTL key. Only a "model_ready" flag goes out over
        # the pub/sub update channel; the backend serves the actual bytes
        # over a plain HTTP GET so they're never duplicated across every
        # websocket listener on this job.
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        save_model(r, job_id, buffer.getvalue())

        update_job_status(
            r, job_id, status="completed",
            final_accuracy=final_accuracy, final_loss=final_loss,
            model_ready=1,
        )
        publish_update(r, job_id, {
            "type": "status",
            "status": "completed",
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
            "model_ready": True,
        })

        return {
            "job_id": job_id,
            "status": "completed",
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
        }

    except Exception as exc:  # noqa: BLE001 -- surface any failure to the client
        update_job_status(r, job_id, status="failed", error=str(exc))
        publish_update(r, job_id, {"type": "status", "status": "failed", "error": str(exc)})
        raise
