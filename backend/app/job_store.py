"""
All job state lives in Redis, keyed by job_id -- never in the memory of
whichever process happens to be handling a given HTTP request or Celery
task. This is what makes active-active work: it doesn't matter whether
Render or Railway accepted the job, and it doesn't matter whether the
worker that picks it up is the one "closest" to that node.

Keys used per job:
  job:{job_id}          -> hash: config + status fields
  job:{job_id}:epochs   -> list: one JSON string per completed epoch
  job:{job_id}:updates  -> pub/sub channel: live epoch + status events
  job:{job_id}:model     -> string: base64-encoded model.pt bytes (short TTL)

This file is intentionally duplicated (byte-for-byte) at worker/job_store.py.
"""
import base64
import datetime
import json
import os

JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", str(60 * 60 * 24)))  # 24h

# Deliberately much shorter than JOB_TTL_SECONDS. The model blob only needs
# to survive long enough for the client to fetch it once after the "model
# ready" event; there's no reason to keep a base64 copy of every trained
# model sitting in Redis for a full day.
MODEL_TTL_SECONDS = int(os.getenv("MODEL_TTL_SECONDS", str(60 * 30)))  # 30 min


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def epochs_key(job_id: str) -> str:
    return f"job:{job_id}:epochs"


def updates_channel(job_id: str) -> str:
    return f"job:{job_id}:updates"


def model_key(job_id: str) -> str:
    return f"job:{job_id}:model"


def save_job(r, job_id: str, data: dict) -> None:
    clean = {k: v for k, v in data.items() if v is not None}
    r.hset(job_key(job_id), mapping=clean)
    r.expire(job_key(job_id), JOB_TTL_SECONDS)


def get_job(r, job_id: str):
    data = r.hgetall(job_key(job_id))
    if not data:
        return None
    return data


def update_job_status(r, job_id: str, status: str, **extra) -> None:
    fields = {
        "status": status,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    fields.update({k: v for k, v in extra.items() if v is not None})
    r.hset(job_key(job_id), mapping=fields)
    r.expire(job_key(job_id), JOB_TTL_SECONDS)


def append_epoch(r, job_id: str, epoch_data: dict) -> None:
    r.rpush(epochs_key(job_id), json.dumps(epoch_data))
    r.expire(epochs_key(job_id), JOB_TTL_SECONDS)


def get_epochs(r, job_id: str):
    raw = r.lrange(epochs_key(job_id), 0, -1)
    return [json.loads(item) for item in raw]


def publish_update(r, job_id: str, payload: dict) -> None:
    r.publish(updates_channel(job_id), json.dumps(payload))


def save_model(r, job_id: str, model_bytes: bytes) -> None:
    # Redis clients here are configured with decode_responses=True (every
    # other key is text), so raw bytes get base64-encoded before storing
    # rather than opening a second binary-safe connection just for this.
    # The pub/sub "updates" channel only ever carries a small "model_ready"
    # flag -- never these bytes -- so live-update listeners aren't paying
    # for a multi-hundred-KB payload on every connected client/node.
    encoded = base64.b64encode(model_bytes).decode("ascii")
    r.set(model_key(job_id), encoded, ex=MODEL_TTL_SECONDS)


def get_model(r, job_id: str):
    encoded = r.get(model_key(job_id))
    if encoded is None:
        return None
    return base64.b64decode(encoded)
