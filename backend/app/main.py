import asyncio
import json
import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from celery_app import celery_app
from job_store import get_epochs, get_job, get_model, save_job, updates_channel
from redis_client import get_redis
from schemas import DATASET_DIMS, TrainRequest, TrainResponse

# Identifies which deployed instance served a given request. Set via env
# var on each platform (NODE_NAME=render / NODE_NAME=railway). Purely for
# demo/debugging visibility into the active-active setup -- see README.
NODE_NAME = os.getenv("NODE_NAME", "local")

app = FastAPI(title="Distributed ML Training Platform", version="1.0.0")

_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins == "*" else [o.strip() for o in _allowed_origins.split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

r = get_redis()


@app.get("/health")
def health():
    """Cheap liveness probe; also reports which node answered."""
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"status": "ok", "node": NODE_NAME, "redis_ok": redis_ok}


@app.post("/train", response_model=TrainResponse)
def train_model(payload: TrainRequest):
    # job_id is a UUID generated right here, independently on whichever
    # node accepts the request. Two nodes generating UUIDs concurrently
    # have effectively zero chance of collision, so no coordination
    # between Render and Railway is needed to hand out IDs safely.
    job_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    job_record = {
        "job_id": job_id,
        "job_name": payload.job_name,
        "dataset": payload.dataset.value,
        "hidden_layers": json.dumps(payload.hidden_layers),
        "activation": payload.activation.value,
        "output_activation": payload.output_activation.value,
        "loss_function": payload.loss_function.value,
        "dropout": json.dumps(payload.dropout),
        "epochs": payload.epochs,
        "learning_rate": payload.learning_rate,
        "batch_size": payload.batch_size,
        "task_type": DATASET_DIMS[payload.dataset]["task_type"].value,
        "status": "queued",
        "node": NODE_NAME,
        "created_at": now,
        "updated_at": now,
    }

    # Written to the shared Redis, not to any local dict/variable. This is
    # the other half of what makes active-active safe: a status or
    # websocket request that lands on the OTHER node a moment later reads
    # this same record, because both nodes talk to the same Redis.
    save_job(r, job_id, job_record)

    celery_app.send_task("train_model", args=[job_record])

    return TrainResponse(job_id=job_id, status="queued", node=NODE_NAME)


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job(r, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")

    job["hidden_layers"] = json.loads(job.get("hidden_layers", "[]"))
    job["dropout"] = json.loads(job.get("dropout", "[]"))
    job["epochs_completed"] = get_epochs(r, job_id)
    job["served_by_node"] = NODE_NAME
    return job


@app.get("/jobs/{job_id}/model")
def download_model(job_id: str):
    """Serves the trained model.pt for a completed job.

    The websocket/pub-sub side only ever announces that a model is ready
    (see job_store.save_model / trainer.py) -- the actual bytes live in
    Redis under a short-TTL key and are streamed out here as a normal
    file download, on whichever node happens to answer, same as every
    other job read in this file.
    """
    job = get_job(r, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")

    model_bytes = get_model(r, job_id)
    if model_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="model not available for this job (not finished yet, or the download window expired)",
        )

    safe_name = "".join(c for c in job.get("job_name", "model") if c.isalnum() or c in "-_") or "model"
    filename = f"{safe_name}-{job_id[:8]}.pt"
    return Response(
        content=model_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.websocket("/ws/jobs/{job_id}")
async def ws_job_updates(websocket: WebSocket, job_id: str):
    await websocket.accept()

    job = get_job(r, job_id)
    if job is None:
        await websocket.send_json({"type": "error", "message": f"job {job_id} not found"})
        await websocket.close()
        return

    # Tell the client which node is actually serving this socket. If the
    # job was submitted via the other node, this proves the point live.
    await websocket.send_json({"type": "node", "node": NODE_NAME})

    # Replay any epochs that already landed in Redis before this socket
    # connected (e.g. client reconnected, or connected to the other node
    # after the job started).
    for epoch_data in get_epochs(r, job_id):
        await websocket.send_json({"type": "epoch", "data": epoch_data})

    if job.get("status") in ("completed", "failed"):
        await websocket.send_json({"type": "status", "status": job.get("status"), "job": job})
        await websocket.close()
        return

    pubsub = r.pubsub()
    pubsub.subscribe(updates_channel(job_id))
    try:
        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message and message.get("type") == "message":
                data = message["data"]
                await websocket.send_text(data)
                try:
                    parsed = json.loads(data)
                    if parsed.get("type") == "status" and parsed.get("status") in ("completed", "failed"):
                        break
                except json.JSONDecodeError:
                    pass
            await asyncio.sleep(0.1)  # yield to the event loop between polls
    except WebSocketDisconnect:
        pass
    finally:
        pubsub.unsubscribe(updates_channel(job_id))
        pubsub.close()
        try:
            await websocket.close()
        except RuntimeError:
            pass  # already closed
