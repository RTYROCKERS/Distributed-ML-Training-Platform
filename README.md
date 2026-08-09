# Distributed ML Training Platform

A FastAPI + Celery + Redis + WebSockets system for submitting "training
jobs" and streaming their progress back live. The backend runs as **two
independent, active-active instances** (one on Render, one on Railway)
sharing a single Redis instance, so either node can accept a request and
either node can serve status/updates for it.

The neural network training itself is real: a PyTorch MLP, architected
from whatever the request specifies (layer widths, activations,
per-layer dropout, output activation, loss function, learning rate,
batch size), is actually trained with real gradient updates on one of
five real scikit-learn datasets. See [What actually
trains](#what-actually-trains) for exactly what happens inside a job.
The point of this project is both halves: the distributed-systems path a
request takes, *and* a genuine (if deliberately small) training loop at
the end of it.

```
Browser (round-robins requests)
   │
   ├──► FastAPI on Render ─┐
   │                       ├──► shared Redis (queue, job state, pub/sub) ◄── Celery worker(s)
   └──► FastAPI on Railway ┘
```

## Contents

- [Running locally](#running-locally)
- [API](#api)
- [Active-active deployment: how and why](#active-active-deployment-how-and-why)
- [Deploying](#deploying)
- [What actually trains](#what-actually-trains)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)

## Running locally

You need Redis, Python, and Node 18+ (for the frontend). Everything talks
to one local Redis, same as production.

```bash
# 1. Start Redis yourself, e.g.
redis-server

# 2. API
cd backend/app
cp ../.env.example .env   # REDIS_URL=redis://localhost:6379/0
pip install -r ../requirements.txt
uvicorn main:app --reload

# 3. Worker (separate terminal)
cd worker
cp .env.example .env
pip install -r requirements.txt
celery -A train:celery_app worker --loglevel=info

# 4. Frontend (separate terminal)
cd frontend/model_trainer
cp .env.example .env      # VITE_API_URLS=http://localhost:8000 by default
npm install
npm run dev
```

Open the printed Vite URL (usually `http://localhost:5173`). Submit a run
and watch the epoch updates stream in.

## API

**`POST /train`**

```jsonc
{
  "job_name": "iris-baseline-v1",
  "dataset": "iris",                  // iris | wine | breast_cancer | digits | california_housing
  "hidden_layers": [64, 32, 16],      // up to 6 layers, up to 100 neurons each
  "activation": "relu",               // relu | sigmoid | tanh | leaky_relu
  "output_activation": "softmax",     // softmax | sigmoid | linear
  "loss_function": "cross_entropy",   // cross_entropy | mse | mae
  "dropout": [0.2, 0.0, 0.3],         // one rate per hidden layer, 0.0–0.9
  "epochs": 10,                       // up to 20
  "learning_rate": 0.001,             // 0.00001–1.0, optional, defaults to 0.001
  "batch_size": 32                    // 4–128, optional, defaults to 32
}
```

`dropout` must have exactly one entry per `hidden_layers` entry.
`california_housing` is a regression dataset, so `cross_entropy` is
rejected for it (it needs discrete class labels, not a continuous
target) — use `mse` or `mae` instead. Requests that exceed any of the
limits above, have mismatched list lengths, or pair a regression dataset
with `cross_entropy`, are rejected with `422` and a specific message —
nothing silently clamps or truncates.

Both `learning_rate` and `batch_size` are real, user-controlled
hyperparameters — they're passed straight to the `Adam` optimizer and
the mini-batch loop in `worker/train_scripts/trainer.py`.

Response:

```json
{ "job_id": "3e1f...c9", "status": "queued", "node": "render" }
```

`job_id` is a UUID generated at accept time, on whichever node accepted
the request.

**`GET /jobs/{job_id}`** — current status, config, and every epoch
recorded so far. Works identically regardless of which node handled the
original `POST /train`.

**`WS /ws/jobs/{job_id}`** — on connect, replays any epochs already
recorded (so a client that connects mid-run, or reconnects, isn't
missing history), then streams new `{"type": "epoch", ...}` messages as
the worker publishes them, and a final `{"type": "status", "status":
"completed" | "failed", ...}` message before closing.

## Active-active deployment: how and why

This is the core distributed-systems feature of the project, and the
part worth being able to explain in detail.

**The setup.** The exact same FastAPI app is deployed twice — once on
Render, once on Railway — both serving live traffic at the same time
(active-active, not active-passive/failover). Both point at one shared
Redis instance (Upstash's free tier works well here since it's reachable
from both platforms over the public internet). One or more Celery
workers, running anywhere that can reach that same Redis, consume the
task queue.

**Why UUID job IDs avoid collisions.** `job_id` is generated with
`uuid4()` *at accept time*, independently, on whichever node happens to
receive the `POST /train`. Neither node knows or cares what the other is
doing. A UUIDv4 has 122 bits of randomness — the two nodes would need to
independently roll the exact same 122-bit value to collide, which is
close enough to impossible that it isn't a real risk at any load this
system will see. Contrast this with an auto-incrementing ID: two nodes
each keeping their own counter would immediately hand out duplicate IDs
(both start their first job at `id=1`), and coordinating a single shared
counter across two independent deployments means either a single point
of failure or the same kind of shared-state problem Redis already solves
for us. UUIDs sidestep the coordination problem entirely by making
coordination unnecessary.

**Why job state has to live in shared Redis, not local memory.** If
`main.py` kept job status and epoch history in an in-process dict, that
dict would only exist on the node that happened to handle the original
request. A `GET /jobs/{id}` or a websocket connection that landed on the
*other* node (which, with round-robin traffic, is roughly half of them)
would get a 404 for a job that is very much running. Instead, every read
and write of job state — the initial config, status transitions, each
epoch's metrics — goes through Redis (`job_store.py`), keyed by
`job_id`. Because both nodes and the worker(s) all talk to the same
Redis instance, it doesn't matter which node accepted the job or which
node is now being asked about it: the data is in one place both can see.
The same goes for the live websocket stream — it's backed by a Redis
pub/sub channel (`job:{job_id}:updates`), not an in-memory queue, so a
socket opened on either node picks up messages published by a worker
that has no idea (and doesn't need to know) which node is watching.

**What would break if either of those were missing.**
- *Auto-increment IDs instead of UUIDs:* duplicate job IDs across nodes
  almost immediately, silently overwriting or corrupting unrelated jobs'
  state in Redis, or requiring a centralized ID-issuing service that
  reintroduces a single point of failure.
- *Local memory instead of shared Redis:* roughly half of all
  `GET`/`WS` requests (whichever land on the node that didn't accept the
  job) return "not found" or hang forever, even though the job is
  actively running — the system would only ever have worked by accident
  in a single-instance deployment, and "active-active" would really just
  be "active with an unreliable decoy".

**A nice side effect: workers scale independently.** Because task
dispatch also goes through the same shared Redis (as the Celery broker),
you can run any number of Celery workers, on either platform or both,
and Celery's competing-consumers behavior spreads work across them
automatically — no code change, no awareness on the worker's part of how
many peers it has.

## Deploying

You need one shared Redis reachable from both platforms — [Upstash's
free tier](https://upstash.com) works well.

**Render (backend instance A)**
1. New Web Service → point at this repo, root directory `backend/`.
2. It'll pick up `backend/render.yaml` as a Blueprint.
3. Env vars: `REDIS_URL` (your Upstash URL), `NODE_NAME=render`,
   `ALLOWED_ORIGINS` (your frontend's deployed origin, or `*` for a demo).
4. Health check path: `/health`.

**Railway (backend instance B)**
1. New Service → point at this repo, root directory `backend/`.
2. It'll pick up `backend/railway.json`.
3. Same env vars as above, but `NODE_NAME=railway`.

**Worker**
Run at least one Celery worker somewhere that can reach the same Redis —
Railway (`worker/railway.json`) is the simplest option, and you can also
run one on Render (`worker/render.yaml`, as a Background Worker) if you
want to demonstrate workers running on both platforms too. Env var:
`REDIS_URL`, same value as the backend services.

**Frontend**
Deploy `frontend/model_trainer` anywhere that serves a static Vite build
(Render Static Site, Railway, Vercel, Netlify...). Set
`VITE_API_URLS=https://your-app.onrender.com,https://your-app.up.railway.app`
at build time — this is the list the frontend round-robins across for
every job submission, and what makes the "submitted via / watching via"
badges in the UI meaningful.

## What actually trains

Every job trains for real, end to end:

1. **`worker/train_scripts/datasets.py`** loads one of five real
   scikit-learn datasets (`load_iris`, `load_wine`,
   `load_breast_cancer`, `load_digits`, or `fetch_california_housing`),
   does a real train/validation split, and fits a `StandardScaler` on
   the training split only.
2. **`worker/train_scripts/model.py`** builds a real `torch.nn.Module`
   — an MLP whose depth, width, per-layer dropout, and activation are
   read directly from the job's config. No architecture is hard-coded.
3. **`worker/train_scripts/trainer.py`** runs a real training loop:
   mini-batches of the requested `batch_size`, a real `Adam` optimizer
   at the requested `learning_rate`, real `backward()` calls, and a
   real validation pass every epoch. The loss function
   (`cross_entropy` / `mse` / `mae`) is wired to the actual PyTorch loss
   classes, matched against the dataset's task type (classification vs.
   regression). After each epoch it writes the real train/val loss and
   the real metric (accuracy for classification, R² for regression) to
   Redis and publishes it on the job's pub/sub channel — which is what
   the WebSocket in `main.py` is streaming out live.

Four of the five datasets (`iris`, `wine`, `breast_cancer`, `digits`)
ship inside scikit-learn and need no network access at runtime.
`california_housing` is fetched once via `sklearn.datasets.fetch_california_housing`
and cached locally afterward (`~/scikit_learn_data`), so it needs
outbound network access the first time it's used on a given machine.

What's still deliberately simple, and why: training runs single-process
per job (see [Roadmap](#roadmap) for the planned DDP step), the dataset
list is fixed at five rather than arbitrary uploads, and there's no
model/checkpoint persistence beyond a job's Redis TTL — this project's
focus is the distributed request/orchestration path plus a real (if
intentionally small) training loop at the end of it, not a general
AutoML platform.

## Project layout

```
backend/
  app/
    main.py          FastAPI app: POST /train, GET /jobs/{id}, WS /ws/jobs/{id}
    schemas.py        TrainRequest validation (layer/dropout/epoch/hyperparam
                        limits, dataset <-> loss_function compatibility)
    job_store.py       Redis-backed job state (hash + list + pub/sub) -- the
                        piece that makes active-active correct
    celery_app.py       Celery app pointed at the shared Redis
    redis_client.py
  Procfile / render.yaml / railway.json

worker/
  train.py              Celery task entrypoint -- hands the job straight to
                          train_scripts/trainer.py
  train_scripts/
    datasets.py           loads one of 5 real scikit-learn datasets, splits
                            + scales them
    model.py               builds a real torch.nn.Module MLP from the job's
                            architecture config
    trainer.py             real training loop: mini-batches, Adam, backward(),
                            real per-epoch metrics published to Redis
  schemas.py / job_store.py / celery_app.py / redis_client.py   (kept in
    sync with backend/app/'s copies -- see note at the top of each file)
  Procfile / railway.json / render.yaml

frontend/model_trainer/
  src/
    App.jsx                page layout
    config.js               dataset list, limits -- mirrors backend/app/schemas.py
    api.js                   round-robin client across configured backend URLs
    components/
      ConfigForm.jsx, LayerEditor.jsx     the network config form
      NetworkDiagram.jsx                   live SVG architecture diagram
      TrainingMonitor.jsx, EpochChart.jsx   live WebSocket-driven training view
      NodeBadge.jsx                         render/railway node indicator
```

## Roadmap

- **PyTorch DistributedDataParallel (DDP).** Right now, one Celery task
  runs one job's training as a single process. The planned next step is
  having that task spawn and coordinate multiple local processes (rank/
  world_size, `DistributedSampler`, rank-0 metric reporting) via DDP, so
  a single job's training itself is parallelized across a worker
  machine's available cores/GPUs — separate from, and in addition to,
  the existing Celery-level distribution of *which* job goes to *which*
  worker. Full design and scope: [`docs/ddp-integration.md`](docs/ddp-integration.md).