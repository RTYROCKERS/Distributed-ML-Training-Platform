# Roadmap: PyTorch DistributedDataParallel (DDP)

**Status: planned, not yet implemented.** Today, each training job runs as
a single Celery task executing a single-process PyTorch training loop
(`worker/train_scripts/trainer.py`). This document describes the design
for parallelizing *one job's* training across multiple local processes
with `torch.nn.parallel.DistributedDataParallel`, and is explicit about
what does and doesn't change.

## What DDP does *not* change

Job-level distribution — which of the (possibly many) Celery workers
picks up a given job, and how the two active-active FastAPI nodes accept
and report on jobs — is already real and stays exactly as-is. DDP is
additive: it's about how *one already-assigned job* uses the compute on
the single machine that's running it. It doesn't touch `main.py`,
`job_store.py`, the Celery broker config, or the active-active design at
all.

## What DDP requires that a single `import torch` line doesn't

1. **Process group rendezvous.** Every participating process must call
   `torch.distributed.init_process_group(backend=...)` and agree on a
   `MASTER_ADDR` / `MASTER_PORT`, its own `rank`, and the total
   `world_size`. This coordination is separate from, and unrelated to,
   Celery's job queue.
2. **A launcher.** The Celery task itself won't train directly — it will
   call `torch.multiprocessing.spawn(...)` (or shell out to `torchrun`)
   to fork `N` local processes on that worker's machine, where `N` is
   however many CPU cores / GPUs that one machine has.
3. **Data sharding.** Each rank needs its own shard of each batch via
   `torch.utils.data.distributed.DistributedSampler` — today's trainer
   feeds the whole batch to a single process.
4. **Gradient sync + rank-0 reporting.** DDP's all-reduce keeps model
   replicas in sync automatically once the process group exists, but only
   rank 0 should call back into `job_store.append_epoch` /
   `publish_update` — otherwise every rank would publish duplicate epoch
   events over the same Redis pub/sub channel.

## Scope of the first version

The first DDP pass is **single-node**: one Celery task spawns and
coordinates `N` local processes on the one machine that picked up the
job. It does not span multiple Celery workers/machines for a single job —
that (true multi-node DDP, coordinating ranks across separate machines)
is a further phase, since it needs a rendezvous mechanism that works
across hosts rather than `localhost`, and is out of scope until the
single-node version is working and there's a concrete need for it.

## Where it plugs in

`trainer.run_training()` keeps its current signature and Redis contract.
The change is internal to how it executes: instead of running the
epoch loop directly, it becomes the entrypoint that spawns `N` worker
processes, each running (effectively) today's epoch loop against its own
data shard and a DDP-wrapped model, with rank 0 doing the Redis
publishing. No change is needed to `job_store.py`, `main.py`, the
WebSocket protocol, or the frontend.
