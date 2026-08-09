"""
Celery worker entrypoint. Run with:

    celery -A train:celery_app worker --loglevel=info

Any number of these can run, on either platform (or both) -- they all
consume from the same shared Celery/Redis broker, so Celery's own
competing-consumers behavior spreads work across however many are
running without any extra coordination on our part. Whichever worker
picks up a given job trains it start to finish (see train_scripts/trainer.py
for the real training loop -- no simulated/dummy training happens here).
"""
from celery_app import celery_app
from train_scripts.trainer import run_training


@celery_app.task(name="train_model", bind=True)
def train_model(self, job: dict):
    return run_training(job)
