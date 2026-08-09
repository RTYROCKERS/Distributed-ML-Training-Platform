import os

import redis
from dotenv import load_dotenv

load_dotenv()

# One Redis instance, one URL. It is the broker for Celery, the Celery
# result backend, the job-state store, AND the pub/sub bus for live epoch
# updates. Everything -- API instances and workers, on either platform --
# points at this same URL. See README section "Why one shared Redis".
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_client = None


def get_redis() -> "redis.Redis":
    global _client
    if _client is None:
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _client
