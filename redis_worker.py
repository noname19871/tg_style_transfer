import os
from typing import List

import redis
from rq import Connection, Queue, SimpleWorker

listen: List[str] = ["default"]

REDIS_ENDPOINT = os.getenv("REDIS_ENDPOINT", default="localhost")
REDIS_PORT = os.getenv("REDIS_PORT", default="6379")

redis_url: str = "redis://" + REDIS_ENDPOINT + ":" + REDIS_PORT

conn = redis.from_url(redis_url)

if __name__ == "__main__":
    with Connection(conn):
        worker = SimpleWorker(list(map(Queue, listen)))
        worker.work()
