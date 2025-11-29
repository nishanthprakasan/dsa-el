# redis_backend.py
import json
import redis
from typing import Optional, List
from .core import CacheEntry
from .utils import canonicalize_input

class RedisBackend:
    """
    Simple Redis backend mapping semhash -> JSON list of entries.
    Each entry stores metadata fields and a pointer to payload (S3 key or local blob).
    Use atomic Lua scripts or transactions in production for concurrency safety.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.client = redis.from_url(redis_url, decode_responses=True)

    def bucket_key(self, semhash: str) -> str:
        return f"semhash:{semhash}"

    def load_bucket(self, semhash: str) -> List[dict]:
        data = self.client.get(self.bucket_key(semhash))
        if not data: return []
        return json.loads(data)

    def save_bucket(self, semhash: str, bucket: List[dict]):
        # bucket is a list of serializable dicts
        self.client.set(self.bucket_key(semhash), json.dumps(bucket))

    def append_entry(self, semhash: str, entry: dict):
        # naive append: read -> modify -> write; for high-throughput use LIST or JSON module in Redis
        bucket = self.load_bucket(semhash)
        bucket.append(entry)
        self.save_bucket(semhash, bucket)

    def get_bucket_size(self, semhash: str) -> int:
        return len(self.load_bucket(semhash))
