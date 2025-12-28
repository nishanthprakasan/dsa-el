from __future__ import annotations
import os
import json
import numpy as np
from typing import List
from upstash_redis import Redis
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .core import CacheEntry

class UpstashBackend:
    def __init__(self):
        self.redis = Redis(
            url=os.getenv("UPSTASH_REDIS_REST_URL"),
            token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
        )

    def bucket_key(self, semhash: str) -> str:
        return f"semhash:{semhash}"

    # ---------- serialization ----------
    def serialize_entry(self, entry: CacheEntry) -> dict:
        return {
            "input_sig": entry.input_sig,
            "output_fp": entry.output_fp,
            "embedding": entry.embedding.tolist(),
            "payload_ptr": entry.payload_ptr,
            "model_version": entry.model_version,
            "created_at": entry.created_at,
            "confidence": entry.confidence,
            "usage_count": entry.usage_count,
        }

    def deserialize_entry(self, data: dict) -> CacheEntry:
        from .core import CacheEntry
        return CacheEntry(
            input_sig=data["input_sig"],
            output_fp=data["output_fp"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            payload_ptr=data["payload_ptr"],
            model_version=data.get("model_version", "1.0"),
            confidence=data.get("confidence"),
            usage_count=data.get("usage_count", 0),
        )

    # ---------- bucket ops ----------
    def load_bucket(self, semhash: str) -> List[dict]:
        raw = self.redis.get(self.bucket_key(semhash))
        if raw is None:
            return []
        return json.loads(raw)

    def save_bucket(self, semhash: str, bucket: List[dict]):
        self.redis.set(self.bucket_key(semhash), json.dumps(bucket))

    def append_entry(self, semhash: str, entry: dict):
        bucket = self.load_bucket(semhash)
        bucket.append(entry)
        self.save_bucket(semhash, bucket)
