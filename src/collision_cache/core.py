import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import os

from .utils import canonicalize_input, input_signature, output_fingerprint

@dataclass
class CacheEntry:
    input_sig: str
    output_fp: str
    embedding: np.ndarray
    payload_ptr: str
    model_version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    confidence: Optional[float] = None
    usage_count: int = 0

class SemanticHasher:
    def __init__(self, emb_dim: int, n_bits: int = 128, seed: int = 42):
        self.emb_dim = emb_dim
        self.n_bits = n_bits
        rng = np.random.RandomState(seed)
        self.proj = rng.normal(size=(n_bits, emb_dim)).astype(np.float32)

    def hash_from_embedding(self, emb: np.ndarray) -> str:
        dots = self.proj.dot(emb)
        bits = (dots >= 0).astype(np.uint8)
        byte_arr = np.packbits(bits)
        return byte_arr.tobytes().hex()

class CollisionAwareSemanticCache:
    def __init__(self, emb_dim: int, hasher_bits: int = 128, seed: int = 42,
             sim_threshold: float = 0.92, max_bucket_size: int = 8,
             payload_resolver: Callable[[str], str] = lambda ptr: ptr,
             backend: Optional[str] = None):
    
        self.buckets = {}   # ALWAYS exists (mirror index)
        self.hasher = SemanticHasher(emb_dim, n_bits=hasher_bits, seed=seed)
        self.lock = threading.Lock()
        self.sim_threshold = sim_threshold
        self.max_bucket_size = max_bucket_size
        self.payload_resolver = payload_resolver
        self.stats = {"lookups":0,"hits":0,"verified_hits":0,"misses":0,"collisions_detected":0,"stores":0}

        if backend == "redis":
            from .redis_backend import RedisBackend  # lazy import
            import os
            redis_url = os.getenv("REDIS_URL")
            redis_password = os.getenv("REDIS_PASSWORD")
            self.backend = RedisBackend(redis_url, redis_password)

        elif backend == "upstash":
            from .upstash_backend import UpstashBackend  # lazy import
            self.backend = UpstashBackend()

        else:
            self.backend = None


    def store(self, prompt, output_text, embedding, payload_ptr, model_version="1.0", confidence=None):
        semhash = self.hasher.hash_from_embedding(embedding)
        insig = input_signature(prompt)
        outfp = output_fingerprint(output_text)

        entry = CacheEntry(
            input_sig=insig,
            output_fp=outfp,
            embedding=embedding.astype(np.float32),
            payload_ptr=payload_ptr,
            model_version=model_version,
            confidence=confidence
        )

        with self.lock:
            # 🔹 Phase-1 behavior (RESTORED)
            self.buckets.setdefault(semhash, []).append(entry)

            # 🔹 Redis persistence (NEW, passive)
            if self.backend:
                self.backend.append_entry(
                    semhash,
                    self.backend.serialize_entry(entry)
                )

            self.stats["stores"] += 1



    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0: return -1.0
        return float(np.dot(a,b)/(na*nb))

    def lookup(self, prompt, embedding_fn):
        self.stats["lookups"] += 1
        _ = canonicalize_input(prompt)
        emb = embedding_fn(prompt)
        semhash = self.hasher.hash_from_embedding(emb)

        with self.lock:
            # ---------- 1. SAME HASH BUCKET ----------
            bucket = self.buckets.get(semhash)

            if bucket:
                insig = input_signature(prompt)

                # exact input match
                for i, e in enumerate(bucket):
                    if e.input_sig == insig:
                        e.usage_count += 1
                        self.stats["hits"] += 1
                        self.stats["verified_hits"] += 1
                        return self.payload_resolver(e.payload_ptr), {
                            "reason": "exact_input_sig",
                            "semhash": semhash
                        }

                # semantic similarity inside bucket
                sims = [self._cosine_similarity(emb, e.embedding) for e in bucket]
                best_idx = int(np.argmax(sims))
                best_sim = sims[best_idx]

                if best_sim >= self.sim_threshold:
                    self.stats["hits"] += 1
                    self.stats["verified_hits"] += 1
                    return self.payload_resolver(bucket[best_idx].payload_ptr), {
                        "reason": "similarity_accept",
                        "similarity": best_sim,
                        "semhash": semhash
                    }

            # ---------- 2. GLOBAL FALLBACK (PHASE-1 CORE) ----------
            best_sim = -1.0
            best_entry = None
            best_semhash = None

            for s_hash, entries in self.buckets.items():
                for e in entries:
                    sim = self._cosine_similarity(emb, e.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = e
                        best_semhash = s_hash

            if best_entry and best_sim >= self.sim_threshold:
                self.stats["hits"] += 1
                self.stats["verified_hits"] += 1
                return self.payload_resolver(best_entry.payload_ptr), {
                    "reason": "fallback_linear_accept",
                    "similarity": best_sim,
                    "semhash": best_semhash
                }

            self.stats["misses"] += 1
            return None, {
                "reason": "miss_no_bucket",
                "best_similarity": best_sim
            }

    def dump_bucket(self, semhash: str):
        with self.lock:
            return [ {"input_sig":e.input_sig,"output_fp":e.output_fp,"payload_ptr":e.payload_ptr,"usage_count":e.usage_count} for e in self.buckets.get(semhash,[]) ]

    def stats_snapshot(self): return dict(self.stats)
