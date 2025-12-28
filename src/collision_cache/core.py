import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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
                 payload_resolver: Callable[[str], str] = lambda ptr: ptr):
        self.hasher = SemanticHasher(emb_dim, n_bits=hasher_bits, seed=seed)
        self.buckets: Dict[str, List[CacheEntry]] = {}
        self.lock = threading.Lock()
        self.sim_threshold = sim_threshold
        self.max_bucket_size = max_bucket_size
        self.payload_resolver = payload_resolver
        self.stats = {"lookups":0,"hits":0,"verified_hits":0,"misses":0,"collisions_detected":0,"stores":0}

    def store(self, prompt: str, output_text: str, embedding: np.ndarray,
              payload_ptr: str, model_version: str = "1.0", confidence: Optional[float] = None):
        semhash = self.hasher.hash_from_embedding(embedding)
        insig = input_signature(prompt)
        outfp = output_fingerprint(output_text)
        entry = CacheEntry(input_sig=insig, output_fp=outfp, embedding=embedding.astype(np.float32),
                           payload_ptr=payload_ptr, model_version=model_version,
                           confidence=confidence)
        with self.lock:
            self.buckets.setdefault(semhash, []).append(entry)
            self.stats["stores"] += 1

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0: return -1.0
        return float(np.dot(a,b)/(na*nb))

    def lookup(self, prompt: str, embedding_fn: Callable[[str], np.ndarray]) -> Tuple[Optional[str], dict]:
        """Lookup a prompt; fast-path bucket then linear fallback across entries."""
        self.stats["lookups"] += 1
        _ = canonicalize_input(prompt)
        emb = embedding_fn(prompt)
        semhash = self.hasher.hash_from_embedding(emb)

        with self.lock:
            bucket = self.buckets.get(semhash)
            if bucket:
                insig = input_signature(prompt)
                for i, e in enumerate(bucket):
                    if e.input_sig == insig:
                        e.usage_count += 1
                        self.stats["hits"] += 1
                        self.stats["verified_hits"] += 1
                        return self.payload_resolver(e.payload_ptr), {"reason": "exact_input_sig", "semhash": semhash, "entry_index": i}
                sims = [self._cosine_similarity(emb, e.embedding) for e in bucket]
                best_idx = int(np.argmax(sims))
                best_sim = sims[best_idx]
                best_entry = bucket[best_idx]
                if best_sim >= self.sim_threshold:
                    best_entry.usage_count += 1
                    self.stats["hits"] += 1
                    self.stats["verified_hits"] += 1
                    return self.payload_resolver(best_entry.payload_ptr), {"reason": "similarity_accept", "semhash": semhash, "similarity": best_sim, "entry_index": best_idx}
                else:
                    self.stats["collisions_detected"] += 1
                    self.stats["misses"] += 1
                    return None, {"reason": "collision_suspected", "semhash": semhash, "best_similarity": best_sim, "entry_index": best_idx, "bucket_size": len(bucket)}
            else:
                best_sim = -1.0
                best_entry = None
                best_sem = None
                for s_hash, entries in self.buckets.items():
                    for e in entries:
                        sim = self._cosine_similarity(emb, e.embedding)
                        if sim > best_sim:
                            best_sim = sim
                            best_entry = e
                            best_sem = s_hash
                if best_entry is not None and best_sim >= self.sim_threshold:
                    best_entry.usage_count += 1
                    self.stats["hits"] += 1
                    self.stats["verified_hits"] += 1
                    return self.payload_resolver(best_entry.payload_ptr), {"reason": "fallback_linear_accept", "semhash": best_sem, "similarity": best_sim}
                self.stats["misses"] += 1
                return None, {"reason": "miss_no_bucket_and_no_fallback", "semhash": semhash, "best_sim": best_sim}

    def dump_bucket(self, semhash: str):
        with self.lock:
            return [ {"input_sig":e.input_sig,"output_fp":e.output_fp,"payload_ptr":e.payload_ptr,"usage_count":e.usage_count} for e in self.buckets.get(semhash,[]) ]

    def stats_snapshot(self): return dict(self.stats)
