# embedding_providers.py
import numpy as np
from typing import Callable

# Example: simple deterministic pseudo-embedding (useful for unit tests)
import hashlib
def simple_hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    t = text.strip().lower().encode("utf-8")
    h = hashlib.sha256(t).digest()
    seed = int.from_bytes(h[:8], "big") % (2**31)
    rng = np.random.RandomState(seed)
    v = rng.normal(size=(dim,)).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

# Example: sentence-transformers adapter
def sentence_transformers_provider(model):
    # model: a SentenceTransformer instance
    def emb_fn(text: str) -> np.ndarray:
        vec = model.encode(text, normalize_embeddings=False)
        return np.asarray(vec, dtype=np.float32)
    return emb_fn
