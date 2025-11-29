# cli.py
import argparse
from .core import CollisionAwareSemanticCache
from .embedding_providers import simple_hash_embedding
from .s3_blobstore import S3BlobStore

def main_demo():
    cache = CollisionAwareSemanticCache(emb_dim=384, hasher_bits=128, seed=123, sim_threshold=0.92)
    # example: store and lookup
    p = "What's the capital of France?"
    emb = simple_hash_embedding(p, dim=384)
    # for demo we store payload inline (in production store pointer via s3_blobstore)
    cache.store(p, "Paris", emb, payload_ptr="inline://paris", model_version="demo-v1")
    payload, info = cache.lookup("Name the capital city of France.", simple_hash_embedding)
    print("lookup:", payload, info)

if __name__ == "__main__":
    main_demo()
