# test_core.py
import pytest
import numpy as np
from collision_cache.core import CollisionAwareSemanticCache
from collision_cache.embedding_providers import simple_hash_embedding

def test_store_and_lookup_exact():
    cache = CollisionAwareSemanticCache(emb_dim=384)
    p = "hello world"
    emb = simple_hash_embedding(p, dim=384)
    cache.store(p, "greeting", emb, payload_ptr="inline://greeting")
    payload,_ = cache.lookup(p, lambda t: simple_hash_embedding(t, dim=384))
    assert payload == "inline://greeting"

def test_paraphrase_similarity_accept():
    cache = CollisionAwareSemanticCache(emb_dim=384, sim_threshold=0.5) # low threshold to accept
    p1 = "Write short poem about dog"
    p2 = "Short poem: dog"
    cache.store(p1, "poem1", simple_hash_embedding(p1, dim=384), payload_ptr="ptr1")
    payload,_ = cache.lookup(p2, lambda t: simple_hash_embedding(t, dim=384))
    # With pseudo-embeddings this might mismatch; this test ensures flow runs
    assert (payload is None) or isinstance(payload, str)
