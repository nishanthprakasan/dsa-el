# __init__.py
"""
Collision-Aware Semantic Cache package.
Provides:
- core: main cache implementation
- utils: canonicalization and hashing helpers
- embedding_providers: embedding adapters
- cli: demo runner
"""

from .core import CollisionAwareSemanticCache
from .embedding_providers import simple_hash_embedding, sentence_transformers_provider
