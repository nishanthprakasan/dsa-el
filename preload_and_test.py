# preload_and_test.py
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# imports for embeddings & cache
from sentence_transformers import SentenceTransformer
from collision_cache.core import CollisionAwareSemanticCache
from collision_cache.embedding_providers import sentence_transformers_provider

def load_preload_json(path="preload.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Create a JSON with prompt->output pairs.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def main():
    print("Loading .env and config...")
    MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    EMB_DIM = int(os.getenv("CACHE_EMB_DIM", "384"))
    HASH_BITS = int(os.getenv("CACHE_HASH_BITS", "128"))
    SIM_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.92"))

    print(f"Model: {MODEL_NAME} | emb_dim={EMB_DIM} | sim_threshold={SIM_THRESHOLD}")

    # load model (may download on first run)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - t0:.1f}s")
    emb_fn = sentence_transformers_provider(model)

    # create cache
    cache = CollisionAwareSemanticCache(
        emb_dim=EMB_DIM,
        hasher_bits=HASH_BITS,
        sim_threshold=SIM_THRESHOLD
    )

    # load preload JSON
    try:
        preload = load_preload_json("preload.json")
    except Exception as e:
        print("Error loading preload.json:", e)
        return

    print(f"Preloading {len(preload)} entries into cache...")
    for i, (prompt, output) in enumerate(preload.items(), start=1):
        emb = emb_fn(prompt)
        cache.store(prompt, output, emb, payload_ptr=f"inline://preload_{i}")
        if i % 50 == 0:
            print(f"  stored {i} entries...")

    print("Preload complete. Cache stats:", cache.stats_snapshot())

    # --- TEST QUERIES---
    test_queries = [
    # --- Expected hits (paraphrases of preload) ---
    "Name the capital city of Italy.",
    "Who is credited with discovering penicillin?",
    "Explain in one line what Python programming is.",
    "Which animal is the national animal of India?",
    "What's 2 plus 2?",
    "Who wrote Pride & Prejudice?",
    "What's the chemical formula for water?",
    "What does 'photosynthesis' mean?",
    "How can I reverse a list in Python?",
    
    # --- Expected misses (different intent / unrelated) ---
    "How to replace a broken screen on a phone?",
    "What are the lyrics to a popular song?",
    "Give me a recipe for banana bread.",
    
    # --- borderline / check thresholds (may be hit if threshold low) ---
    "What is the boiling temperature of H2O at sea level?",
    "Which mountain is the highest above sea level?"
]


    print("\nRunning tests on", len(test_queries), "queries...\n")
    for q in test_queries:
        emb_q = emb_fn(q)
        sim_to_candidates = None
        # compute similarity to best stored item using fallback logic
        # (we will call lookup which uses the cache fallback)
        res, info = cache.lookup(q, emb_fn)

        # For debugging: compute similarity to the top fallback candidate if available
        if "similarity" in info:
            sim_to_candidates = info["similarity"]
        else:
            # compute similarity vs each stored embedding (slow but useful for debug)
            best_sim = -1.0
            for s_hash, entries in cache.buckets.items():
                for e in entries:
                    sim = cache._cosine_similarity(emb_q, e.embedding)
                    if sim > best_sim:
                        best_sim = sim
            sim_to_candidates = best_sim

        print("Query:", q)
        print("Similarity (best):", f"{sim_to_candidates:.4f}" if sim_to_candidates is not None else "N/A")
        print("Lookup result:", res)
        print("Reason:", info.get("reason"))
        print("-" * 60)

    print("\nFinal cache stats:", cache.stats_snapshot())
    print("Done.")

if __name__ == "__main__":
    main()
    