# preload_and_test.py
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer
from collision_cache.core import CollisionAwareSemanticCache
from collision_cache.embedding_providers import sentence_transformers_provider


def load_preload_json(path="preload.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("Loading .env and config...")
    MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    EMB_DIM = int(os.getenv("CACHE_EMB_DIM", "384"))
    SIM_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.75"))

    print(f"Model: {MODEL_NAME} | emb_dim={EMB_DIM} | sim_threshold={SIM_THRESHOLD}")

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    emb_fn = sentence_transformers_provider(model)

    # ---- CACHE (Phase-1 logic + Redis persistence) ----
    cache = CollisionAwareSemanticCache(
        emb_dim=EMB_DIM,
        sim_threshold=SIM_THRESHOLD,
        backend="upstash"
    )

    # ---- PRELOAD ----
    preload = load_preload_json("preload.json")
    print(f"Preloading {len(preload)} entries into cache...")

    for i, (prompt, output) in enumerate(preload.items(), start=1):
        emb = emb_fn(prompt)
        cache.store(prompt, output, emb, payload_ptr=f"inline://preload_{i}")

    print("Preload complete. Cache stats:", cache.stats_snapshot())

    # ---- TEST QUERIES ----
    test_queries = [
        # Expected hits
        "Who is credited with discovering penicillin?",
        "Explain in one line what Python programming is.",
        "Which animal is the national animal of India?",
        "What's 2 plus 2?",
        "What's the chemical formula for water?",
        "How can I reverse a list in Python?",

        # Expected misses
        "How to replace a broken screen on a phone?",
        "Give me a recipe for banana bread.",

        # Borderline
        "What is the boiling temperature of H2O at sea level?",
        "Which mountain is the highest above sea level?"
    ]

    print("\nRunning tests on", len(test_queries), "queries...\n")

    for q in test_queries:
        result, info = cache.lookup(q, emb_fn)
        similarity = info.get("similarity")

        print("Query:", q)
        print("Lookup result:", result)
        print("Reason:", info.get("reason"))
        if similarity is not None:
            print("Similarity:", round(similarity, 4))
        print("-" * 60)

    # ---- INTERACTIVE MODE ----
    print("\nEntering interactive cache test mode.")
    print("Type a query to test the cache.")
    print("Type 'exit' to quit.\n")

    while True:
        user_q = input(">> ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("Exiting interactive mode.")
            break

        result, info = cache.lookup(user_q, emb_fn)

        if result is not None:
            print("CACHE HIT")
            print("Result:", result)

        else:
            print("CACHE MISS")
            print("Reason:", info.get("reason"))

            # ---- STORE QUERY ONLY (NO ANSWER) ----
            emb_new = emb_fn(user_q)

            cache.store(
                prompt=user_q,
                output_text="", 
                embedding=emb_new,
                payload_ptr="inline://query_only"
            )

            print("Query stored in cache")

        print("-" * 60)


if __name__ == "__main__":
    main()
