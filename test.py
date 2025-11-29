import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from collision_cache.core import CollisionAwareSemanticCache
from collision_cache.embedding_providers import sentence_transformers_provider


def main():
    print("Loading config...")
    load_dotenv()

    MODEL_NAME = os.getenv("EMBED_MODEL_NAME")
    THRESH = float(os.getenv("CACHE_SIM_THRESHOLD"))

    print(f"ENV threshold = {THRESH}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    emb_fn = sentence_transformers_provider(model)

    cache = CollisionAwareSemanticCache(
        emb_dim=int(os.getenv("CACHE_EMB_DIM")),
        hasher_bits=int(os.getenv("CACHE_HASH_BITS")),
        sim_threshold=THRESH
    )

    # -----------------------
    # STORE ENTRY
    # -----------------------
    store_prompt = "What type of object is the Sun?"
    store_output = "The Sun is a star."
    emb_store = emb_fn(store_prompt)

    cache.store(
        store_prompt,
        store_output,
        emb_store,
        payload_ptr="inline://sun_is_star"
    )

    print("\nStored Entry:")
    print("Prompt:", store_prompt)
    print("Output:", store_output)
    print("-" * 60)

    # -----------------------
    # PARAPHRASES
    # -----------------------
    paraphrases = [
        "What kind of celestial body is the Sun?",
        "Is the Sun a planet or a star?",
        "What category of object does the Sun belong to?",
        "What is the Sun?",
        "Describe the Sun.",
        "Is the Sun considered a star?",
        "What object is the Sun classified as?",
        # borderline cases
        "What is the size of the Sun?",
        "How far is the Sun?",
        "Explain nuclear fusion in the Sun."
    ]

    # -----------------------
    # TESTING LOOP
    # -----------------------
    print("\nRunning test cases...\n")

    for q in paraphrases:
        emb_q = emb_fn(q)
        sim = cache._cosine_similarity(emb_q, emb_store)
        result, info = cache.lookup(q, emb_fn)

        print(f"Query: {q}")
        print(f"Similarity vs stored: {sim:.4f}")
        print(f"Lookup result: {result}")
        print(f"Reason: {info['reason']}")
        print("-" * 60)

    print("\nFinal Stats:", cache.stats_snapshot())
    print("Done.")


if __name__ == "__main__":
    main()
