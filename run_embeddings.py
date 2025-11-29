import os, time
from dotenv import load_dotenv

# load environment variables
load_dotenv()

print("ENV:", os.getenv("ENV", "not-set"))
MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
emb_dim = int(os.getenv("CACHE_EMB_DIM", "384"))
sim_threshold = float(os.getenv("CACHE_SIM_THRESHOLD", "0.92"))

print(f"Model to load: {MODEL_NAME}")
print(f"Embedding dim: {emb_dim}, similarity threshold: {sim_threshold}")

# import cache system
from collision_cache.core import CollisionAwareSemanticCache
from collision_cache.embedding_providers import sentence_transformers_provider

# load model
print("Loading SentenceTransformer model, please wait...")
t0 = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded in {time.time() - t0:.1f} seconds")

emb_fn = sentence_transformers_provider(model)

# create cache
cache = CollisionAwareSemanticCache(
    emb_dim=emb_dim,
    hasher_bits=int(os.getenv("CACHE_HASH_BITS", "128")),
    sim_threshold=sim_threshold
)

# store entry
prompt_store = "What is the capital of France?"
output_store = "Paris"

print("Computing embedding for the store prompt...")
emb_store = emb_fn(prompt_store)
cache.store(prompt_store, output_store, emb_store, payload_ptr="inline://paris")

print("Store complete. Cache stats after store:", cache.stats_snapshot())

# lookup paraphrase
query = "Tell me the capital city of France."
print("\nComputing embedding for query...")
emb_q = emb_fn(query)

print("Looking up in cache...")
res, info = cache.lookup(query, emb_fn)

print("\nRESULT:", res)
print("INFO:", info)
print("\nFINAL STATS:", cache.stats_snapshot())

print("\nDone.")
