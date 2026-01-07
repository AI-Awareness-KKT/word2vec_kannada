from gensim.models import KeyedVectors
import os

INPUT_VEC = "model/kannada_word2vec.vec"
OUTPUT_VEC = "model/kannada_word2vec_100k.vec"
MAX_WORDS = 100_000  # change to 50_000 if still large

print("Loading original vec model...")
kv = KeyedVectors.load_word2vec_format(INPUT_VEC, binary=True)

print(f"Original vocab size: {len(kv)}")

# Get top-N words (by frequency order)
top_words = kv.index_to_key[:MAX_WORDS]

# Create a new KeyedVectors instance
new_kv = KeyedVectors(vector_size=kv.vector_size)

# Add vectors
new_kv.add_vectors(
    top_words,
    [kv[word] for word in top_words]
)

print(f"Reduced vocab size: {len(new_kv)}")

# Save reduced model (single-file)
new_kv.save_word2vec_format(OUTPUT_VEC, binary=True)

print("✅ Reduced vec model saved successfully")
