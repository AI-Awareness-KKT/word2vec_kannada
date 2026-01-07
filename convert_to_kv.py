from gensim.models import Word2Vec
import os

# Load the FULL original Word2Vec model
w2v = Word2Vec.load(
    os.path.join(
        "original_model",   # keep your folder name as-is
        "kannada_word2vec.model"
    )
)

# Save as TRUE single-file (no .npy will be created)
w2v.wv.save_word2vec_format(
    os.path.join("model", "kannada_word2vec.vec"),
    binary=True
)

print("✅ Single-file Word2Vec vector file created (no .npy)")
