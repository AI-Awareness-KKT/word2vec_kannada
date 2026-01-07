from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Load Word2Vec model
model_path = "/u_Old_models/kannada_word2vec/kannada_word2vec.model"
model = Word2Vec.load(model_path)

# Try loading Kannada font
font_path = "Font/Nirmala.ttf"  # Or use your Noto Sans Kannada path
if os.path.exists(font_path):
    kannada_font = fm.FontProperties(fname=font_path)
else:
    print("Kannada font not found. Labels may not render properly.")
    kannada_font = None

# Get inputs
word = input("Enter a Kannada word: ")
try:
    topn = int(input("Enter number of similar words to display: "))
except ValueError:
    topn = 5

# Generate plot
if word in model.wv:

    print(f"\nTop {topn} similar words to '{word}':")
    for sim_word, score in model.wv.most_similar(word, topn=topn):
        print(f"{sim_word}: {score:.4f}")

    similar_words = model.wv.most_similar(word, topn=topn)
    words = [word] + [w for w, _ in similar_words]
    vectors = [model.wv[w] for w in words]

    # Reduce dimensions
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)

    # Plotting
    plt.figure(figsize=(10, 8))
    for i, w in enumerate(words):
        color = 'red' if i == 0 else 'blue'
        plt.scatter(result[i, 0], result[i, 1], color=color)
        plt.annotate(w, xy=(result[i, 0] + 0.01, result[i, 1] + 0.01),
                     fontproperties=kannada_font,
                     fontsize=12, color=color)

    plt.title(f"2D PCA of Top {topn} Similar Words to '{word}'", fontsize=16, fontproperties=kannada_font)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("Word not found in vocabulary.")
