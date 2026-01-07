# ---------------------------------------------
# Matplotlib (server-safe)
# ---------------------------------------------
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------
# Imports
# ---------------------------------------------
from flask import Flask, render_template, request
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import base64
import sys

# ---------------------------------------------
# Flask App
# ---------------------------------------------
app = Flask(__name__)

# ---------------------------------------------
# Base directory
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------
# Model path (REDUCED SINGLE-FILE .vec)
# ---------------------------------------------
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "kannada_word2vec_100k.vec"   # <-- make sure this file exists
)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found at: {MODEL_PATH}")
    sys.exit(1)

# ---------------------------------------------
# Load Word2Vec vectors (.vec format)
# ---------------------------------------------
try:
    model = KeyedVectors.load_word2vec_format(
        MODEL_PATH,
        binary=True
    )
except Exception as e:
    print("❌ Failed to load .vec model")
    raise e

print("✅ Word2Vec (.vec) model loaded successfully")
print("📘 Vocabulary size:", len(model))

# ---------------------------------------------
# Load Kannada font (optional)
# ---------------------------------------------
FONT_PATH = os.path.join(BASE_DIR, "Font", "Nirmala.ttf")
kannada_font = None

if os.path.exists(FONT_PATH):
    kannada_font = fm.FontProperties(fname=FONT_PATH)
    print("✅ Kannada font loaded")
else:
    print("⚠ Kannada font not found, using default font")

# ---------------------------------------------
# Routes
# ---------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image = None

    if request.method == "POST":
        word = request.form.get("word", "").strip()
        topn = int(request.form.get("topn", 5))

        if word in model:
            similar_words = model.most_similar(word, topn=topn)

            # Prepare PCA data
            words = [word] + [w for w, _ in similar_words]
            vectors = [model[w] for w in words]

            pca = PCA(n_components=2)
            reduced = pca.fit_transform(vectors)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))

            for i, w in enumerate(words):
                color = "red" if i == 0 else "blue"
                ax.scatter(reduced[i, 0], reduced[i, 1], color=color)
                ax.annotate(
                    w,
                    (reduced[i, 0] + 0.01, reduced[i, 1] + 0.01),
                    fontproperties=kannada_font,
                    fontsize=12,
                    color=color
                )

            ax.set_title(
                f"Top {topn} Similar Words to '{word}'",
                fontproperties=kannada_font
            )
            ax.grid(True)

            # Convert plot to image
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format="png")
            buffer.seek(0)
            image = base64.b64encode(buffer.read()).decode("utf-8")
            buffer.close()
            plt.close()

            result = [(w, f"{s:.4f}") for w, s in similar_words]
        else:
            result = "Word not found in vocabulary."

    return render_template(
        "kannada_index.html",
        result=result,
        image=image
    )

# ---------------------------------------------
# Local run (Render uses gunicorn)
# ---------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
