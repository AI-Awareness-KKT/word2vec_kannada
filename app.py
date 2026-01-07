import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import base64

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Base directory
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------
# Load Word2Vec model (ONLY .model file)
# -------------------------------------------------
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "kannada_new_word2vec",
    "kannada_word2vec.model"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = Word2Vec.load(MODEL_PATH)
print("✅ Word2Vec model loaded")
print("📘 Vocabulary size:", len(model.wv))

# -------------------------------------------------
# Load Kannada font (optional but recommended)
# -------------------------------------------------
FONT_PATH = os.path.join(BASE_DIR, "Font", "Nirmala.ttf")
kannada_font = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

if kannada_font:
    print("✅ Kannada font loaded")
else:
    print("⚠ Kannada font not found, text may not render correctly")

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image = None

    if request.method == "POST":
        word = request.form.get("word", "").strip()
        topn = int(request.form.get("topn", 5))

        if word in model.wv:
            # Get similar words
            similar_words = model.wv.most_similar(word, topn=topn)

            # Prepare PCA data
            words = [word] + [w for w, _ in similar_words]
            vectors = [model.wv[w] for w in words]

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

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5001)
