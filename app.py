import sqlite3  # Add this import
from flask import Flask, render_template, request
from annoy import AnnoyIndex
import numpy as np
import torch
import clip
from settings import ANNOY_PATH, VECTOR_SIZE, SQLITE_PATH
from db_manager import fetch_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

# Load Annoy index and filenames
annoy_index = AnnoyIndex(VECTOR_SIZE, 'angular')
annoy_index.load(ANNOY_PATH)
with open("image_index.txt", "r") as f:
    filenames = [line.strip() for line in f]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def search_images():
    results = []
    if request.method == 'POST':
        query = request.form.get('query')  # This matches the input field in the HTML
        if query:  # Ensure query is not empty
            # Process the query with CLIP model to get its embedding
            text = clip.tokenize([query]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
            text_features = text_features.cpu().numpy()
            text_features = text_features / np.linalg.norm(text_features)  # Normalize text features

            # Fetch all embeddings from SQLite for search
            conn = sqlite3.connect(SQLITE_PATH)
            c = conn.cursor()
            c.execute('SELECT id, filepath, embedding FROM images')
            images = c.fetchall()
            conn.close()

            # Calculate similarity and rank the results
            similarities = []
            for image_id, filepath, embedding in images:
                image_vector = np.frombuffer(embedding, dtype=np.float32)
                similarity = np.dot(text_features[0], image_vector)  # Cosine similarity
                similarities.append((filepath, similarity))

            # Sort results by similarity and fetch the top 10
            results = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
        else:
            # Handle the case when 'query' is missing or empty
            results = ["Please enter a search query."]

    return render_template('search.html', images=[r[0] for r in results])

if __name__ == '__main__':
    app.run(debug=True)
