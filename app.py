from flask import Flask, render_template, request
import faiss
import numpy as np
import torch
import clip
import os

# Load the FAISS index and CLIP model
index = faiss.read_index("image_index.faiss")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def search_images():
    results = []
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            # Process query (text) using CLIP model
            text = clip.tokenize([query]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
            text_features = text_features.cpu().numpy()[0]
            text_features /= np.linalg.norm(text_features)  # Normalize the query

            # Search in FAISS index
            D, I = index.search(np.expand_dims(text_features, axis=0), k=10)  # top 10 results

            # Get corresponding image file paths (you need to maintain a list of paths)
            image_folder = r"C:\Users\TUFA17\[target_dir\validation]"  # The folder where you stored images
            image_paths = [os.path.join(image_folder, f"{i}.jpg") for i in I[0]]
            results = image_paths

    return render_template('search.html', images=results)

if __name__ == '__main__':
    app.run(debug=True)