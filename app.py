from flask import Flask, render_template, request, send_from_directory
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

# Folder where resized images are stored (processed)
image_folder = r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\OpenImages\resize"

# Dynamically create the mapping from FAISS index to filenames
image_filenames = os.listdir(image_folder)  # Get all filenames in the 'resize' folder
index_to_filename = {i: image_filenames[i] for i in range(len(image_filenames))}  # Map indices to filenames

# Serve static files (images) from the 'resize' folder
@app.route('/images/<path:filename>')
def serve_image(filename):
    print(f"Serving image: {filename}")  # Debug: Print image being served
    return send_from_directory(image_folder, filename)

@app.route('/', methods=['GET', 'POST'])
def search_images():
    results = []
    query = ""  # Initialize the query variable to an empty string
    
    if request.method == 'POST':
        query = request.form.get('query')  # Get the query from the form
        
        if query:
            # Process query (text) using CLIP model
            text = clip.tokenize([query]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
            text_features = text_features.cpu().numpy()[0]
            text_features /= np.linalg.norm(text_features)  # Normalize the query

            # Search in FAISS index
            D, I = index.search(np.expand_dims(text_features, axis=0), k=10)  # top 10 results

            # DEBUG: Log the indices returned by FAISS
            print(f"FAISS returned indices: {I[0]}")

            # Get corresponding image file paths using the mapping
            try:
                image_paths = [f"/images/{index_to_filename[i]}" for i in I[0]]  # Map FAISS index to image filename
                print(f"Generated image paths: {image_paths}")  # Debug: Log generated image paths
            except KeyError as e:
                print(f"KeyError: {e} not found in index_to_filename mapping")
                return "Error: Image not found."

            results = image_paths

    return render_template('search.html', images=results, query=query)


if __name__ == '__main__':
    app.run(debug=True)
