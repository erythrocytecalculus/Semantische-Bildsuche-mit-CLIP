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

@app.route('/umap')
def umap_view():
    import umap
    import plotly.express as px
    import numpy as np
    from PIL import Image
    import base64
    from io import BytesIO
    import os

    try:
        # Load the image embeddings
        embeddings = np.load("image_embeddings.npy")

        # Perform UMAP projection
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
        umap_embeddings = reducer.fit_transform(embeddings)

        # Image folder path
        image_folder = "path/to/your/images"  # Update with the correct path to your image folder
        filenames = os.listdir(image_folder)

        # Convert images to base64 for hover functionality
        def img_to_base64(path, size=(128, 128)):
            try:
                img = Image.open(path).convert("RGB").resize(size)
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode()
            except Exception as e:
                print(f"Error converting image {path} to base64: {e}")
                return None

        # Create the hover data (base64 images)
        hover_images = []
        for f in filenames[:len(umap_embeddings)]:
            img_path = os.path.join(image_folder, f)
            base64_image = img_to_base64(img_path)
            if base64_image:  # Only add valid base64 images
                hover_images.append(base64_image)
            else:
                hover_images.append('')  # Add empty string if image conversion failed

        # Create UMAP plot
        fig = px.scatter(
            x=umap_embeddings[:, 0],
            y=umap_embeddings[:, 1],
            title="UMAP Visualization of CLIP Image Embeddings"
        )

        # Add hover functionality (display images on hover)
        fig.update_traces(
            marker=dict(size=6, opacity=0.7),
            hovertemplate='<img src="data:image/png;base64,%{customdata}" width="128"><extra></extra>',
            customdata=hover_images
        )

        # Convert the figure to HTML
        graph_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        # Ensure the rendered HTML template properly receives the 'graph' variable
        return render_template("umap.html", graph=graph_html)

    except Exception as e:
        # Handle any errors and print them to Flask console for debugging
        print(f"Error generating UMAP plot: {e}")
        return f"Error generating UMAP plot: {e}"

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
