import numpy as np
import umap
import os
import plotly.express as px
from PIL import Image
import base64
from io import BytesIO

# Load embeddings
embeddings = np.load("image_embeddings.npy")

# UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)

# Image folder
image_folder = r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\OpenImages\resize"
filenames = os.listdir(image_folder)

# Convert images to base64 for hover
def img_to_base64(path, size=(128, 128)):
    img = Image.open(path).convert("RGB").resize(size)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

hover_images = [
    img_to_base64(os.path.join(image_folder, f))
    for f in filenames[:len(umap_embeddings)]
]

fig = px.scatter(
    x=umap_embeddings[:, 0],
    y=umap_embeddings[:, 1],
    title="Interactive UMAP of CLIP Image Embeddings",
)

fig.update_traces(
    marker=dict(size=6, opacity=0.7),
    hovertemplate='<img src="data:image/png;base64,%{customdata}" width="128"><extra></extra>',
    customdata=hover_images
)

fig.show()