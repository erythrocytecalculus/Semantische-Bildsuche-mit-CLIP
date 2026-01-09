import numpy as np
import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

# Load embeddings
embeddings = np.load("image_embeddings.npy")

# Run UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)

# Image folder
image_folder = r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\OpenImages\resize"
filenames = os.listdir(image_folder)

# Create plot
fig, ax = plt.subplots(figsize=(14, 12))
ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=2, alpha=0.3)

# Function to add thumbnails
def add_thumbnail(ax, img_path, x, y, zoom=0.25):
    img = Image.open(img_path).convert("RGB")
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

# Add thumbnails (limit for performance)
max_images = 300

for i in range(min(max_images, len(filenames))):
    img_path = os.path.join(image_folder, filenames[i])
    add_thumbnail(ax, img_path, umap_embeddings[i, 0], umap_embeddings[i, 1])

ax.set_title("UMAP with Image Thumbnails")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

plt.show()