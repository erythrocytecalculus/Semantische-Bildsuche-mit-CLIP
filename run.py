import os
import numpy as np
from annoy import AnnoyIndex
from PIL import Image
import torch
import clip
from settings import ANNOY_PATH, SQLITE_PATH, VECTOR_SIZE, ANNOY_TREE_COUNT
from db_manager import insert_image_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

image_folder = "static/images"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def normalize_vector(vector):
    """Normalize a vector to unit length."""
    return vector / np.linalg.norm(vector, axis=1, keepdims=True)

features_list = []
for imgfile in image_files:
    image_path = os.path.join(image_folder, imgfile)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features = image_features.cpu().numpy()
    image_features = normalize_vector(image_features)  # Normalize features
    features_list.append((imgfile, image_features[0]))

# Insert image embeddings into the SQLite DB
for imgfile, features in features_list:
    insert_image_embedding(SQLITE_PATH, os.path.join(image_folder, imgfile), features)

# Build Annoy index
annoy_index = AnnoyIndex(VECTOR_SIZE, 'angular')
for idx, (imgfile, features) in enumerate(features_list):
    annoy_index.add_item(idx, features)
annoy_index.build(ANNOY_TREE_COUNT)
annoy_index.save(ANNOY_PATH)

# Optionally, save filenames (index mapping) for later retrieval
with open("image_index.txt", "w") as f:
    for imgfile, _ in features_list:
        f.write(f"{imgfile}\n")
