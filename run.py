import os
import torch
import clip
import numpy as np
from PIL import Image
import faiss

# Function to preprocess images (resize them)
def preprocess_images(image_folder, output_folder, size=(256, 256)):
    """Preprocess images: resize and save to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            img = img.resize(size)  # Resize the image
            img.save(os.path.join(output_folder, filename))

    print(f"Processed images saved to {output_folder}")

# Function to extract CLIP embeddings for each image
def extract_embeddings(image_folder, output_file="image_embeddings.npy"):
    """Extract CLIP embeddings for each image and save them to a file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    embeddings = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(img)

            feature = feature.cpu().numpy()[0]
            feature = feature / np.linalg.norm(feature)  # Normalize the embedding
            embeddings.append(feature)

    embeddings = np.array(embeddings)
    np.save(output_file, embeddings)  # Save embeddings to a file
    print(f"Embeddings saved to {output_file}")

# Function to build FAISS index
def build_faiss_index(embeddings_file="image_embeddings.npy", index_file="image_index.faiss"):
    """Build a FAISS index for the image embeddings and save it."""
    embeddings = np.load(embeddings_file)

    # Initialize FAISS index (Inner Product: cosine similarity)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)  # Normalize the embeddings
    index.add(embeddings)  # Add embeddings to the index

    faiss.write_index(index, index_file)  # Save the FAISS index
    print(f"FAISS index saved to {index_file}")

if __name__ == "__main__":
    # Set your folders here
    image_folder = r"C:\Users\TUFA17\[target_dir\validation]"  # e.g., "C:/OpenImages/processed"
    output_folder = r"C:\Users\TUFA17\[target_dir\resized]"  # e.g., "C:/OpenImages/resized"

    # Preprocess images (resize them)
    preprocess_images(image_folder, output_folder)

    # Extract embeddings for the resized images
    extract_embeddings(output_folder, output_file="image_embeddings.npy")

    # Build FAISS index from the embeddings
    build_faiss_index(embeddings_file="image_embeddings.npy", index_file="image_index.faiss")
