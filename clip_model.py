import clip
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load('ViT-L/14@336px', DEVICE)

def encode_image(image_path):
    """Convert an image file into a vector representation using the CLIP model."""
    img_tensor = PREPROCESS(Image.open(image_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vector = MODEL.encode_image(img_tensor)
    return vector.cpu().numpy().flatten()

def encode_text(text):
    """Convert input text into its embedding vector."""
    tokens = clip.tokenize([text]).to(DEVICE)
    with torch.no_grad():
        vector = MODEL.encode_text(tokens)
    return vector.cpu().numpy().flatten()
