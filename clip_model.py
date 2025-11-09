import clip
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-L/14@336px", DEVICE)

def encode_image(image_path):
    """Convert an image to a vector representation using CLIP."""
    image_tensor = PREPROCESS(Image.open(image_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vector = MODEL.encode_image(image_tensor)
    return vector.cpu().numpy().flatten()

def encode_text(text):
    """Generate a vector from text using CLIP."""
    with torch.no_grad():
        vector = MODEL.encode_text(clip.tokenize([text]).to(DEVICE))
    return vector.cpu().numpy().flatten()