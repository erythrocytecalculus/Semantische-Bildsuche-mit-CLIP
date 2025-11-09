from PIL import Image
import os
import shutil

def open_image(filepath):
    """Open an image with the default viewer."""
    img = Image.open(filepath)
    img.show()

def copy_images_to_directory(image_paths, target_folder):
    """Copy multiple image files to a specified directory."""
    os.makedirs(target_folder, exist_ok=True)
    for path in image_paths:
        shutil.copy(path, target_folder)
