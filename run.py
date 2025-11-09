import os
from clip_model import encode_image
from db_manager import (
    get_annoy_index, save_annoy_index,
    create_sqlite_db, insert_image_path
)
import settings

def build_database(image_dir):
    create_sqlite_db(settings.SQLITE_PATH)
    annoy_index = get_annoy_index(settings.VECTOR_SIZE)

    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for idx, image_name in enumerate(images):
        full_path = os.path.join(image_dir, image_name)
        vector = encode_image(full_path)
        annoy_index.add_item(idx, vector)
        insert_image_path(settings.SQLITE_PATH, full_path)

    save_annoy_index(annoy_index, settings.ANNOY_PATH, settings.ANNOY_TREE_COUNT)

if __name__ == '__main__':
    # Update this with your local images folder
    build_database(r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\static\images")