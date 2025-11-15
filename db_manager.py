import sqlite3
import numpy as np

def create_sqlite_db(path):
    """Create SQLite database to store image paths and embeddings."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT NOT NULL,
                    embedding BLOB NOT NULL)''')
    conn.commit()
    conn.close()

def insert_image_embedding(db_path, img_path, embedding):
    """Insert image path and embedding into the SQLite database."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO images (filepath, embedding) VALUES (?, ?)', 
              (img_path, embedding.tobytes()))
    conn.commit()
    conn.close()

def fetch_embedding(db_path, image_id):
    """Retrieve image embedding by ID."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT embedding FROM images WHERE id = ?', (image_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return np.frombuffer(result[0], dtype=np.float32)
    return None
