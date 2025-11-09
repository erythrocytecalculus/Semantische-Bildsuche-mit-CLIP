from annoy import AnnoyIndex
import sqlite3
import os

def get_annoy_index(dim, path=None):
    """Create or load Annoy index from disk."""
    index = AnnoyIndex(dim, 'angular')
    if path and os.path.exists(path):
        index.load(path)
    return index

def save_annoy_index(index, path, trees=2):
    """Build trees and save Annoy index."""
    index.build(trees)
    index.save(path)

def create_sqlite_db(path):
    """Initialize SQLite database with image metadata table."""
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_image_path(db_path, filepath):
    """Add file path in SQLite DB and return its ID."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO images (filepath) VALUES (?)', (filepath,))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id

def query_image_path(db_path, image_id):
    """Fetch image filepath using its unique ID."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT filepath FROM images WHERE id = ?', (image_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None
