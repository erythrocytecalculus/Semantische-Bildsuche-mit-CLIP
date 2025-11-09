# Semantic Image Search with CLIP

A modular Python project for searching images based on natural language descriptions using OpenAI’s CLIP model and the Annoy vector database. Supports efficient retrieval, large datasets, GPU acceleration, and features a modern web-based UI.

---

## Features

- Text-to-image semantic image retrieval in English
- Scalable database with Annoy for fast search
- Supports GPU (CUDA) for fast encoding
- Modular code structure: easy to maintain/extend
- Simple Flask web interface for queries

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)
- [Credits](#credits-and-references)

---

## Project Structure

├── app.py # Main web server (Flask)
├── run.py # Image encoding and database setup
├── clip_utils.py # CLIP embedding functions
├── db_manager.py # Database management (Annoy/SQLite)
├── image_utils.py # Image viewing/copying helpers
├── settings.py # Config variables
├── templates/
│ └── search.html # Web UI HTML template
└── README.md # Project documentation


---

## Requirements

- Python 3.8+
- torch
- pillow
- annoy
- flask
- numpy
- OpenAI CLIP (install via pip and Git)

---

## Setup

git clone https://your-repo-url
cd your-repo-folder

pip install torch pillow annoy numpy flask git+https://github.com/openai/CLIP.git


1. Prepare a folder of images (jpg, png, jpeg).
2. Edit `settings.py` to set custom paths, if needed.

---

## Usage

### Database Build

python run.py

*(Ensure your images folder is set properly in run.py)*

### Start Web Server

python app.py

- Access via browser: [http://localhost:5000](http://localhost:5000)
- Enter an English description to view matching images.

---

## Customization

- Change image folder and DB filenames via `settings.py`
- Update `search.html` to customize UI design
- Add your own features: filtering, detailed logs, downloads

---

## License

MIT License — see LICENSE file for details

---

## Credits and References

- OpenAI CLIP: https://github.com/openai/CLIP
- Annoy: https://github.com/spotify/annoy
- Flask: https://flask.palletsprojects.com/
- Modularized and enhanced from coursework, refactored for clarity and extensibility.

---

## Example Search

> "dog playing in the snow"
- Returns the most relevant images matching this description in your images folder.

---

## Screenshots

(Add screenshots of your web UI or results here for visual appeal!)

---
