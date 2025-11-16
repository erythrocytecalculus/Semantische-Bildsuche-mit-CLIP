# Semantic Image Search with CLIP

A modular Python project that allows you to search for images based on natural language descriptions using OpenAI’s CLIP model and the FAISS vector database. The system supports efficient retrieval, large datasets, GPU acceleration, and features a modern web-based UI built with Flask.

---

## Features

- **Text-to-Image Semantic Image Retrieval**: Search for images using natural language descriptions in English.
- **FAISS for Fast Search**: Utilizes FAISS for scalable and fast image retrieval from large datasets.
- **GPU Acceleration**: Leverages GPU (CUDA) for faster image encoding using CLIP.
- **Modular Architecture**: Easy to extend or modify for custom use cases.
- **Modern Web UI**: A clean and responsive web interface for querying and displaying results.
- **Customizable**: Simple to adapt to your own dataset or integrate additional features.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Customization](#customization)

---

## Project Structure

```
├── app.py # Main web server (Flask)
├── run.py # Image encoding and database setup
├── clip_model.py # CLIP model embedding functions
├── db_manager.py # FAISS database management
├── image_utils.py # Image viewing/copying helpers
├── settings.py # Configuration variables (VECTOR_SIZE, ANNOY_TREE_COUNT)
├── templates/
│ └── search.html # Web UI HTML template
├── open_images/
│ ├── resize/ # Resized images for processing
│ └── validation/ # Validation set of images
└── README.md # Project documentation
```

---

## Requirements

- **Python 3.8+**
- **torch**
- **pillow**
- **faiss**
- **flask**
- **numpy**
- **OpenAI CLIP** (install via pip and Git)

---

## Setup

### Step 1: Download the Open Images Dataset from AWS S3

To use the Open Images dataset, download it from the AWS S3 bucket with the following command:

```
aws s3 --no-sign-request sync s3://open-images-dataset/validation open_images/validation
```
This command will download the validation set (approximately 12GB) of the Open Images dataset, which contains 41,620 images. These images will be stored in the open_images/validation folder.

### Step 2: Clone the repository
```
git clone https://github.com/erythrocytecalculus/Semantische-Bildsuche-mit-CLIP.git
```

### Step 3: Navigate to the project directory
```
cd your-repo-folder
```

### Step 4: Install dependencies
```
pip install torch pillow annoy numpy flask git+https://github.com/openai/CLIP.git
```
### Step 5: Configure paths

You’ll need to set the paths for the image folders (e.g., validation and resize) in the respective files.

- In run.py: Modify the paths for the image_folder and output_folder (where the resized images are saved).
```
if __name__ == "__main__":
    # Set your folders here
    image_folder = r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\OpenImages\validation"  # Path to image folder
    output_folder = r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\OpenImages\resize"   # Path to output resized images
```

- In app.py: Set the path for the folder where resized images are stored.
```
# Folder where resized images are stored (processed)
image_folder = r"C:\Users\TUFA17\OneDrive\Desktop\Semantische-Bildsuche-mit-CLIP\OpenImages\resize"

```

---

## Usage

### Step 1: Build the Database
Before starting the server, you need to process your images and build the FAISS index.
```
python run.py
```
*Ensure that the images folder is properly set in the run.py script and points to the open_images/validation folder (or wherever your images are stored)*

### Step 2: Start the Web Server
Once the database is built, start the Flask web server:
```
python app.py
```
- Access via browser: http://localhost:5000
- Search: Enter an English text description to search for matching images from the Open Images dataset

---

## Example Search

Search Term: "dog playing in the snow"
- This will return the most relevant images matching the description from the open_images/validation folder.
<img width="1681" height="837" alt="image" src="https://github.com/user-attachments/assets/6360bc0f-fa6b-4943-9204-a3fc07a099a6" />

---

## Credits and References

- OpenAI CLIP: https://github.com/openai/CLIP

- FAISS: https://github.com/facebookresearch/faiss

- Flask: https://flask.palletsprojects.com/

- Open Images Dataset: https://github.com/cvdfoundation/open-images-dataset

- Annoy: https://github.com/spotify/annoy

---
