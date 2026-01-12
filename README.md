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
- **UMAP for Visualization**: Visualizes high-dimensional image embeddings in 2D space using **UMAP**. It helps you explore and understand the relationships between images based on their visual features.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [UMAP Visualization](#umap-visualization)

---

## Project Structure

```
├── app.py # Main web server (Flask)
├── run.py # Image encoding and database setup
├── templates/
│ └── search.html # Web UI HTML template
├── open_images/
│ ├── resize/ # Resized images for processing
│ └── validation/ # Validation set of images
├── umap_interactive.py # UMAP visualization and interactive plot setup
├── umap_thumbnails.py # Code to add thumbnails to UMAP plot
├── umap.html # UMAP HTML template for rendering
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
- **umap-learn** (for dimensionality reduction)

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
pip install torch pillow faiss numpy flask git+https://github.com/openai/CLIP.git umap-learn

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

## UMAP Visualization

<img width="1919" height="621" alt="Screenshot 2026-01-09 201237" src="https://github.com/user-attachments/assets/8e10cedd-a3c8-4915-841d-3cfb061596d1" />


### Interactive 2D Visualization of CLIP Image Embeddings

This project also includes a **UMAP visualization** of the image embeddings. UMAP (Uniform Manifold Approximation and Projection) reduces the high-dimensional image features (from the CLIP model) to 2D, making it easy to explore the relationships between images visually. The UMAP projection allows you to:

- **Visualize Clusters**: Images that are visually similar will be grouped together.
- **Hover Over Points**: Display coordinates when hovering over points on the UMAP plot.

To access the UMAP visualization, visit http://127.0.0.1:5000/umap

---

## Example Search

Search Term: "dog playing in the snow"

<img width="1681" height="837" alt="image" src="https://github.com/user-attachments/assets/6360bc0f-fa6b-4943-9204-a3fc07a099a6" />

---

## Credits and References

- OpenAI CLIP: https://github.com/openai/CLIP

- FAISS: https://github.com/facebookresearch/faiss

- Flask: https://flask.palletsprojects.com/

- UMAP: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)

- Open Images Dataset: https://github.com/cvdfoundation/open-images-dataset

---
