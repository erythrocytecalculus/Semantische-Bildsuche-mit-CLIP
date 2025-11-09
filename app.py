from flask import Flask, request, render_template
from clip_model import encode_text
from db_manager import get_annoy_index, query_image_path
import settings

app = Flask(__name__)
annoy_index = get_annoy_index(settings.VECTOR_SIZE, settings.ANNOY_PATH)

@app.route('/', methods=['GET', 'POST'])
def search_images():
    found_images = []
    if request.method == 'POST':
        desc = request.form.get("description")
        if desc and desc.isascii():  # Basic English filter
            vec = encode_text(desc)
            ids = annoy_index.get_nns_by_vector(vec, 10)
            found_images = [query_image_path(settings.SQLITE_PATH, _id) for _id in ids]
    return render_template('search.html', images=found_images)

if __name__ == '__main__':
    app.run(debug=True)
