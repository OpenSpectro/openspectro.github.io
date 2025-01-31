import os
from flask import Flask, render_template, request

app = Flask(__name__)

# Base directory for HTML files
BASE_DIR = 'Absorbance-Graph'

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get subfolders in the base directory
    subfolders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

    # Determine selected folder
    selected_folder = request.form.get('folder') if request.method == 'POST' else 'Glucose'
    if selected_folder not in subfolders:
        # Fallback if 'Glucose' isn't valid or doesn't exist
        selected_folder = subfolders[0] if subfolders else ''

    # Get HTML files in the selected folder
    folder_path = os.path.join(BASE_DIR, selected_folder)
    html_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.html')],
        key=lambda x: int(x.split('.')[0])
    )

    # Determine file index
    file_index = int(request.form.get('file_index', 1))
    # Ensure the file exists in the folder; else pick the first one
    selected_file = f"{file_index}.html" if f"{file_index}.html" in html_files else html_files[0]

    return render_template(
        'index.html',
        subfolders=subfolders,
        selected_folder=selected_folder,
        html_files=html_files,
        selected_file=selected_file,
        file_index=file_index,
        threshold=file_index  # Pass threshold as label
    )

@app.route('/html/<folder>/<file>', methods=['GET'])
def serve_html(folder, file):
    """
    Serve HTML content for the iframe.
    """
    file_path = os.path.join(BASE_DIR, folder, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

if __name__ == '__main__':
    app.run(debug=True, port=8000)
