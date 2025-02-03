from flask import Flask, render_template, request
import os

app = Flask(__name__)
BASE_DIR = 'Absorbance-Graph'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/database', methods=['GET', 'POST'])
def database():
    # Folder selection logic
    subfolders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]
    selected_folder = request.form.get('folder', 'Blood_Glucose_Control_Solution-2025-01-21-11-36-56')
    
    # Threshold handling
    folder_path = os.path.join(BASE_DIR, selected_folder)
    html_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.html')],
        key=lambda x: int(x.split('.')[0])
    )
    thresholds = [int(f.split('.')[0]) for f in html_files]
    
    # Get selected threshold
    selected_threshold = int(request.form.get('threshold', thresholds[0])) if thresholds else 0
    current_index = thresholds.index(selected_threshold) if thresholds else 0
    
    return render_template('database.html',
        subfolders=subfolders,
        selected_folder=selected_folder,
        thresholds=thresholds,
        current_index=current_index,
        selected_threshold=selected_threshold,
        selected_file=f"{selected_threshold}.html"
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
    app.run(debug=True, port=8888)