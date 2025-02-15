from flask import Blueprint, render_template, request
from openspectro import BIOMARKERS
import os

entity_name = 'visualization'

visualization = Blueprint(entity_name, __name__)

@visualization.route('/', methods=['GET', 'POST'])
def page():
    biomarkers = BIOMARKERS
    # print("Loaded Biomarkers:", biomarkers)  # Debugging line

    if request.method == 'POST':
        print(f"POST Form is {request.form}")
        # Process form data
        biomarker_id = request.form.get('biomarker_id')
        orientation = request.form.get('orientation')
        viz_type = request.form.get('viz_type')
        dimension = request.form.get('dimension')
        intensity_threshold = request.form.get('intensity_threshold', type=float)
        absorbance_threshold = request.form.get('absorbance_threshold', type=float)

        # Generate graph (pseudo-code, adjust according to your actual graph generation logic)
        graph_html = generate_graph(biomarker_id, orientation, viz_type, dimension, intensity_threshold, absorbance_threshold)

        return render_template(f"{entity_name}/{entity_name}.html", biomarkers=biomarkers, CSSLink=f"../static/css/{entity_name}/{entity_name}.css", graph_html=graph_html, form_data=request.form)

    return render_template(f"{entity_name}/{entity_name}.html", biomarkers=biomarkers, CSSLink=f"../static/css/{entity_name}/{entity_name}.css")

def file_search(orientation, biomarker_id, dimension):
    biomarkers = BIOMARKERS
    biomarker_name = next((b for b in biomarkers if b['ID'] == biomarker_id), None)["BiomarkerName"]
    if orientation == "Orthogonal":
        database_dir = f"../database/{orientation}/"
    elif orientation == "PassThrough":
        database_dir = f"../database/{orientation}/{dimension}/"
    
    # Initialize biomarker_intensity as None
    biomarker_intensity = None
    
    # Iterate through files in the directory
    for file_name in os.listdir(database_dir):
        if file_name.startswith(biomarker_name) and file_name.endswith('.csv'):
            file_path = os.path.join(database_dir, file_name)
            try:
                # Load the CSV file using numpy
                biomarker_intensity = np.genfromtxt(file_path, delimiter=',')
                break  # Stop after finding the first matching file
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue
    
    return biomarker_intensity

def generate_graph(biomarker_id, orientation, viz_type, dimension, intensity_threshold, absorbance_threshold):
    # Dummy graph generation for demonstration
    import plotly.graph_objs as go
    import numpy as np

    if orientation == "Orthogonal":
        
        sample_intensity = file_search(orientation, biomarker_id, dimension)
        
        if viz_type == ""

    elif orientation == "PassThrough":


    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

    if dimension == '3D':
        fig.update_layout(scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ))
    else:
        fig.update_layout(
            xaxis_title='X Axis',
            yaxis_title='Y Axis'
        )

    return fig.to_html(full_html=False)