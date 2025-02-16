from flask import Blueprint, render_template, request
from openspectro import BIOMARKERS, clear_ID, background_ID, spectrometer_wavelengths, background_intensity, laser_wavelengths
import plotly.graph_objs as go
import numpy as np
import os
import pandas as pd


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

def file_search(orientation, biomarker_id, dimension="3D"):
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
            # Load the CSV file using numpy
            if dimension == "2D":
                return np.genfromtxt(file_path, delimiter=',')
            elif dimension == "3D":
                biomarker_intensity = pd.read_csv(file_path, header = None)
                return biomarker_intensity.iloc[1:].values
    
    return biomarker_intensity, biomarker_name

def default_graph(dimension):
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

def generate_graph(biomarker_id, orientation, viz_type, dimension, intensity_threshold, absorbance_threshold):
    # Dummy graph generation for demonstration

    if orientation == "Orthogonal":
        
        sample_intensity, sample_name = file_search(orientation, biomarker_id)
        clear_intensity, _ = file_search(orientation, clear_ID)
        
        if viz_type == "fluorescence":
            return default_graph("3D")
        
        elif viz_type == "intensity":
            sample_intensity -= background_intensity
            fig = go.Figure(data=[go.Surface(
                x=spectrometer_wavelengths,
                y=laser_wavelengths,
                z=sample_intensity
            )])
            
            fig.update_layout(
                title=f'Intensity for sample: {sample_name}',
                scene=dict(
                    xaxis=dict(
                        title='Spectrometer Wavelength (nm)',
                        range=[spectrometer_wavelengths.min(), spectrometer_wavelengths.max()],
                    ),
                    yaxis=dict(
                        title='Laser Wavelength (nm)',
                        range=[laser_wavelengths.min(), laser_wavelengths.max()],
                    ),
                    zaxis=dict(
                        title='Transmission Intensity'
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=0.5)
                    )
                )
            )

            return fig.to_html(full_html=False)

    # elif orientation == "PassThrough":

    #     sample_intensity, sample_name = file_search(orientation, biomarker_id, dimension)
    #     clear_intensity, _ = file_search(orientation, clear_ID, dimension)

    #     if viz_type == "absorbance":
    #         if dimension == "2D":

    #         elif dimension == "3D":

    #     elif viz_type == "intensity":
    #         if dimension == "2D":

    #         elif dimension == "3D":



