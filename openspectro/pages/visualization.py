from flask import Blueprint, render_template, request
from openspectro import BIOMARKERS, clear_ID, background_ID, spectrometer_wavelengths, background_intensity, laser_wavelengths, WEBSITE_NAME
import plotly.graph_objs as go
import numpy as np
import os
import pandas as pd


entity_name = 'visualization'

visualization = Blueprint(entity_name, __name__)

@visualization.route('/', methods=['GET', 'POST'])
def page():
    biomarkers = BIOMARKERS
    message_box = ""

    if request.method == 'POST':
        print(f"POST Form is {request.form}")
        # Process form data
        biomarker_id = request.form.get('biomarker_id')
        orientation = request.form.get('orientation')
        viz_type = request.form.get('viz_type')
        dimension = request.form.get('dimension')

        intensity_threshold = request.form.get('intensity_threshold', type=float)
        absorbance_threshold = request.form.get('absorbance_threshold', type=float)

        # Check for missing thresholds and set defaults
        if intensity_threshold is None:
            intensity_threshold = 100.0
            message_box += "Using default Intensity Threshold = 100. "

        if absorbance_threshold is None:
            absorbance_threshold = 0.1
            message_box += "Using default Absorbance Threshold = 0.1."

        # Generate the graph
        graph_html = generate_graph(
            biomarker_id, orientation, viz_type, dimension,
            intensity_threshold, absorbance_threshold
        )

        return render_template(
            f"{entity_name}/{entity_name}.html",
            biomarkers=biomarkers,
            CSSLink=f"../static/css/{entity_name}/{entity_name}.css",
            graph_html=graph_html,
            form_data=request.form,
            message_box=message_box  # 👈 pass to template
        )

    return render_template(
        f"{entity_name}/{entity_name}.html",
        biomarkers=biomarkers,
        CSSLink=f"../static/css/{entity_name}/{entity_name}.css"
    )

def file_search(orientation, biomarker_id, dimension="3D"):
    biomarkers = BIOMARKERS
    # print(f"Search criteria is {orientation}, {biomarker_id} and {dimension}")
    biomarker_name = next((b for b in biomarkers if b['ID'] == int(biomarker_id)), None)["BiomarkerName"]
    # print(f"Biomarker name is {biomarker_name}")
    if orientation == "Orthogonal":
        database_dir = f"{WEBSITE_NAME.lower()}/database/{orientation}/"
    elif orientation == "PassThrough":
        database_dir = f"{WEBSITE_NAME.lower()}/database/{orientation}/{dimension}/"
    
    # Initialize biomarker_intensity as None
    biomarker_intensity = None
    
    # Iterate through files in the directory
    for file_name in os.listdir(database_dir):
        if file_name.startswith(biomarker_name) and file_name.endswith('.csv'):
            file_path = os.path.join(database_dir, file_name)
            # Load the CSV file using numpy
            if dimension == "2D":
                return np.genfromtxt(file_path, delimiter=','), biomarker_name
            elif dimension == "3D":
                biomarker_intensity = pd.read_csv(file_path, header = None)
                return biomarker_intensity.iloc[1:].values, biomarker_name
    
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
            sample_intensity -= background_intensity
            
            # Create the main surface
            fig = go.Figure(data=[go.Surface(
                x=spectrometer_wavelengths,
                y=laser_wavelengths,
                z=sample_intensity
            )])

            # --- Add the diagonal plane y = x (for x >= 0, y >= 0) ---
            # 1) Define a 1D range for x
            x_min = max(0, spectrometer_wavelengths.min())  # to ensure positivity
            x_max = spectrometer_wavelengths.max()
            x_vals = np.linspace(x_min, x_max, 50)

            # 2) Decide on a range for z
            z_min = sample_intensity.min()
            z_max = sample_intensity.max()
            z_vals = np.linspace(z_min, z_max, 50)

            # 3) Create a meshgrid so we can form a rectangular plane in (x,z)
            X, Z = np.meshgrid(x_vals, z_vals)

            # 4) y = x along this plane
            Y = X

            # 5) Add the diagonal plane as a partially transparent surface
            plane_surface = go.Surface(
                x=X,
                y=Y,
                z=Z,
                showscale=False,      # Hide separate color scale
                opacity=0.3           # Partially transparent for clarity
            )
            fig.add_trace(plane_surface)
            # --- End plane addition ---

            fig.update_layout(
                title=f'Orthogonal Intensity graph for sample: {sample_name}',
                width=1200,    # <--- Adjust as desired
                height=700,    # <--- Adjust as desired
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
                        title='Orthogonal Intensity'
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=0.5)
                    )
                )
            )

            return fig.to_html(full_html=False)

        elif viz_type == "intensity":
            sample_intensity -= background_intensity
            
            # Create the main surface
            fig = go.Figure(data=[go.Surface(
                x=spectrometer_wavelengths,
                y=laser_wavelengths,
                z=sample_intensity
            )])

            # --- Add the diagonal plane y = x (for x >= 0, y >= 0) ---
            # 1) Define a 1D range for x
            x_min = max(0, spectrometer_wavelengths.min())  # to ensure positivity
            x_max = spectrometer_wavelengths.max()
            x_vals = np.linspace(x_min, x_max, 50)

            # 2) Decide on a range for z
            z_min = sample_intensity.min()
            z_max = sample_intensity.max()
            z_vals = np.linspace(z_min, z_max, 50)

            # 3) Create a meshgrid so we can form a rectangular plane in (x,z)
            X, Z = np.meshgrid(x_vals, z_vals)

            # 4) y = x along this plane
            Y = X

            # 5) Add the diagonal plane as a partially transparent surface
            plane_surface = go.Surface(
                x=X,
                y=Y,
                z=Z,
                showscale=False,      # Hide separate color scale
                opacity=0.3           # Partially transparent for clarity
            )
            fig.add_trace(plane_surface)
            # --- End plane addition ---

            fig.update_layout(
                title=f'Orthogonal Intensity graph for sample: {sample_name}',
                width=1200,    # <--- Adjust as desired
                height=700,    # <--- Adjust as desired
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
                        title='Orthogonal Intensity'
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=0.5)
                    )
                )
            )

            return fig.to_html(full_html=False)

    elif orientation == "PassThrough":

        sample_intensity, sample_name = file_search(orientation, biomarker_id, dimension)
        clear_intensity, _ = file_search(orientation, clear_ID, dimension)
        # print(f"sample_intensity is {sample_intensity}")

        if viz_type == "absorbance":
            if dimension == "2D":
                numerator = sample_intensity - background_intensity
                denominator = clear_intensity - background_intensity
                # Avoid divide by zero by replacing non-positive values
                safe_numerator   = np.where(numerator   <= 0, 1e-10, numerator)
                safe_denominator = np.where(denominator <= 0, 1e-10, denominator)

                # Compute ratio
                ratio = np.divide(safe_numerator, safe_denominator)

                # 1) If the absolute difference < threshold => ratio=1 => A=0
                difference = np.abs(safe_numerator - safe_denominator)
                ratio[difference < intensity_threshold] = 1.0

                # 2) If numerator < threshold => ratio=1 => A=0
                ratio[numerator < intensity_threshold] = 1.0

                # 3) If numerator > denominator => ratio=1 => A=0
                mask = (safe_numerator > safe_denominator)
                ratio[mask] = 1.0

                # 4) ratio <= 0 is invalid for log, set ratio=1 => A=0
                ratio[ratio <= 0] = 1.0

                # Compute A
                A = -np.log10(ratio)
                A[A < 0] = 0  # Just in case any floating precision issues

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=spectrometer_wavelengths,
                    y=A,
                    mode='lines+markers',
                    name=f'2D Pass-through Absorbance Curve graph for sample {sample_name} (Intensity Threshold = {intensity_threshold}; Absorbance Threshold = {absorbance_threshold})',
                    hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>'
                ))

                return fig.to_html(full_html=False)

            elif dimension == "3D":
                numerator = sample_intensity - background_intensity
                denominator = clear_intensity - background_intensity
                        # Avoid divide by zero by replacing non-positive values
                safe_numerator   = np.where(numerator   <= 0, 1e-10, numerator)
                safe_denominator = np.where(denominator <= 0, 1e-10, denominator)

                # Compute ratio
                ratio = np.divide(safe_numerator, safe_denominator)

                # 1) If the absolute difference < threshold => ratio=1 => A=0
                difference = np.abs(safe_numerator - safe_denominator)
                ratio[difference < intensity_threshold] = 1.0

                # 2) If numerator < threshold => ratio=1 => A=0
                ratio[numerator < intensity_threshold] = 1.0

                # 3) If numerator > denominator => ratio=1 => A=0
                mask = (safe_numerator > safe_denominator)
                ratio[mask] = 1.0

                # 4) ratio <= 0 is invalid for log, set ratio=1 => A=0
                ratio[ratio <= 0] = 1.0

                # Compute A
                A = -np.log10(ratio)
                A[A < 0] = 0  # Just in case any floating precision issues

                # 6) Create a 3D surface plot with Plotly
                fig = go.Figure(data=[go.Surface(
                    x=spectrometer_wavelengths,
                    y=laser_wavelengths,
                    z=A
                )])

                # Adjust layout, labels, camera, etc.
                fig.update_layout(
                    title=f'3D Absorbance for sample: {sample_name} (Intensity Threshold = {intensity_threshold}; Absorbance Threhsold = {absorbance_threshold})',
                    width=1200,    # <--- Adjust as desired
                    height=700,    # <--- Adjust as desired
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
                            title=f'Absorbance'
                        ),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=0.5)
                        )
                    )
                )

                return fig.to_html(full_html=False)

        elif viz_type == "intensity":
            if dimension == "2D":
                sample_intensity -= background_intensity
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=spectrometer_wavelengths,
                    y=sample_intensity,
                    mode='lines+markers',
                    name=f'2D Pass-through Intensity Curve graph for sample {sample_name}',
                    hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<extra></extra>'
                ))
                return fig.to_html(full_html=False)
            elif dimension == "3D":
                sample_intensity -= background_intensity
                fig = go.Figure(data=[go.Surface(
                    x=spectrometer_wavelengths,
                    y=laser_wavelengths,
                    z=sample_intensity
                )])
                fig.update_layout(
                    title=f'3D Pass-through Intensity graph for sample: {sample_name}',
                    width=1200,    # <--- Adjust as desired
                    height=700,    # <--- Adjust as desired
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
                            title='Pass-through Intensity'
                        ),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=0.5)
                        )
                    )
                )
                return fig.to_html(full_html=False)

