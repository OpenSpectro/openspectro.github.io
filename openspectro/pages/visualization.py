from flask import Blueprint, render_template, request
from openspectro import load_biomarkers

entity_name = 'visualization'

visualization = Blueprint(entity_name, __name__)

@visualization.route('/', methods=['GET', 'POST'])
def visualize():
    biomarkers = load_biomarkers()
    print("Loaded Biomarkers:", biomarkers)  # Debugging line

    if request.method == 'POST':
        # Process form data
        biomarker_id = request.form.get('biomarker_id')
        viz_type = request.form.get('viz_type')
        dimension = request.form.get('dimension')
        intensity_threshold = request.form.get('intensity_threshold', type=float)
        absorbance_threshold = request.form.get('absorbance_threshold', type=float)

        # Generate graph (pseudo-code, adjust according to your actual graph generation logic)
        graph_html = generate_graph(biomarker_id, viz_type, dimension, intensity_threshold, absorbance_threshold)

        return render_template(f"{entity_name}/{entity_name}.html", biomarkers=biomarkers, CSSLink = f"../static/css/{entity_name}/{entity_name}.css", graph_html=graph_html, form_data=request.form)

    return render_template(f"{entity_name}/{entity_name}.html", biomarkers=biomarkers, CSSLink = f"../static/css/{entity_name}/{entity_name}.css")

def generate_graph(biomarker_id, viz_type, dimension, intensity_threshold, absorbance_threshold):
    # Dummy graph generation for demonstration
    import plotly.graph_objs as go
    import numpy as np

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