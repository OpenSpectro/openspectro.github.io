from flask import Blueprint, render_template
import os
import json
from openspectro import BIOMARKERS

entity_name = 'optimization'

optimization = Blueprint(entity_name, __name__)

@optimization.route('/2D/<int:id>', methods=['GET'])
def page2D(id):
    # Path to the JSON file containing biomarker data
    biomarker = BIOMARKERS

    # Pass the biomarker data to the template
    return render_template(f"{entity_name}/{entity_name}.html", CSSLink=f"/static/css/{entity_name}/{entity_name}.css", biomarker=biomarker)

@optimization.route('/3D/<int:id>', methods=['GET'])
def page3D(id):
    # Path to the JSON file containing biomarker data
    biomarker = BIOMARKERS

    # Pass the biomarker data to the template
    return render_template(f"{entity_name}/{entity_name}.html", CSSLink=f"/static/css/{entity_name}/{entity_name}.css", biomarker=biomarker)