from flask import Blueprint, render_template
import os
import json
from openspectro import BIOMARKERS

entity_name = 'detail'

detail = Blueprint(entity_name, __name__)

@detail.route('/<int:id>', methods=['GET'])
def page(id):
    # Path to the JSON file containing biomarker data
    biomarkers = BIOMARKERS
    
    # Find the biomarker with the matching ID
    biomarker = next((b for b in biomarkers if b['ID'] == id), None)
    
    if biomarker is None:
        return "Biomarker not found", 404

    # Pass the biomarker data to the template
    return render_template(f"{entity_name}/{entity_name}.html", CSSLink=f"/static/css/{entity_name}/{entity_name}.css", biomarker=biomarker, moleculeLink=f"../static/molecules/{id}.png", cuvetteLink=f"../static/cuvette/{id+1}.png")