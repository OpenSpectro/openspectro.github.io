from flask import Blueprint, render_template
import os
import json

entity_name = 'fluorescence'

fluorescence = Blueprint(entity_name, __name__)

@fluorescence.route('/<int:id>', methods=['GET'])
def page(id):
    # Path to the JSON file containing biomarker data
    database_json_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'database.json')
    
    # Load the JSON data
    with open(database_json_path, 'r') as file:
        biomarkers = json.load(file)
    
    # Find the biomarker with the matching ID
    biomarker = next((b for b in biomarkers if b['ID'] == id), None)
    
    if biomarker is None:
        return "Biomarker not found", 404

    # Pass the biomarker data to the template
    return render_template(f"{entity_name}/{entity_name}.html", CSSLink=f"/static/css/{entity_name}/{entity_name}.css", biomarker=biomarker)