from flask import Blueprint, render_template
import os
import json

entity_name = 'database'

database = Blueprint(entity_name, __name__)

@database.route('/', methods=['GET', 'POST'])
def page():
    # Path to the JSON file containing biomarker data
    database_json_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'database.json')
    
    # Load the JSON data
    with open(database_json_path, 'r') as file:
        biomarkers = json.load(file)

    # Pass the biomarkers data to the template
    return render_template(f"{entity_name}/{entity_name}.html", CSSLink=f"/static/css/{entity_name}/{entity_name}.css", biomarkers=biomarkers, iconLink = f"../static/figure/{entity_name}/database.png")