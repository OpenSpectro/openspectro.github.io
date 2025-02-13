from flask import Blueprint, render_template
import os
import json

database = Blueprint('database', __name__)

@database.route('/', methods=['GET', 'POST'])
def page():
    # Path to the JSON file containing biomarker data
    database_json_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'database.json')
    
    # Load the JSON data
    with open(database_json_path, 'r') as file:
        biomarkers = json.load(file)
    
    # Pass the biomarkers data to the template
    return render_template("database/database.html", CSSLink="/static/css/database/style.css", biomarkers=biomarkers, iconLink = "../static/figure/database/database.png")