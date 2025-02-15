from flask import Blueprint, render_template
from openspectro import load_biomarkers

entity_name = 'database'

database = Blueprint(entity_name, __name__)

@database.route('/', methods=['GET', 'POST'])
def page():
    # Path to the JSON file containing biomarker data
    biomarkers = load_biomarkers()

    # Pass the biomarkers data to the template
    return render_template(f"{entity_name}/{entity_name}.html", CSSLink=f"/static/css/{entity_name}/{entity_name}.css", biomarkers=biomarkers, iconLink = f"../static/figure/{entity_name}/database.png")