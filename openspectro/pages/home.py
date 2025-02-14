from flask import Blueprint, render_template

entity_name = 'home'

home = Blueprint('home', __name__)

@home.route('/', methods=['GET', 'POST'])
def page():
    return render_template(f"{entity_name}/{entity_name}.html", CSSLink = f"../static/css/{entity_name}/{entity_name}.css")