from flask import Blueprint, render_template

home = Blueprint('home', __name__)

@home.route('/', methods=['GET', 'POST'])
def page():
    return render_template("home/home.html", CSSLink = "../static/css/home/style.css")