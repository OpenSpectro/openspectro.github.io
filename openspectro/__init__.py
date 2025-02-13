from flask import Flask
from datetime import datetime, timedelta
lineSeperator = ""
WEBSITE_NAME = "OpenSpectro"

def create_app():
    app = Flask(__name__)
    # app.config["SECRET_KEY"] = "letsGoOpenSpectro"
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)  # Sessions expire after 30 minutes
    
    # Database table page
    from .pages.home import home
    
    # Database table page
    from .pages.database import database

    # # Biomarker detail page
    from .pages.detail import detail

    # # Spectral Response Page
    # from .pages.fluorescence import fluorescence
    # from .pages.absorbance import absorbance

    # # Optimization table page
    # from .pages.optimization import optimization

    # # Instruction page for all formulas
    # from .pages.instruction import instruction

    # Register address here 
    app.register_blueprint(home, url_prefix = "/")
    app.register_blueprint(database, url_prefix = "/database")
    app.register_blueprint(detail, url_prefix = "/detail")
    # app.register_blueprint(fluorescence, url_prefix = "/fluorescence")
    # app.register_blueprint(absorbance, url_prefix = "/absorbance")
    # app.register_blueprint(optimization, url_prefix = "/optimization")
    # app.register_blueprint(instruction, url_prefix = "/instruction")

    return app