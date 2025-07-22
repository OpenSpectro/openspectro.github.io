from flask import Flask
from datetime import timedelta
import numpy as np
import pandas as pd
import os
import json

lineSeperator = ""
WEBSITE_NAME = "OpenSpectro"
clear_ID = 998
background_ID = 999

laser_wavelengths = np.arange(380, 1101, 1)
spectrometer_wavelengths = pd.read_csv("openspectro/database/PassThrough/2D/Background.csv")["# Wavelength"].values
background_intensity = pd.read_csv("openspectro/database/PassThrough/2D/Background.csv")["Intensity"].values.astype(float)

def load_biomarkers():
    database_json_path = os.path.join(os.path.dirname(__file__), 'database', 'database.json')
    with open(database_json_path, 'r') as file:
        biomarkers = json.load(file)
    # print(f"All biomarkers are {biomarkers}")
    return biomarkers

BIOMARKERS = load_biomarkers()


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

    # # Visualization page
    from .pages.visualization import visualization
    
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
    app.register_blueprint(visualization, url_prefix = "/visualization")
    # app.register_blueprint(fluorescence, url_prefix = "/fluorescence")
    # app.register_blueprint(absorbance, url_prefix = "/absorbance")
    # app.register_blueprint(optimization, url_prefix = "/optimization")
    # app.register_blueprint(instruction, url_prefix = "/instruction")

    return app