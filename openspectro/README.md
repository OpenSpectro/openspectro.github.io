git clone https://github.com/OpenSpectro/openspectro.github.io.git
cd openspectro.github.io
pip install -r requirements.txt
gunicorn main:app