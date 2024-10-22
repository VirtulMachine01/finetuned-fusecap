from flask import Flask
from modules.routes import init_routes
from flask_cors import CORS
from modules.config_loaders import load_config

# Create Flask app
app = Flask(__name__)
CORS(app)
config = load_config()
# Initialize routes
init_routes(app)

if __name__ == "__main__":
    app.run(debug=True,port=config["port_number"])
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # debug=False - for production environment

# python app.py
# gunicorn -w 1 -b 127.0.0.1:5000 app:app
# Python 3.12.4