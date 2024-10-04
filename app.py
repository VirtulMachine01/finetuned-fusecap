from flask import Flask, render_template
from modules.routes import init_routes

# Create Flask app
app = Flask(__name__)

# Initialize routes
init_routes(app)

if __name__ == "__main__":
    app.run(debug=True)
