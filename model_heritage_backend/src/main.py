import os
import sys
import logging

# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from src.config import Config
from src.routes.user import user_bp
from src.routes.models import models_bp
from src.routes.graph import graph_bp
from src.services.neo4j_service import neo4j_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config.from_object(Config)

# Enable CORS for all routes
CORS(app)

# Register blueprints
app.register_blueprint(user_bp, url_prefix='/api')
app.register_blueprint(models_bp, url_prefix='/api')
app.register_blueprint(graph_bp, url_prefix='/api')

# Initialize Neo4j constraints (no SQLAlchemy needed)
with app.app_context():
    try:
        if neo4j_service.is_connected():
            neo4j_service.create_constraints()
            logger.info("Neo4j constraints initialized")
        else:
            logger.warning("Neo4j not connected - graph features will be unavailable")
    except Exception as e:
        logger.error(f"Failed to initialize Neo4j: {e}")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "index.html not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
