import os

class Config:
    """Application configuration"""
    
    # Neo4j Database
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')
    
    # Upload settings
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'safetensors', 'pt', 'bin', 'pth'}
    
    # Weights storage
    WEIGHTS_FOLDER = 'weights'
    CENTROIDS_FOLDER = 'weights/centroids'

    # Models storage
    MODEL_FOLDER = 'weights/models'
    
    # Other settings
    SECRET_KEY = 'asdf#FGSgvasgf$5$WGT'