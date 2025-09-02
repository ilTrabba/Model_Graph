#!/usr/bin/env python3
"""
Database initialization script for Model Heritage backend.
Since we're using Neo4j-only architecture, this script only sets up Neo4j constraints.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.main import app
from src.services.neo4j_service import neo4j_service

def init_database():
    """Initialize the Neo4j database constraints and indexes."""
    with app.app_context():
        print("Initializing Neo4j constraints and indexes...")
        if neo4j_service.is_connected():
            neo4j_service.create_constraints()
            print("Neo4j initialization complete.")
        else:
            print("Neo4j not connected. Please check your Neo4j configuration.")

if __name__ == '__main__':
    init_database()