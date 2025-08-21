#!/usr/bin/env python3
"""
Database initialization script for Model Heritage backend.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.main import app
from src.models.user import db

def init_database():
    """Initialize the database and create all tables."""
    with app.app_context():
        print("Creating database tables...")
        db.create_all()
        print("Database initialization complete.")

if __name__ == '__main__':
    init_database()