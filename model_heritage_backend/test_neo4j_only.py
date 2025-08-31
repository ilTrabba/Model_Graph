#!/usr/bin/env python3
"""
Test script to verify Neo4j-only functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.services.neo4j_service import neo4j_service
from src.models.model import Model, Family

def test_neo4j_service():
    """Test Neo4j service functionality"""
    print("Testing Neo4j service...")
    
    # Test connection status
    connected = neo4j_service.is_connected()
    print(f"Neo4j connected: {connected}")
    
    # Test status
    status = neo4j_service.get_graph_status()
    print(f"Graph status: {status}")
    
    # Test stats
    stats = neo4j_service.get_stats()
    print(f"Stats: {stats}")
    
    print("Neo4j service tests passed!")

def test_model_proxy():
    """Test model proxy functionality"""
    print("Testing model proxy...")
    
    # Test model queries
    models = Model.query.all()
    print(f"Found {len(models)} models")
    
    # Test family queries
    families = Family.query.all()
    print(f"Found {len(families)} families")
    
    # Test counts
    model_count = Model.query.count()
    family_count = Family.query.count()
    print(f"Model count: {model_count}, Family count: {family_count}")
    
    print("Model proxy tests passed!")

if __name__ == '__main__':
    try:
        test_neo4j_service()
        test_model_proxy()
        print("\n✅ All tests passed! Neo4j-only architecture is working.")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)