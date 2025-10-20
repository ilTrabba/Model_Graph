from flask import Blueprint, jsonify, request
from src.services.neo4j_service import neo4j_service
from src.db_entities.entity import Model, Family
import logging

logger = logging.getLogger(__name__)

graph_bp = Blueprint('graph', __name__)

@graph_bp.route('/graph/status', methods=['GET'])
def get_graph_status():
    """Get Neo4j connection status"""
    try:
        neo4j_connected = neo4j_service.is_connected()
        
        if not neo4j_connected:
            return jsonify({
                'neo4j_connected': False,
                'error': 'Neo4j not connected'
            }), 503
        
        # Get Neo4j stats
        graph_data = neo4j_service.get_full_graph()
        
        return jsonify({
            'neo4j_connected': True,
            'neo4j_nodes': graph_data.get('node_count', 0),
            'neo4j_edges': graph_data.get('edge_count', 0),
            'message': 'Neo4j-only architecture active'
        })
        
    except Exception as e:
        logger.error(f"Failed to get graph status: {e}")
        return jsonify({
            'neo4j_connected': False,
            'error': 'Failed to get status',
            'details': str(e)
        }), 500

@graph_bp.route('/graph/full', methods=['GET'])
def get_full_graph():
    """Return complete graph data for visualization"""
    try:
        if not neo4j_service.is_connected():
            return jsonify({
                'error': 'Neo4j not connected',
                'nodes': [],
                'edges': [],
                'message': 'Please check Neo4j connection and try syncing data'
            }), 503
        
        graph_data = neo4j_service.get_full_graph()
        
        if 'error' in graph_data:
            return jsonify({
                'error': graph_data['error'],
                'nodes': [],
                'edges': []
            }), 500
        
        return jsonify(graph_data)
        
    except Exception as e:
        logger.error(f"Failed to get full graph: {e}")
        return jsonify({
            'error': 'Failed to retrieve graph data',
            'details': str(e),
            'nodes': [],
            'edges': []
        }), 500

@graph_bp.route('/graph/family/<family_id>', methods=['GET'])
def get_family_subgraph(family_id):
    """Get specific family subgraph"""
    try:
        if not neo4j_service.is_connected():
            return jsonify({
                'error': 'Neo4j not connected',
                'family_id': family_id,
                'nodes': [],
                'edges': []
            }), 503
        
        graph_data = neo4j_service.get_family_subgraph(family_id)
        
        if 'error' in graph_data:
            return jsonify({
                'error': graph_data['error'],
                'family_id': family_id,
                'nodes': [],
                'edges': []
            }), 404 if 'not found' in graph_data['error'].lower() else 500
        
        return jsonify(graph_data)
        
    except Exception as e:
        logger.error(f"Failed to get family subgraph for {family_id}: {e}")
        return jsonify({
            'error': 'Failed to retrieve family subgraph',
            'details': str(e),
            'family_id': family_id,
            'nodes': [],
            'edges': []
        }), 500

@graph_bp.route('/graph/model/<model_id>/sync', methods=['POST'])
def sync_single_model(model_id):
    """Legacy endpoint - no longer needed in Neo4j-only architecture"""
    return jsonify({
        'success': True,
        'message': f'Model {model_id} - synchronization not needed in Neo4j-only architecture',
        'architecture': 'neo4j-only'
    })