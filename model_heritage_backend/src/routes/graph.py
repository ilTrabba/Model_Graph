from flask import Blueprint, jsonify, request
from src.services.neo4j_service import neo4j_service
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
                'neo4j_nodes': 0,
                'neo4j_edges': 0,
                'message': 'Neo4j not connected'
            })
        
        # Get Neo4j stats
        graph_data = neo4j_service.get_full_graph()
        
        return jsonify({
            'neo4j_connected': True,
            'neo4j_nodes': graph_data.get('node_count', 0),
            'neo4j_edges': graph_data.get('edge_count', 0),
            'message': 'Neo4j connected and operational'
        })
        
    except Exception as e:
        logger.error(f"Failed to get graph status: {e}")
        return jsonify({
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

@graph_bp.route('/graph/sync', methods=['POST'])
def sync_graph_data():
    """Manual data refresh endpoint (Neo4j-only architecture)"""
    try:
        if not neo4j_service.is_connected():
            return jsonify({
                'success': False,
                'error': 'Neo4j not connected',
                'message': 'Cannot refresh without Neo4j connection'
            }), 503
        
        # Check if we should clear existing data first
        clear_existing = request.json.get('clear_existing', False) if request.json else False
        
        if clear_existing:
            neo4j_service.clear_all_data()
            logger.info("Cleared existing Neo4j data")
        
        # Refresh constraints and indexes
        neo4j_service.create_constraints()
        
        # Get current stats
        graph_data = neo4j_service.get_full_graph()
        
        return jsonify({
            'success': True,
            'message': 'Neo4j data refreshed successfully',
            'neo4j_nodes': graph_data.get('node_count', 0),
            'neo4j_edges': graph_data.get('edge_count', 0)
        })
            
    except Exception as e:
        logger.error(f"Failed to refresh graph data: {e}")
        return jsonify({
            'success': False,
            'error': 'Refresh failed',
            'details': str(e)
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

@graph_bp.route('/graph/model/<model_id>/refresh', methods=['POST'])
def refresh_single_model(model_id):
    """Refresh a single model in Neo4j"""
    try:
        if not neo4j_service.is_connected():
            return jsonify({
                'success': False,
                'error': 'Neo4j not connected'
            }), 503
        
        # Check if model exists in Neo4j
        model_data = neo4j_service.get_model_by_id(model_id)
        if not model_data:
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404
        
        # Model is already in Neo4j, so just return success
        return jsonify({
            'success': True,
            'message': f'Model {model_id} is already in Neo4j'
        })
            
    except Exception as e:
        logger.error(f"Failed to refresh single model {model_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to refresh model',
            'details': str(e)
        }), 500

@graph_bp.route('/graph/clear', methods=['POST'])
def clear_graph_data():
    """Clear all data from Neo4j (for testing/reset)"""
    try:
        if not neo4j_service.is_connected():
            return jsonify({
                'success': False,
                'error': 'Neo4j not connected'
            }), 503
        
        if neo4j_service.clear_all_data():
            return jsonify({
                'success': True,
                'message': 'All graph data cleared successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to clear graph data'
            }), 500
            
    except Exception as e:
        logger.error(f"Failed to clear graph data: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear graph data',
            'details': str(e)
        }), 500