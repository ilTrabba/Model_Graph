from flask import Blueprint, jsonify, request
from src.services.neo4j_service import neo4j_service
from src.services.sync_service import sync_service
import logging

logger = logging.getLogger(__name__)

graph_bp = Blueprint('graph', __name__)

@graph_bp.route('/graph/status', methods=['GET'])
def get_graph_status():
    """Get Neo4j connection and synchronization status"""
    try:
        status = sync_service.get_sync_status()
        return jsonify(status)
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
    """Manual synchronization endpoint"""
    try:
        if not neo4j_service.is_connected():
            return jsonify({
                'success': False,
                'error': 'Neo4j not connected',
                'message': 'Cannot sync without Neo4j connection'
            }), 503
        
        # Check if we should clear existing data first
        clear_existing = request.json.get('clear_existing', False) if request.json else False
        
        if clear_existing:
            neo4j_service.clear_all_data()
            logger.info("Cleared existing Neo4j data before sync")
        
        result = sync_service.sync_all_data()
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Failed to sync graph data: {e}")
        return jsonify({
            'success': False,
            'error': 'Sync failed',
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

@graph_bp.route('/graph/model/<model_id>/sync', methods=['POST'])
def sync_single_model(model_id):
    """Sync a single model to Neo4j"""
    try:
        if not neo4j_service.is_connected():
            return jsonify({
                'success': False,
                'error': 'Neo4j not connected'
            }), 503
        
        result = sync_service.sync_single_model(model_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400 if 'not found' in result.get('error', '').lower() else 500
            
    except Exception as e:
        logger.error(f"Failed to sync single model {model_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to sync model',
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