from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import hashlib
import uuid
from datetime import datetime, timezone
import json

from src.models.model import Model, Family
from src.services.neo4j_service import neo4j_service
from src.config import Config

models_bp = Blueprint('models', __name__)

MODEL_FOLDER = Config.MODEL_FOLDER
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_file_checksum(file_path):
    """Calculate SHA-256 checksum of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_weight_signature_stub(file_path):
    """Stub implementation for weight signature extraction"""
    # This is a simplified stub - in real implementation would analyze actual weights
    file_size = os.path.getsize(file_path)
    
    # Mock signature based on file size (for demo purposes)
    estimated_params = file_size // 4  # Assuming float32 weights
    estimated_layers = max(10, estimated_params // 1000000)  # Rough estimate
    
    signature = {
        'total_parameters': estimated_params,
        'layer_count': estimated_layers,
        'structural_hash': hashlib.md5(f"{estimated_params}_{estimated_layers}".encode()).hexdigest()[:16]
    }
    
    return signature

# cambiare nome della funzione
def assign_to_family_and_find_parent(model_data):
    """Use the new clustering system for family assignment and parent finding"""
    try:
        # Import clustering system (with graceful fallback if dependencies missing)
        from src.clustering.model_management import ModelManagementSystem
        
        # Initialize the management system
        mgmt_system = ModelManagementSystem()
        
        # Create a proxy object for compatibility with clustering system
        model_proxy = Model(**model_data)
        
        # Process the model through the complete pipeline
        result = mgmt_system.process_new_model(model_proxy)
        
        if result.get('status') == 'success':
            return result.get('family_id'), result.get('parent_id'), result.get('parent_confidence', 0.0)
        else:
            current_app.logger.error(f"Clustering system failed: {result.get('error')}")
            # Fallback to stub implementation
            return _fallback_family_assignment(model_data)
            
    except ImportError as e:
        current_app.logger.warning(f"Clustering system not available, using fallback: {e}")
        return _fallback_family_assignment(model_data)
    except Exception as e:
        current_app.logger.error(f"Clustering system failed, using fallback: {e}")
        return _fallback_family_assignment(model_data)

def _fallback_family_assignment(model_data):
    """Fallback family assignment using simple parameter-based rules"""
    param_count = model_data.get('total_parameters', 0)
    structural_hash = model_data.get('structural_hash', '')
    
    # Simple family assignment based on parameter count ranges
    if param_count < 1000000:  # < 1M params
        family_id = "small_models"
    elif param_count < 100000000:  # < 100M params
        family_id = "medium_models"
    else:
        family_id = "large_models"
    
    # Check if family exists, create if not
    families = neo4j_service.get_all_families()
    family_exists = any(f.get('id') == family_id for f in families)
    
    if not family_exists:
        family_data = {
            'id': family_id,
            'structural_pattern_hash': structural_hash,
            'member_count': 0,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        neo4j_service.create_family(family_data)
    
    # Update family member count (this is a simplified approach)
    family_models = neo4j_service.get_family_models(family_id)
    new_member_count = len(family_models) + 1
    
    family_update = {
        'id': family_id,
        'member_count': new_member_count,
        'updated_at': datetime.now(timezone.utc).isoformat()
    }
    neo4j_service.create_or_update_family(family_update)
    
    # Try to use tree-based approach for comprehensive relationship updates
    try:
        from src.clustering.model_management import ModelManagementSystem
        from src.clustering.tree_builder import MoTHerTreeBuilder
        
        # Initialize tree builder
        tree_builder = MoTHerTreeBuilder()
        
        # Get all family models
        family_models = neo4j_service.get_family_models(family_id)
        ok_models = [m for m in family_models if m.get('status') == 'ok']
        
        # Create model proxies for tree building
        model_proxies = []
        for model_dict in ok_models:
            try:
                model_proxy = Model(**model_dict)
                model_proxies.append(model_proxy)
            except Exception as e:
                current_app.logger.warning(f"Failed to create proxy for model {model_dict.get('id')}: {e}")
        
        # Add our new model to the list
        new_model_proxy = Model(**model_data)
        model_proxies.append(new_model_proxy)
        
        if len(model_proxies) >= 2:
            # Build complete family tree
            tree, confidence_scores = tree_builder.build_family_tree(model_proxies)
            
            if tree.number_of_nodes() > 0:
                # Update all model relationships based on tree
                parent_id = None
                parent_confidence = 0.0
                
                for model_proxy in model_proxies:
                    predecessors = list(tree.predecessors(model_proxy.id))
                    
                    if predecessors:
                        # Model has a parent
                        new_parent_id = predecessors[0]
                        new_confidence = confidence_scores.get(model_proxy.id, 0.0)
                        
                        # Update in Neo4j
                        neo4j_service.update_model(model_proxy.id, {
                            'parent_id': new_parent_id,
                            'confidence_score': new_confidence
                        })
                        
                        # If this is our new model, capture the parent info
                        if model_proxy.id == model_data.get('id'):
                            parent_id = new_parent_id
                            parent_confidence = new_confidence
                    else:
                        # Model is a root
                        neo4j_service.update_model(model_proxy.id, {
                            'parent_id': None,
                            'confidence_score': 0.0
                        })
                        
                        # If this is our new model, capture the root status
                        if model_proxy.id == model_data.get('id'):
                            parent_id = None
                            parent_confidence = 0.0
                
                current_app.logger.info(f"Successfully updated tree relationships for family {family_id}")
                return family_id, parent_id, parent_confidence
            else:
                current_app.logger.warning("Tree building produced empty tree, falling back to individual parent finding")
        else:
            current_app.logger.info("Insufficient models for tree building, using individual parent finding")
        
        # Fallback to individual parent finding
        from src.algorithms.mother_algorithm import find_model_parent_mother
        model_proxy = Model(**model_data)
        parent_id, confidence = find_model_parent_mother(model_proxy, family_id)
        return family_id, parent_id, confidence
        
    except Exception as e:
        current_app.logger.error(f"Tree-based approach failed: {e}")
        # Final fallback to parameter similarity
        parent_id, confidence = _fallback_parameter_similarity(model_data, family_id)
        return family_id, parent_id, confidence

def _fallback_parameter_similarity(model_data, family_id):
    """Fallback implementation using parameter count similarity"""
    # Simple heuristic: find model in same family with similar parameter count
    family_models = neo4j_service.get_family_models(family_id)
    
    # Filter only OK status models
    ok_models = [m for m in family_models if m.get('status') == 'ok']
    
    if not ok_models:
        return None, 0.0
    
    # Find closest by parameter count
    closest_model = None
    min_diff = float('inf')
    model_params = model_data.get('total_parameters', 0)
    
    for candidate in ok_models:
        if candidate.get('id') != model_data.get('id'):
            candidate_params = candidate.get('total_parameters', 0)
            diff = abs(candidate_params - model_params)
            if diff < min_diff:
                min_diff = diff
                closest_model = candidate
    
    if closest_model:
        # Mock confidence based on parameter similarity
        max_params = max(model_params, closest_model.get('total_parameters', 0))
        confidence = 1.0 - (min_diff / max_params) if max_params > 0 else 0.0
        confidence = max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
        
        return closest_model.get('id'), confidence
    
    return None, 0.0

@models_bp.route('/models', methods=['GET'])
def list_models():
    """List all models with optional search"""
    search = request.args.get('search', '').strip()
    
    models_data = neo4j_service.get_all_models(search=search or None)
    models_data.sort(key=lambda m: (m.get('name') or '').lower())
    
    return jsonify({
        'models': models_data,
        'total': len(models_data)
    })

@models_bp.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get specific model with lineage"""
    model_data = neo4j_service.get_model_by_id(model_id)
    if not model_data:
        return jsonify({'error': 'Model not found'}), 404
    
    # Get lineage
    lineage = neo4j_service.get_model_lineage(model_id)
    model_data['lineage'] = lineage
    
    return jsonify(model_data)

@models_bp.route('/models', methods=['POST'])
def upload_model():
    """Upload and process new model"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Get optional metadata
    name = request.form.get('name', file.filename)
    description = request.form.get('description', '')
    
    # Generate unique model ID
    model_id = str(uuid.uuid4())
    
    # Save file
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    filename = secure_filename(f"{model_id}_{file.filename}")
    file_path = os.path.join(MODEL_FOLDER, filename)
    file.save(file_path)
    
    try:
        # Calculate checksum
        checksum = calculate_file_checksum(file_path)
        
        # Check for duplicate
        existing = neo4j_service.get_model_by_checksum(checksum)
        if existing:
            os.remove(file_path)  # Clean up duplicate file
            return jsonify({'error': 'Model already exists', 'existing_id': existing.get('id')}), 409
        
        # Extract weight signature
        signature = extract_weight_signature_stub(file_path)

        # Create model record
        model_data = {
            'id': model_id,
            'name': name,
            'description': description,
            'file_path': file_path,
            'checksum': checksum,
            'total_parameters': signature['total_parameters'],
            'layer_count': signature['layer_count'],
            'structural_hash': signature['structural_hash'],
            'status': 'processing',
            'weights_uri': 'weights/' + filename,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Save to Neo4j
        if not neo4j_service.upsert_model(model_data):
            raise Exception("Failed to save model to Neo4j")
        
        # Use new clustering system for family assignment and parent finding
        family_id, parent_id, confidence = assign_to_family_and_find_parent(model_data)
        
        # Update model with results
        model_updates = {
            'family_id': family_id,
            'status': 'ok',
            'processed_at': datetime.now(timezone.utc).isoformat()
        }
        
        #if parent_id:
        #    model_updates['parent_id'] = parent_id
        #    model_updates['confidence_score'] = confidence
        
        # Update the model in Neo4j
        #if not neo4j_service.update_model(model_id, model_updates):
            #raise Exception("Failed to update model in Neo4j")
        
        # Create family relationship if family was assigned
        if family_id:
            neo4j_service.create_belongs_to_relationship(model_id, family_id)
        
        # Create parent-child relationship if parent was found
        #if parent_id:
            #neo4j_service.create_parent_child_relationship(parent_id, model_id, confidence)
        
        # Get final model data for response
        final_model_data = neo4j_service.get_model_by_id(model_id)
        
        return jsonify({
            'model_id': model_id,
            'status': 'ok',
            'message': 'Model uploaded and processed successfully',
            'model': final_model_data
        }), 201
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove model record if created in Neo4j
        # Note: In a production system, you might want more sophisticated cleanup
        current_app.logger.error(f"Model upload failed for {model_id}: {e}")
        
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@models_bp.route('/families', methods=['GET'])
def list_families():
    """List all families"""
    families_data = neo4j_service.get_all_families()
    
    return jsonify({
        'families': families_data,
        'total': len(families_data)
    })

@models_bp.route('/families/<family_id>/models', methods=['GET'])
def get_family_models(family_id):
    """Get all models in a family"""
    # Get family data
    families = neo4j_service.get_all_families()
    family_data = None
    for f in families:
        if f.get('id') == family_id:
            family_data = f
            break
    
    if not family_data:
        return jsonify({'error': 'Family not found'}), 404
    
    # Get models in family
    models_data = neo4j_service.get_family_models(family_id)
    
    return jsonify({
        'family': family_data,
        'models': models_data
    })

@models_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    stats = neo4j_service.get_stats()
    return jsonify(stats)

# New clustering API endpoints

@models_bp.route('/clustering/recluster', methods=['POST'])
def recluster_all():
    """Perform complete reclustering of all models"""
    try:
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.recluster_all_models()
        
        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'message': 'Reclustering completed successfully',
                'families_created': result.get('families_created', 0),
                'trees_rebuilt': result.get('trees_rebuilt', 0),
                'total_updates': result.get('total_relationship_updates', 0)
            })
        else:
            return jsonify({
                'status': 'error',
                'error': result.get('error', 'Unknown error during reclustering')
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Reclustering failed: {e}")
        return jsonify({'error': f'Reclustering failed: {str(e)}'}), 500

@models_bp.route('/families/<family_id>/tree/rebuild', methods=['POST'])
def rebuild_family_tree(family_id):
    """Rebuild genealogical tree for a specific family"""
    try:
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.rebuild_family_tree(family_id)
        
        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'message': f'Tree rebuilt for family {family_id}',
                'models_updated': result.get('models_updated', 0),
                'tree_valid': result.get('tree_valid', False),
                'tree_statistics': result.get('tree_statistics', {})
            })
        elif result.get('status') == 'skipped':
            return jsonify({
                'status': 'skipped',
                'message': result.get('reason', 'Tree rebuild skipped')
            })
        else:
            return jsonify({
                'status': 'error',
                'error': result.get('error', 'Unknown error during tree rebuild')
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Tree rebuild failed for family {family_id}: {e}")
        return jsonify({'error': f'Tree rebuild failed: {str(e)}'}), 500

@models_bp.route('/families/<family_id>/genealogy', methods=['GET'])
def get_family_genealogy(family_id):
    """Get complete genealogy information for a family"""
    try:
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.get_family_genealogy(family_id)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 404
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Failed to get genealogy for family {family_id}: {e}")
        return jsonify({'error': f'Failed to get genealogy: {str(e)}'}), 500

@models_bp.route('/models/<model_id>/lineage', methods=['GET'])
def get_model_lineage(model_id):
    """Get complete lineage information for a specific model"""
    try:
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.get_model_lineage(model_id)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 404
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Failed to get lineage for model {model_id}: {e}")
        return jsonify({'error': f'Failed to get lineage: {str(e)}'}), 500

@models_bp.route('/clustering/statistics', methods=['GET'])
def get_clustering_statistics():
    """Get comprehensive clustering system statistics"""
    try:
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.get_system_statistics()
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
        
    except Exception as e:
        current_app.logger.error(f"Failed to get clustering statistics: {e}")
        return jsonify({'error': f'Failed to get statistics: {str(e)}'}), 500

@models_bp.route('/models/<model_id>/reprocess', methods=['POST'])
def reprocess_model(model_id):
    """Reprocess a model through the clustering pipeline"""
    try:
        model_data = neo4j_service.get_model_by_id(model_id)
        if not model_data:
            return jsonify({'error': 'Model not found'}), 404
        
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        model_proxy = Model(**model_data)
        result = mgmt_system.process_new_model(model_proxy)
        
        if result.get('status') == 'success':
            return jsonify({
                'status': 'success',
                'message': f'Model {model_id} reprocessed successfully',
                'family_id': result.get('family_id'),
                'parent_id': result.get('parent_id'),
                'confidence': result.get('parent_confidence', 0.0)
            })
        else:
            return jsonify({
                'status': 'error',
                'error': result.get('error', 'Unknown error during reprocessing')
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Reprocessing failed for model {model_id}: {e}")
        return jsonify({'error': f'Reprocessing failed: {str(e)}'}), 500

