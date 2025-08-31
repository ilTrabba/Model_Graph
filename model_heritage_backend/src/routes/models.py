from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import hashlib
import uuid
from datetime import datetime
import json

from src.models.model import db, Model, Family
from src.services.sync_service import sync_service

models_bp = Blueprint('models', __name__)

UPLOAD_FOLDER = 'weights'
ALLOWED_EXTENSIONS = {'safetensors', 'pt', 'bin', 'pth', 'html'}

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

def assign_to_family_and_find_parent(model):
    """Use the new clustering system for family assignment and parent finding"""
    try:
        # Import clustering system (with graceful fallback if dependencies missing)
        from src.clustering.model_management import ModelManagementSystem
        
        # Initialize the management system
        mgmt_system = ModelManagementSystem()
        
        # Process the model through the complete pipeline
        result = mgmt_system.process_new_model(model)
        
        if result.get('status') == 'success':
            return result.get('family_id'), result.get('parent_id'), result.get('parent_confidence', 0.0)
        else:
            current_app.logger.error(f"Clustering system failed: {result.get('error')}")
            # Fallback to stub implementation
            return _fallback_family_assignment(model)
            
    except ImportError as e:
        current_app.logger.warning(f"Clustering system not available, using fallback: {e}")
        return _fallback_family_assignment(model)
    except Exception as e:
        current_app.logger.error(f"Clustering system failed, using fallback: {e}")
        return _fallback_family_assignment(model)

def _fallback_family_assignment(model):
    """Fallback family assignment using simple parameter-based rules"""
    signature = {
        'total_parameters': model.total_parameters,
        'structural_hash': model.structural_hash
    }
    
    # Simple family assignment based on parameter count ranges
    param_count = signature['total_parameters']
    
    if param_count < 1000000:  # < 1M params
        family_id = "small_models"
    elif param_count < 100000000:  # < 100M params
        family_id = "medium_models"
    else:
        family_id = "large_models"
    
    # Check if family exists, create if not
    family = Family.query.filter_by(id=family_id).first()
    if not family:
        family = Family(
            id=family_id,
            structural_pattern_hash=signature['structural_hash'],
            member_count=0
        )
        db.session.add(family)
    
    family.member_count += 1
    family.updated_at = datetime.utcnow()
    
    # Try to find parent using original MoTHer algorithm
    try:
        from src.algorithms.mother_algorithm import find_model_parent_mother
        parent_id, confidence = find_model_parent_mother(model, family_id)
        return family_id, parent_id, confidence
    except Exception as e:
        current_app.logger.error(f"MoTHer algorithm failed: {e}")
        parent_id, confidence = _fallback_parameter_similarity(model, family_id)
        return family_id, parent_id, confidence

def _fallback_parameter_similarity(model, family_id):
    """Fallback implementation using parameter count similarity"""
    # Simple heuristic: find model in same family with similar parameter count
    family_models = Model.query.filter_by(family_id=family_id, status='ok').all()
    
    if not family_models:
        return None, 0.0
    
    # Find closest by parameter count
    closest_model = None
    min_diff = float('inf')
    
    for candidate in family_models:
        if candidate.id != model.id:
            diff = abs(candidate.total_parameters - model.total_parameters)
            if diff < min_diff:
                min_diff = diff
                closest_model = candidate
    
    if closest_model:
        # Mock confidence based on parameter similarity
        max_params = max(model.total_parameters, closest_model.total_parameters)
        confidence = 1.0 - (min_diff / max_params) if max_params > 0 else 0.0
        confidence = max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
        
        return closest_model.id, confidence
    
    return None, 0.0

@models_bp.route('/models', methods=['GET'])
def list_models():
    """List all models with optional search"""
    search = request.args.get('search', '').strip()
    
    query = Model.query
    if search:
        query = query.filter(Model.name.contains(search))
    
    models = query.order_by(Model.name).all()
    
    return jsonify({
        'models': [model.to_dict() for model in models],
        'total': len(models)
    })

@models_bp.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get specific model with lineage"""
    model = Model.query.get_or_404(model_id)
    
    model_data = model.to_dict()
    model_data['lineage'] = model.get_lineage()
    
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
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filename = secure_filename(f"{model_id}_{file.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    try:
        # Calculate checksum
        checksum = calculate_file_checksum(file_path)
        
        # Check for duplicate
        existing = Model.query.filter_by(checksum=checksum).first()
        if existing:
            os.remove(file_path)  # Clean up duplicate file
            return jsonify({'error': 'Model already exists', 'existing_id': existing.id}), 409
        
        # Extract weight signature
        signature = extract_weight_signature_stub(file_path)

        # Create model record
        model = Model(
            id=model_id,
            name=name,
            description=description,
            file_path=file_path,
            checksum=checksum,
            total_parameters=signature['total_parameters'],
            layer_count=signature['layer_count'],
            structural_hash=signature['structural_hash'],
            status='processing',
            weights_uri='weights/'+filename
        )
        
        db.session.add(model)
        db.session.commit()
        
        # Use new clustering system for family assignment and parent finding
        family_id, parent_id, confidence = assign_to_family_and_find_parent(model)
        
        # Update model with results
        model.family_id = family_id
        if parent_id:
            model.parent_id = parent_id
            model.confidence_score = confidence
        
        # Mark as processed (clustering system may have already done this)
        if model.status != 'ok':
            model.status = 'ok'
            model.processed_at = datetime.utcnow()
        
        db.session.commit()
        
        # Sync to Neo4j if connected
        try:
            sync_service.sync_single_model(model_id)
        except Exception as e:
            # Log the error but don't fail the upload
            print(f"Failed to sync model to Neo4j: {e}")
        
        return jsonify({
            'model_id': model_id,
            'status': 'ok',
            'message': 'Model uploaded and processed successfully',
            'model': model.to_dict()
        }), 201
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Remove model record if created
        model = Model.query.get(model_id)
        if model:
            db.session.delete(model)
            db.session.commit()
        
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@models_bp.route('/families', methods=['GET'])
def list_families():
    """List all families"""
    families = Family.query.order_by(Family.created_at.desc()).all()
    
    return jsonify({
        'families': [family.to_dict() for family in families],
        'total': len(families)
    })

@models_bp.route('/families/<family_id>/models', methods=['GET'])
def get_family_models(family_id):
    """Get all models in a family"""
    family = Family.query.get_or_404(family_id)
    models = Model.query.filter_by(family_id=family_id).order_by(Model.created_at).all()
    
    return jsonify({
        'family': family.to_dict(),
        'models': [model.to_dict() for model in models]
    })

@models_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    total_models = Model.query.count()
    total_families = Family.query.count()
    processing_models = Model.query.filter_by(status='processing').count()
    
    return jsonify({
        'total_models': total_models,
        'total_families': total_families,
        'processing_models': processing_models
    })

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
        model = Model.query.get_or_404(model_id)
        
        from src.clustering.model_management import ModelManagementSystem
        
        mgmt_system = ModelManagementSystem()
        result = mgmt_system.process_new_model(model)
        
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

