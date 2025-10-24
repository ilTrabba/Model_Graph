import os
import hashlib
import uuid
import logging

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from src.log_handler import logHandler
from src.services.neo4j_service import neo4j_service
from src.config import Config
from src.clustering.model_management import ModelManagementSystem

logger = logging.getLogger(__name__)
models_bp = Blueprint('models', __name__)
mgmt_system = ModelManagementSystem()

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
        
        result = mgmt_system.process_new_model(model_data)

        if result.get('status') != 'success':
            raise Exception(f"System failed (model status is not success): {result.get('error', 'Unknown error')}")
        
        family_id = result.get('family_id')
        
        # Create family relationship if family was assigned
        if family_id:
            neo4j_service.create_belongs_to_relationship(model_id, family_id)
        
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
        
        logHandler.error_handler(e, "upload_model", f"Model upload failed for {model_id}: {e}")
                
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