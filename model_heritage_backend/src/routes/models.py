import os
import hashlib
import uuid
import logging
import tempfile
import re
import json
from urllib.parse import urlparse

from safetensors import safe_open
from safetensors.torch import load_file, save_file
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from datetime import datetime, timezone
from src.log_handler import logHandler
from src.services.neo4j_service import neo4j_service
from src.config import Config
from src.clustering.model_management import ModelManagementSystem
from src.utils.normalization_system import normalize_safetensors_layers
from src.utils.normalization_system import save_layer_mapping_json

logger = logging.getLogger(__name__)
models_bp = Blueprint('models', __name__)
mgmt_system = ModelManagementSystem()

MODEL_FOLDER = Config.MODEL_FOLDER
README_FOLDER = 'readmes'
ALLOWED_EXTENSIONS = Config.ALLOWED_EXTENSIONS
ALLOWED_README_EXTENSIONS = {'md', 'txt'}
MAX_README_SIZE = 5 * 1024 * 1024  # 5MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_readme_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_README_EXTENSIONS

def validate_url(url):
    """Validate URL format using urllib.parse"""
    if not url:
        return True  # Empty URL is valid (optional field)
    try:
        result = urlparse(url)
        return all([result.scheme in ('http', 'https'), result.netloc])
    except Exception:
        return False

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
    try:
        """List all models with optional search"""
        search = request.args.get('search', '').strip()
        
        models_data = neo4j_service.get_all_models(search=search or None)
        models_data = sorted(models_data, key=lambda m: m.name.lower())

        models = []
        for m in models_data:
            models.append(m.to_dict())
        
        return jsonify({
            'models': models,
            'total': len(models)
        })
    except Exception as e:
        logHandler.error_handler(e, "list_models")
        return jsonify({'error': 'Failed to retrieve models', 'details': str(e)}), 500

@models_bp.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get model details by ID"""
    try:
        model = neo4j_service.get_model_by_id(model_id)
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Convert Model object to dictionary
        model_data = model.to_dict()
        
        # Get lineage information
        lineage = neo4j_service.get_model_lineage(model_id)
        model_data['lineage'] = lineage
        
        return jsonify(model_data)
        
    except Exception as e:
        logHandler.error_handler(e, "get_model_by_id")
        return jsonify({'error': 'Failed to retrieve model', 'details': str(e)}), 500

@models_bp.route('/models', methods=['POST'])
def upload_model():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Get metadata
    name = request.form.get('name', file.filename)
    description = request.form.get('description', '')
    model_id = str(uuid.uuid4())
    
    # Get new optional fields
    license_value = request.form.get('license', '')
    task_value = request.form.get('task', '')  # Comma-separated string from frontend
    dataset_url = request.form.get('dataset_url', '')
    is_foundation_model = request.form.get('is_foundation_model', 'false').lower() == 'true'
    
    # Validate dataset URL if provided
    if dataset_url and not validate_url(dataset_url):
        return jsonify({'error': 'Invalid dataset URL format'}), 400
    
    # Handle README file upload
    readme_uri = None
    readme_file = request.files.get('readme_file')
    if readme_file and readme_file.filename:
        if not allowed_readme_file(readme_file.filename):
            return jsonify({'error': 'Invalid README file type. Allowed: .md, .txt'}), 400
        
        # Check file size
        readme_file.seek(0, 2)  # Seek to end
        file_size = readme_file.tell()
        readme_file.seek(0)  # Reset to beginning
        
        if file_size > MAX_README_SIZE:
            return jsonify({'error': 'README file too large. Maximum size is 5MB'}), 400
        
        # Save README file
        os.makedirs(README_FOLDER, exist_ok=True)
        readme_filename = secure_filename(f"{model_id}_readme.md")
        readme_path = os.path.join(README_FOLDER, readme_filename)
        readme_file.save(readme_path)
        readme_uri = f"readmes/{readme_filename}"

    # Leggi il file in memoria
    file_bytes = file.read()

    # Salva temporaneamente per estrarre i metadata
    with tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors') as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    # Carica tensori e metadata
    tensors_dict = load_file(tmp_path)
    original_keys = list(tensors_dict.keys())
    with safe_open(tmp_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()

    # Rimuovi il file temporaneo
    os.unlink(tmp_path)

    try:
        norm_tensors_dict = normalize_safetensors_layers(tensors_dict)
    except ValueError as e:
        return jsonify({'error': f'Normalization failed: {str(e)}'}), 400

    normalized_keys = list(norm_tensors_dict.keys())

    # Salva il mapping JSON (fingerprint)
    num_layers = save_layer_mapping_json(
        original_keys=original_keys,
        normalized_keys=normalized_keys,
        model_id=model_id,
        original_filename=file.filename
    )

    # Save file
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    filename = secure_filename(f"{model_id}_{file.filename}")
    file_path = os.path.join(MODEL_FOLDER, filename)

    # Salva il file con tensori normalizzati e metadata originali
    save_file(norm_tensors_dict, file_path, metadata=metadata)

    try:
        # Calculate checksum
        checksum = calculate_file_checksum(file_path)
        
        # Check for duplicate
        existing = neo4j_service.get_model_by_checksum(checksum)
        if existing:
            os.remove(file_path)  # Clean up duplicate file
            if readme_uri and os.path.exists(readme_uri):
                os.remove(readme_uri)
            return jsonify({'error': 'Model already exists', 'existing_id': existing.get('id')}), 409
        
        # FIXME: Extract weight signature
        signature = extract_weight_signature_stub(file_path)
        
        # Parse task as list if comma-separated
        task_list = [t.strip() for t in task_value.split(',') if t.strip()] if task_value else []

        # Create model record
        model_data = {
            'id': model_id,
            'name': name,
            'description': description,
            'file_path': file_path,
            'checksum': checksum,
            'total_parameters': signature['total_parameters'],
            'layer_count': num_layers,
            'structural_hash': signature['structural_hash'],
            'status': 'processing',
            'weights_uri': 'weights/' + filename,
            'created_at': datetime.now(timezone.utc).isoformat(),
            # New optional fields
            'license': license_value if license_value else None,
            'task': task_list,
            'dataset_url': dataset_url if dataset_url else None,
            'dataset_url_verified': None if dataset_url else None,  # null = pending verification
            'readme_uri': readme_uri,
            'is_foundation_model': is_foundation_model
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
        final_model_data = neo4j_service.get_model_by_id(model_id).to_dict()

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

@models_bp.route('/models/<model_id>/readme', methods=['GET'])
def get_model_readme(model_id):
    """Get README content for a model"""
    try:
        model = neo4j_service.get_model_by_id(model_id)
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        readme_uri = model.readme_uri
        if not readme_uri:
            return jsonify({'error': 'No README available for this model'}), 404
        
        # Read README file
        readme_path = readme_uri  # readme_uri is already relative path like "readmes/xxx_readme.md"
        if not os.path.exists(readme_path):
            return jsonify({'error': 'README file not found'}), 404
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return jsonify({
            'model_id': model_id,
            'content': content,
            'readme_uri': readme_uri
        })
        
    except Exception as e:
        logHandler.error_handler(e, "get_model_readme")
        return jsonify({'error': 'Failed to retrieve README', 'details': str(e)}), 500

@models_bp.route('/families/<family_id>/models', methods=['GET'])
def get_family_models(family_id):
    """Get all models in a family"""

    # Verify family exists
    family_data = neo4j_service.get_family_by_id(family_id)
    if not family_data:
        return jsonify({'error': 'Family not found'}), 404
    
    # Get models in family
    models_data = neo4j_service.get_family_models(family_id, status='ok')
    
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