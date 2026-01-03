import os
import hashlib
import uuid
import logging
import tempfile
import json
import io
import torch
import gc
import shutil
import zipfile

from flask import send_file
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
from src.mother_algorithm.mother_utils import calc_ku, load_model_weights
from src.utils.normalization_system import normalize_safetensors_layers, save_layer_mapping_json
from src.utils.sharded_file_error import ShardedFileError


logger = logging.getLogger(__name__)
models_bp = Blueprint('models', __name__)
mgmt_system = ModelManagementSystem()
sharded_file_error = ShardedFileError()

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

from safetensors import safe_open
import hashlib
import os
from collections import Counter

def extract_weight_signature(file_path: str, num_layers: int) -> dict:
    """
    Estrae:
      - hidden_size (derivato dalle shapes)
      - numero di layers (fornito)
      - structural hash basato su (layers + hidden_size)
      - total_parameters (nuovo)

    Non usa i valori dei pesi.
    Funziona con modelli arbitrari.
    """

    # ---------------------------
    # 1. Carico shapes dal file
    # ---------------------------
    shapes = []
    dtypes = []
    total_parameters = 0

    with safe_open(file_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)

            shapes.append(shape)
            dtypes.append(dtype)

            # aggiungo al conteggio dei parametri
            num_params = 1
            for dim in shape:
                num_params *= dim
            total_parameters += num_params

    # --------------------------------------------
    # 2. Deduzione dell'HIDDEN SIZE in modo generico
    # --------------------------------------------
    dims = []

    for shape in shapes:
        if len(shape) == 2 and shape[0] == shape[1]:
            dims.append(shape[0])
            dims.append(shape[1])

    dims = [d for d in dims if d >= 128]

    if not dims:
        raise RuntimeError("Impossibile dedurre hidden_size dal file (nessuna matrice significativa trovata).")

    hidden_size = Counter(dims).most_common(1)[0][0]

    # ------------------------------
    # 3. Calcolo hash strutturale
    # ------------------------------
    base_string = f"{num_layers}_{hidden_size}"
    structural_hash = hashlib.md5(base_string.encode()).hexdigest()[:16]

    # ------------------------------
    # 4. Output
    # ------------------------------
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "structural_hash": structural_hash,
        "total_parameters": total_parameters
    }


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


# --- FUNZIONE HELPER (INVARIATA MA FONDAMENTALE) ---
def merge_and_convert_shards(source_dir, output_file_path):
    """
    Scansiona una cartella, trova tutti i file modello validi (. bin, .pt, .safetensors),
    li unisce in memoria e salva un UNICO file . safetensors.
    
    Supporta anche file senza estensione (validati con torch.load).
    """
    # Include file senza estensione ('')
    extensions = {'.bin', '.pt', '.pth', '.ckpt', '.safetensors', ''}
    shard_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file)
            ext_lower = ext.lower()
            
            # File con estensione valida
            if ext_lower in extensions and ext_lower != '':
                shard_files.append(file_path)
            
            # File SENZA estensione - valida con torch.load
            elif ext == '':
                try:
                    torch.load(file_path, map_location="cpu")
                    shard_files.append(file_path)
                    logger.info(f"[MERGE] Included extensionless file: {file}")
                except Exception: 
                    # Non è un file torch valido, ignora
                    logger.debug(f"[MERGE] Skipped non-torch file: {file}")
                    continue
    
    if not shard_files:
        raise FileNotFoundError("Nessun file modello valido (. bin, .safetensors, ecc) trovato.")

    # Sort intelligente (numerico per sharded, alfabetico altrimenti)
    shard_files = sharded_file_error.sort_sharded_files(shard_files)
    
    print(f"[MERGE] Trovati {len(shard_files)} file da unire.  Inizio processo...")

    combined_state_dict = {}
    seen_keys = {}  # Track which file each key came from (for duplicate detection)

    for i, shard_path in enumerate(shard_files):
        print(f"[MERGE] Elaborazione {i+1}/{len(shard_files)}: {os.path.basename(shard_path)}")
        _, ext = os.path.splitext(shard_path)
        ext = ext.lower()

        try:
            # Caricamento Ibrido
            if ext == '.safetensors':
                from safetensors. torch import load_file
                shard_data = load_file(shard_path)
            else:
                shard_data = torch.load(shard_path, map_location="cpu")
                if isinstance(shard_data, torch.nn.Module):
                    shard_data = shard_data. state_dict()
                elif isinstance(shard_data, dict):
                    if "state_dict" in shard_data:  
                        shard_data = shard_data["state_dict"]
                    elif "model" in shard_data:  
                        shard_data = shard_data["model"]

            # Unione con controllo duplicati
            for key, tensor in shard_data.items():
                if isinstance(tensor, torch.Tensor):
                    # Check for duplicate keys across shards
                    if key in combined_state_dict:
                        previous_file = seen_keys[key]
                        current_file = os.path.basename(shard_path)
                        raise ValueError(
                            f"Duplicate tensor key '{key}' found in multiple shards:\n"
                            f"  - First occurrence: {previous_file}\n"
                            f"  - Duplicate in:  {current_file}\n"
                            f"This indicates corrupted or incorrectly sharded model files."
                        )
                    
                    combined_state_dict[key] = tensor. clone().detach()
                    seen_keys[key] = os.path.basename(shard_path)
            
            del shard_data
            gc.collect()
            
        except Exception as e:
            raise Exception(f"Errore nel file {shard_path}: {e}")

    print(f"[MERGE] Salvataggio unico file in {output_file_path}...")
    from safetensors.torch import save_file
    save_file(combined_state_dict, output_file_path)
    
    print(f"[MERGE] Completato:  {len(combined_state_dict)} tensori unificati da {len(shard_files)} file")
    
    del combined_state_dict, seen_keys
    gc.collect()

@models_bp.route('/models', methods=['POST'])
def upload_model():
    """
    Upload and process model files (single or multiple sharded files).
    
    Supports:
    - Single . safetensors file
    - Multiple sharded . safetensors files (model-00001-of-00003.safetensors pattern)
    - Single .bin/. pt/.pth/. ckpt file (with smart extraction if archived)
    - .zip archives containing model files
    - Files without extensions (validated as PyTorch binaries)
    
    Returns:
        JSON response with model data and processing status
    """
    # ============================================================================
    # 1. VALIDAZIONE INPUT
    # ============================================================================
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # Leggiamo la LISTA di file (supporta multi-upload)
    files_list = request.files.getlist('file')
    
    if not files_list or files_list[0].filename == '':
        return jsonify({'error': 'No file selected'}), 400

    first_file = files_list[0]
    
    # ============================================================================
    # 2. ESTRAZIONE METADATA DAL FORM
    # ============================================================================
    name = request.form.get('name', first_file.filename)
    description = request.form.get('description', '')
    model_id = str(uuid.uuid4())
    license_value = request.form.get('license', '')
    task_value = request.form.get('task', '')
    dataset_url = request.form.get('dataset_url', '')
    is_foundation_model = request.form.get('is_foundation_model', 'false').lower() == 'true'
    
    # Validazione URL del dataset
    if dataset_url and not validate_url(dataset_url):
        return jsonify({'error': 'Invalid dataset URL format'}), 400

    # ============================================================================
    # 3. GESTIONE README FILE (opzionale)
    # ============================================================================
    readme_uri = None
    readme_file = request.files.get('readme_file')
    if readme_file and readme_file.filename:
        if not allowed_readme_file(readme_file.filename):
            return jsonify({'error': 'README must be . md or .txt file'}), 400
        
        # Check README size
        readme_file.seek(0, os.SEEK_END)
        readme_size = readme_file.tell()
        readme_file.seek(0)
        
        if readme_size > MAX_README_SIZE:
            return jsonify({'error': f'README file too large (max {MAX_README_SIZE // (1024*1024)}MB)'}), 400
        
        os.makedirs(README_FOLDER, exist_ok=True)
        readme_filename = secure_filename(f"{model_id}_readme.md")
        readme_path = os.path.join(README_FOLDER, readme_filename)
        readme_file.save(readme_path)
        readme_uri = f"readmes/{readme_filename}"

    # ============================================================================
    # 4. PREPARAZIONE PATHS TEMPORANEI
    # ============================================================================
    # File temporaneo per il safetensors finale (pre-normalizzazione)
    safetensors_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors')
    tmp_path = safetensors_tmp.name
    safetensors_tmp.close()

    # Directory temporanea per estrazione/elaborazione file
    extract_tmp_dir = tempfile.mkdtemp()

    try:
        loaded_object = None  # Per file binari singoli che richiedono conversione
        metadata = {}  # Metadata da safetensors (se presente)
        
        # ========================================================================
        # 5. SCENARIO A: MULTIPLI FILE CARICATI
        # ========================================================================
        if len(files_list) > 1:
            logger.info(f"[UPLOAD] Detected multiple file upload: {len(files_list)} files")
            
            # ----------------------------------------------------------------
            # 5A. VALIDAZIONE SHARDED SAFETENSORS (se applicabile)
            # ----------------------------------------------------------------
            if sharded_file_error.is_likely_sharded_upload(files_list):
                try:
                    validation_result = sharded_file_error.validate_sharded_safetensors(files_list)
                    logger.info(
                        f"✅ [UPLOAD] Sharded safetensors validated: {validation_result['base_name']} "
                        f"({validation_result['total_shards']} shards, sequential)"
                    )
                    
                    # Usa i file già ordinati dalla validazione (1, 2, 3, ...)
                    files_list = validation_result['sorted_files']
                    
                except ShardedFileError as e:
                    logger. error(f"❌ [UPLOAD] Sharded validation failed: {e}")
                    return jsonify({
                        'error': f'Invalid sharded files: {str(e)}'
                    }), 400
            else:
                logger.info(f"[UPLOAD] Multiple files detected but not sharded pattern, treating as multi-file upload")
            
            # ----------------------------------------------------------------
            # 5B. SALVATAGGIO FILE TEMPORANEI
            # ----------------------------------------------------------------
            for f in files_list:
                fname = secure_filename(f.filename)
                # Gestione path relativi (se browser invia strutture cartelle)
                if '/' in fname or '\\' in fname:
                    fname = os.path.basename(fname)
                if fname: 
                    file_temp_path = os.path.join(extract_tmp_dir, fname)
                    f.save(file_temp_path)
                    logger.debug(f"[UPLOAD] Saved temporary file: {fname}")
            
            # ----------------------------------------------------------------
            # 5C.  MERGE DI TUTTI I FILE
            # ----------------------------------------------------------------
            try:
                logger.info(f"[UPLOAD] Starting merge of {len(files_list)} files...")
                merge_and_convert_shards(extract_tmp_dir, tmp_path)
                logger.info(f"✅ [UPLOAD] Merge completed successfully")
            except ValueError as e:
                # Errore di duplicate keys tra shard
                logger.error(f"❌ [UPLOAD] Merge failed (duplicate keys): {e}")
                return jsonify({'error': str(e)}), 400
            except Exception as e:
                logger. error(f"❌ [UPLOAD] Merge failed: {e}")
                return jsonify({'error': f'Failed to merge files: {str(e)}'}), 500

        # ========================================================================
        # 6. SCENARIO B: SINGOLO FILE CARICATO
        # ========================================================================
        else:
            file = files_list[0]
            filename = secure_filename(file.filename)
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            logger.info(f"[UPLOAD] Single file upload: {filename} (extension: {ext})")

            # ----------------------------------------------------------------
            # 6A. CASO:  FILE ZIP
            # ----------------------------------------------------------------
            if ext == '.zip':
                logger. info(f"[UPLOAD] Processing ZIP file: {filename}")
                zip_path = os.path.join(extract_tmp_dir, "upload. zip")
                file.save(zip_path)
                
                try:
                    # Estrai ZIP
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_tmp_dir)
                    logger.info(f"[UPLOAD] ZIP extracted successfully")
                    
                    # Scansiona file estratti
                    found_models = sharded_file_error.scan_for_model_files(extract_tmp_dir, include_no_extension=True)
                    logger.info(f"[UPLOAD] Found {len(found_models)} model files in ZIP")
                    
                    if len(found_models) > 1:
                        # Multi-shard nello ZIP -> Merge
                        logger.info(f"[UPLOAD] Multiple files in ZIP, merging...")
                        merge_and_convert_shards(extract_tmp_dir, tmp_path)
                    elif len(found_models) == 1:
                        # Singolo file nello ZIP
                        single_file = found_models[0]
                        logger.info(f"[UPLOAD] Single file in ZIP: {os.path.basename(single_file)}")
                        
                        if single_file.lower().endswith('.safetensors'):
                            # Safetensors -> copia direttamente
                            shutil. move(single_file, tmp_path)
                            # Estrai metadata
                            with safe_open(tmp_path, framework="pt", device="cpu") as f:
                                metadata = f.metadata() or {}
                        else:
                            # Binario -> carica per conversione
                            loaded_object = torch.load(single_file, map_location="cpu")
                    else:
                        raise FileNotFoundError("ZIP archive is empty or contains no valid model files")
                        
                except zipfile.BadZipFile:
                    return jsonify({'error': 'Invalid ZIP file format'}), 400
                except Exception as e:
                    logger.error(f"❌ [UPLOAD] ZIP processing failed: {e}")
                    return jsonify({'error': f'ZIP extraction failed: {str(e)}'}), 400

            # ----------------------------------------------------------------
            # 6B. CASO: FILE BINARIO (. bin, .pt, .pth, .ckpt)
            # ----------------------------------------------------------------
            elif ext in ['.bin', '.pt', '.pth', '.ckpt']:
                logger.info(f"[UPLOAD] Processing binary file: {filename}")
                bin_path = os.path.join(extract_tmp_dir, filename)
                file.save(bin_path)
                
                # Tentativo di caricamento intelligente
                success, loaded_object = sharded_file_error.smart_load_bin(bin_path, extract_tmp_dir)
                
                if not success:
                    # Nessuna strategia ha funzionato
                    logger.error(f"❌ [UPLOAD] Unable to load binary file: {filename}")
                    return jsonify({
                        'error': f'Unable to load {filename}:  not a valid PyTorch binary or archive'
                    }), 400
                
                # Se loaded_object è None, significa che è stato estratto come archivio
                if loaded_object is None: 
                    logger.info(f"[UPLOAD] Binary file was extracted as archive")
                    # Scansiona directory estratta
                    found_models = sharded_file_error.scan_for_model_files(extract_tmp_dir, include_no_extension=True)
                    logger.info(f"[UPLOAD] Found {len(found_models)} files in extracted archive")
                    
                    if len(found_models) > 1:
                        # Multipli file nell'archivio -> Merge
                        logger.info(f"[UPLOAD] Multiple files in archive, merging...")
                        merge_and_convert_shards(extract_tmp_dir, tmp_path)
                    elif len(found_models) == 1:
                        # Singolo file nell'archivio -> Carica
                        logger.info(f"[UPLOAD] Single file in archive, loading...")
                        loaded_object = torch.load(found_models[0], map_location="cpu")
                    else: 
                        return jsonify({
                            'error': 'No valid model files found in binary archive'
                        }), 400
                else:
                    logger.info(f"[UPLOAD] Binary file loaded successfully as PyTorch object")

            # ----------------------------------------------------------------
            # 6C. CASO: FILE SAFETENSORS SINGOLO
            # ----------------------------------------------------------------
            elif ext == '.safetensors': 
                logger.info(f"[UPLOAD] Processing single safetensors file: {filename}")
                file.save(tmp_path)
                
                # Estrai metadata
                with safe_open(tmp_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata() or {}
                    logger.info(f"[UPLOAD] Safetensors metadata extracted: {len(metadata)} entries")
            
            # ----------------------------------------------------------------
            # 6D. CASO:  ESTENSIONE NON SUPPORTATA
            # ----------------------------------------------------------------
            else:
                logger.error(f"❌ [UPLOAD] Unsupported file extension: {ext}")
                return jsonify({'error':  f'Invalid file type: {ext}.  Supported:  .safetensors, .pt, .bin, .pth, .zip'}), 400

        # ========================================================================
        # 7. CONVERSIONE BINARIO -> SAFETENSORS (se necessario)
        # ========================================================================
        if loaded_object is not None: 
            logger.info("[UPLOAD] Converting PyTorch object to safetensors...")
            state_dict = {}
            
            # Estrai state_dict dall'oggetto caricato
            if isinstance(loaded_object, torch.nn.Module):
                logger.debug("[UPLOAD] Object is torch.nn.Module, extracting state_dict")
                state_dict = loaded_object.state_dict()
            elif isinstance(loaded_object, dict):
                # Controlla varie convenzioni di naming
                if "state_dict" in loaded_object: 
                    logger.debug("[UPLOAD] Found 'state_dict' key in dict")
                    state_dict = loaded_object["state_dict"]
                elif "model" in loaded_object: 
                    logger.debug("[UPLOAD] Found 'model' key in dict")
                    state_dict = loaded_object["model"]
                else:
                    logger.debug("[UPLOAD] Using dict directly as state_dict")
                    state_dict = loaded_object
            else:
                logger.error(f"❌ [UPLOAD] Unexpected object type: {type(loaded_object)}")
                return jsonify({'error': f'Unexpected model format: {type(loaded_object)}'}), 400
            
            # Pulisci state_dict (solo tensori, clonati e detached)
            clean_state_dict = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor):
                    clean_state_dict[k] = v.clone().detach()
                else:
                    logger.warning(f"[UPLOAD] Skipping non-tensor key: {k} (type: {type(v)})")
            
            logger.info(f"[UPLOAD] Cleaned state_dict: {len(clean_state_dict)} tensors")
            
            # Salva come safetensors
            save_file(clean_state_dict, tmp_path)
            logger.info(f"✅ [UPLOAD] Conversion to safetensors completed")
            
            # Cleanup memoria
            del loaded_object, state_dict, clean_state_dict
            gc.collect()

        # ========================================================================
        # 8. CARICAMENTO E NORMALIZZAZIONE
        # ========================================================================
        logger.info("[UPLOAD] Loading safetensors for normalization...")
        tensors_dict = load_file(tmp_path)
        original_keys = list(tensors_dict. keys())
        logger.info(f"[UPLOAD] Loaded {len(original_keys)} original layer names")

        # Rimuovi file temporaneo (abbiamo già caricato in memoria)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        # Normalizzazione layer names
        try:
            logger.info("[UPLOAD] Starting layer name normalization...")
            norm_tensors_dict = normalize_safetensors_layers(tensors_dict)
            normalized_keys = list(norm_tensors_dict.keys())
            logger.info(f"✅ [UPLOAD] Normalization completed:  {len(normalized_keys)} normalized layers")
            
            del tensors_dict
            gc.collect()
        except ValueError as e:
            logger.error(f"❌ [UPLOAD] Normalization failed: {e}")
            return jsonify({'error': f'Normalization failed: {str(e)}'}), 400

        # ========================================================================
        # 9. SALVATAGGIO FINGERPRINT (mapping original -> normalized)
        # ========================================================================
        logger.info("[UPLOAD] Saving layer mapping fingerprint...")
        num_layers = save_layer_mapping_json(
            original_keys, 
            normalized_keys, 
            model_id, 
            first_file.filename
        )
        logger.info(f"✅ [UPLOAD] Fingerprint saved: {num_layers} structural layers detected")

        # ========================================================================
        # 10. SALVATAGGIO FILE SAFETENSORS NORMALIZZATO FINALE
        # ========================================================================
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        
        # Nome file finale:  {model_id}_{model_name}. safetensors
        final_clean_name = os.path.splitext(secure_filename(name))[0] + ".safetensors"
        final_filename = secure_filename(f"{model_id}_{final_clean_name}")
        file_path = os.path.join(MODEL_FOLDER, final_filename)

        logger.info(f"[UPLOAD] Saving final normalized safetensors:  {final_filename}")
        save_file(norm_tensors_dict, file_path, metadata=metadata)
        logger.info(f"✅ [UPLOAD] Final file saved: {file_path}")
        
        del norm_tensors_dict
        gc.collect()

        # ========================================================================
        # 11. POST-PROCESSING:  Checksum, Signature, Neo4j
        # ========================================================================
        try:
            # Calcola checksum del file finale
            logger.info("[UPLOAD] Calculating file checksum...")
            checksum = calculate_file_checksum(file_path)
            
            # Verifica duplicati
            existing = neo4j_service.get_model_by_checksum(checksum)
            if existing:
                logger. warning(f"⚠️ [UPLOAD] Model already exists with same checksum: {existing. get('id')}")
                os.remove(file_path)
                if readme_uri and os.path.exists(readme_uri):
                    os.remove(readme_uri)
                return jsonify({
                    'error': 'Model already exists',
                    'existing_id': existing. get('id')
                }), 409
            
            # Estrai signature strutturale (hidden_size, structural_hash, total_parameters)
            logger.info("[UPLOAD] Extracting weight signature...")
            signature = extract_weight_signature(file_path, num_layers)
            logger.info(
                f"[UPLOAD] Signature:  {signature['total_parameters']} params, "
                f"hash={signature['structural_hash']}"
            )
            
            # Parse task list
            task_list = [t.strip() for t in task_value.split(',') if t.strip()] if task_value else []

            # Calcola kurtosis (per MoTHer algorithm)
            logger.info("[UPLOAD] Calculating kurtosis...")
            model_weights = load_model_weights(file_path=file_path)
            kurtosis = calc_ku(model_weights)
            logger.info(f"[UPLOAD] Kurtosis: {kurtosis}")
            del model_weights
            gc.collect()

            # Prepara dati per Neo4j
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
                'weights_uri': 'weights/' + final_filename,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'distance_from_parent': 0.0,
                'kurtosis': kurtosis,
                'license':  license_value or None,
                'task':  task_list,
                'dataset_url': dataset_url or None,
                'dataset_url_verified': None if dataset_url else None,
                'readme_uri': readme_uri,
                'is_foundation_model': is_foundation_model
            }

            # Salva in Neo4j
            logger.info("[UPLOAD] Saving model to Neo4j...")
            if not neo4j_service.create_model(model_data):
                raise Exception("Failed to save model to Neo4j")
            logger.info(f"✅ [UPLOAD] Model saved to Neo4j:  {model_id}")
            
            # Processing famiglia e genealogia (ModelManagementSystem)
            logger.info("[UPLOAD] Processing model family and genealogy...")
            result = mgmt_system.process_new_model(model_data)

            if result. get('status') != 'success':
                raise Exception(f"ModelManagementSystem failed: {result.get('error', 'Unknown error')}")
            
            # Crea relazione BELONGS_TO con famiglia
            family_id = result.get('family_id')
            if family_id: 
                logger.info(f"[UPLOAD] Creating BELONGS_TO relationship with family {family_id}")
                neo4j_service.create_belongs_to_relationship(model_id, family_id)
            
            # Recupera dati finali del modello (con relazioni)
            final_model_data = neo4j_service. get_model_by_id(model_id).to_dict()
            
            logger.info(f"✅✅✅ [UPLOAD] Model upload completed successfully:  {model_id}")

            return jsonify({
                'model_id': model_id,
                'status': 'ok',
                'message': 'Processed successfully',
                'model':  final_model_data
            }), 201
            
        except Exception as e:
            # Cleanup su errore in post-processing
            logger.error(f"❌ [UPLOAD] Post-processing failed: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

    # ============================================================================
    # 12. GESTIONE ERRORI
    # ============================================================================
    except ShardedFileError as e:
        # Errore specifico per validazione sharded files
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"❌ [UPLOAD] Sharded file validation error: {e}")
        return jsonify({
            'error': f'Sharded file validation failed: {str(e)}'
        }), 400
        
    except Exception as e: 
        # Errore generico
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logHandler.error_handler(e, "upload_model", f"Upload failed: {e}")
        return jsonify({
            'error': f'Processing failed: {str(e)}'
        }), 500
        
    finally:
        # ========================================================================
        # 13. CLEANUP FINALE (sempre eseguito)
        # ========================================================================
        if os.path.exists(extract_tmp_dir):
            try:
                shutil.rmtree(extract_tmp_dir)
                logger.debug(f"[UPLOAD] Cleaned up temporary directory: {extract_tmp_dir}")
            except Exception as e: 
                logger.warning(f"[UPLOAD] Failed to cleanup temp directory: {e}")

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
    
@models_bp.route('/models/<model_id>/download', methods=['GET'])
def download_model_weights(model_id):
    """Download model weights with original layer names restored"""
    try:
        # 1. Get model from Neo4j
        model = neo4j_service.get_model_by_id(model_id)
        if not model: 
            return jsonify({'error': 'Model not found'}), 404
        
        file_path = model.file_path
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # 2. Load fingerprint JSON to get mapping and original filename
        fingerprint_path = os.path.join('weights', 'fingerprints', f'{model_id}_mapping.json')
        if not os.path.exists(fingerprint_path):
            return jsonify({'error': 'Fingerprint file not found'}), 404
        
        with open(fingerprint_path, 'r') as f:
            fingerprint = json.load(f)
        
        mapping = fingerprint. get('mapping', {})
        original_filename = fingerprint.get('original_filename', f'{model_id}.safetensors')
        
        # 3. Load the stored safetensors file (normalized layer names)
        tensors_dict = load_file(file_path)
        
        # 4. Load original metadata
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
        
        # 5. Create new tensors dict with original layer names (key -> value mapping)
        restored_tensors = {}
        for normalized_name, tensor in tensors_dict.items():
            # mapping:  normalized_name (key) -> original_name (value)
            original_name = mapping.get(normalized_name, normalized_name)
            restored_tensors[original_name] = tensor
        
        # 6. Save to temporary file in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors') as tmp_file:
            tmp_path = tmp_file.name
        
        save_file(restored_tensors, tmp_path, metadata=metadata)
        
        # 7. Read the file into memory and delete temp file
        with open(tmp_path, 'rb') as f:
            file_data = f.read()
        os.unlink(tmp_path)
        
        # 8. Send file as download
        return send_file(
            io.BytesIO(file_data),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=original_filename
        )
        
    except Exception as e:
        logHandler.error_handler(e, "download_model_weights")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500