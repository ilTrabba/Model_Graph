"""
FamilyClusteringSystem

This module manages model families and automatic clustering based on weight similarities.
It handles family assignment, centroid calculation, and family management operations.
"""

import logging
import numpy as np
import torch
import uuid
import re
import safetensors.torch
import os

from src.log_handler import logHandler
from typing import Union
from src.mother_algorithm.mother_utils import load_model_weights
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from enum import Enum
from datetime import datetime
from safetensors import safe_open
from ..db_entities.entity import Model, Family
from src.services.neo4j_service import neo4j_service
from .distance_calculator import ModelDistanceCalculator


logger = logging.getLogger(__name__)

class ClusteringMethod(Enum):
    """Available clustering methods"""
    DBSCAN = "dbscan"
    KMEANS = "kmeans" 
    THRESHOLD = "threshold"
    AUTO = "auto"

class FamilyClusteringSystem:
    """
    Manage model families and automatic clustering based on weight similarities.
    
    Features:
    - Assign new models to existing families or create new ones
    - Maintain family centroids and statistics
    - Use configurable distance thresholds and clustering algorithms
    - Integrate with existing Family database model
    """
    
    def __init__(self,
                 distance_calculator: Optional[ModelDistanceCalculator] = None,
                 family_threshold: float = 0.2,
                 min_family_size: int = 2,
                 clustering_method: ClusteringMethod = ClusteringMethod.THRESHOLD):
        """
        Initialize the family clustering system.
        
        Args:
            distance_calculator: Calculator for model distances
            family_threshold: Distance threshold for family assignment
            min_family_size: Minimum number of models to form a family
            clustering_method: Method to use for clustering
        """
        self.distance_calculator = distance_calculator or ModelDistanceCalculator()
        self.family_threshold = family_threshold
        self.min_family_size = min_family_size
        self.clustering_method = clustering_method
        
        # Centroid storage configuration - use same approach as UPLOAD_FOLDER
        self.centroids_dir = os.path.join('weights', 'centroids')
        os.makedirs(self.centroids_dir, exist_ok=True)
    
    def get_centroid_file_path(self, family_id: str) -> str:
        """Get the file path for a family's centroid file using SafeTensors format."""
        return os.path.join(self.centroids_dir, f"{family_id}.safetensors")
    
    def save_family_centroid(self, family_id: str, centroid: Dict[str, Any]) -> bool:
        """
        Save family centroid to SafeTensors file with enhanced metadata.
        
        Args:
            family_id: ID of the family
            centroid: Dictionary containing centroid weights
            
        Returns:
            True if successfully saved, False otherwise
        """
        logger.info(f"=== ATTEMPTING TO SAVE CENTROID FOR FAMILY {family_id} ===")
        logger.info(f"Absolute centroids path: {os.path.abspath(self.centroids_dir)}")
        
        try:
            if not centroid:
                logger.warning(f"Cannot save empty centroid for family {family_id}")
                return False
            
            centroid_path = self.get_centroid_file_path(family_id)
            logger.info(f"Target path: {centroid_path}")
            logger.info(f"Absolute target path: {os.path.abspath(centroid_path)}")
            
            # Ensure centroids directory exists
            os.makedirs(os.path.dirname(centroid_path), exist_ok=True)
            logger.info(f"✅ Directory created/verified: {os.path.dirname(centroid_path)}")
            
            # Prepare metadata for SafeTensors
            metadata = {
                'family_id': family_id,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'version': '1.0',
                'layer_count': str(len(centroid)),
                'layer_keys': str(list(centroid.keys())),
                'distance_metric': 'cosine',
                'format': 'safetensors'
            }
            
            # Save using SafeTensors format
            safetensors.torch.save_file(centroid, centroid_path, metadata=metadata)
            logger.info(f"✅ SUCCESSFULLY SAVED centroid to {centroid_path} (SafeTensors format)")
            
            # Verify that the file was actually created
            if os.path.exists(centroid_path):
                file_size = os.path.getsize(centroid_path)
                logger.info(f"✅ File verified: {centroid_path} ({file_size} bytes)")
            else:
                logger.error(f"❌ File NOT found after save: {centroid_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving centroid for family {family_id}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def load_family_centroid(self, family_id: str) -> Optional[Dict[str, Any]]:
        """
        Load family centroid from SafeTensors file.
        
        Args:
            family_id: ID of the family
            
        Returns:
            Dictionary containing centroid weights or None if not found
        """
        try:
            centroid_path = self.get_centroid_file_path(family_id)
            
            if not os.path.exists(centroid_path):
                logger.debug(f"No centroid file found for family {family_id} at {centroid_path}")
                return None
            
            # Check if it's a SafeTensors file
            if centroid_path.endswith('.safetensors'):
                try:
                    # Load SafeTensors file
                    centroid_data = safetensors.torch.load_file(centroid_path)
                    logger.info(f"Loaded SafeTensors centroid for family {family_id} from {centroid_path}")
                    return centroid_data
                except Exception as st_error:
                    logger.warning(f"Failed to load as SafeTensors, trying legacy format: {st_error}")
            
            # Fallback to legacy PyTorch format for backward compatibility
            centroid_data = torch.load(centroid_path, map_location='cpu')
            
            # Handle both new format (with metadata) and old format (direct weights)
            if isinstance(centroid_data, dict) and 'weights' in centroid_data:
                logger.info(f"Loaded legacy centroid for family {family_id} from {centroid_path}")
                return centroid_data['weights']
            else:
                # Assume it's the old format - direct weights
                logger.info(f"Loaded legacy centroid for family {family_id} from {centroid_path}")
                return centroid_data
                
        except Exception as e:
            logger.error(f"Error loading centroid for family {family_id}: {e}")
            return None
    
    def centroid_to_embedding(self, centroid: Dict[str, Any]) -> List[float]:
        try:
            if not centroid:
                return [0.0]
            
            # Create a compact signature instead of full weights
            signature = []
            
            for param_name, tensor in centroid.items():
                if isinstance(tensor, torch.Tensor):
                    # Use statistical measures instead of raw weights
                    tensor_np = tensor.detach().cpu().numpy()
                    stats = [
                        float(tensor_np.mean()),
                        float(tensor_np.std()),
                        float(tensor_np.min()),
                        float(tensor_np.max()),
                        float(tensor_np.shape[0] if len(tensor_np.shape) > 0 else 1)
                    ]
                    signature.extend(stats)
            
            return signature[:100]  # Limit to 100 values max
        except Exception as e:
            logger.error(f"Error converting centroid to embedding: {e}")
            return [0.0]
        
    def assign_model_to_family(self, 
                             model: Model,
                             model_weights: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """
        Assign a model to an existing family or create a new one.
        
        Args:
            model: Model object to assign
            model_weights: Pre-loaded model weights (optional, will load if None)
            
        Returns:
            Tuple of (family_id, confidence_score)
        """
        try:
            # Load model weights if not provided
            if model_weights is None:
                model_weights = load_model_weights(model.file_path)
                if model_weights is None:
                    raise Exception("Model weights could not be loaded")

            # Get all existing models with the same structural pattern
            candidate_centroids = neo4j_service.get_all_centroids()

            # If there aren't candidate centroids create a new one with the model weights for the centroid
            if not candidate_centroids:
                family_id = self.create_new_family(model, model_weights)
                confidence = 1.0
            else:
                
                # Calculate distances to family centroids
                best_family_id, confidence = self.find_best_family_match(
                    model_weights, candidate_centroids
                )
                
                if confidence >= 0.2:
                    family_id = best_family_id
                else:
                    # Create new family with the model weights
                    family_id = self.create_new_family(model, model_weights)
                    confidence = 1.0
            
            # Update model with family assignment
            neo4j_service.update_model(model.id, {'family_id': family_id})

            return family_id, confidence
        except Exception as e:
            logHandler.error_handler(e, "assign_model_to_family")
    
    def calculate_family_centroid(self, family_id: str, family_models: List[Model]) -> Optional[Dict[str, Any]]:
        """
        Calculate the centroid (average weights) for a family.
        
        Args:
            family_id: ID of the family
            
        Returns:
            Dictionary representing the family centroid weights
        """
        try:
            if not family_models:
                return None
            
            # Load weights for all family models
            family_weights = []
            for model in family_models:
                weights = load_model_weights(model.file_path)
                if weights is not None:
                    family_weights.append(weights)
            
            if not family_weights:
                return None
            
            # Calculate centroid by averaging weights
            centroid = self.calculate_weights_centroid(family_weights)
            
            # Save centroid to file for incremental updates
            if centroid:
                self.save_family_centroid(family_id, centroid)
                
                # Update Neo4j centroid node with actual embedding and metadata
                try:
                    if neo4j_service.is_connected():

                        # Update embedding in FamilyCentroid (for backward compatibility)
                        embedding = self.centroid_to_embedding(centroid)
                        
                        # Update enhanced Centroid node with metadata
                        self.update_centroid_metadata(neo4j_service, family_id, centroid, len(family_weights))
                        
                        neo4j_service.create_has_centroid_relationship(family_id)
                except Exception as neo4j_error:
                    logger.warning(f"Failed to update Neo4j centroid for family {family_id}: {neo4j_error}")
            
            logger.info(f"Calculated centroid for family {family_id} with {len(family_weights)} models")
            return centroid
            
        except Exception as e:
            logHandler.error_handler(f"Error calculating family centroid for {family_id}: {e}", "calculate_family_centroid")
            return None
    
    def find_candidate_families(self, model: Model) -> List[Family]:
        """
        Find candidate families for a model based on structural similarity.
        """
        try:
            # For now, consider all families as candidates
            # In future versions, could filter by structural_hash or other criteria
            return neo4j_service.get_all_families()
            
        except Exception as e:
            logHandler.error_handler(e, "find_candidate_families")
    
    def find_best_family_match(self,
                               model_weights: Dict[str, Any],
                               candidate_centroids: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Find the best family match for a model.
        
        Returns:
            Tuple of (best_family_id, confidence_score)
        """
        try:
            best_family_id = None
            best_distance = float('inf')
            
            for centroid in candidate_centroids:
                centroid_weights = None

                # prendere centroide già esistente della famiglia corrente con un if
                centroid_path = centroid["path"]
                family_id = centroid["family_id"]

                if os.path.exists(centroid_path):
                    try:
                        with safe_open(centroid_path, framework="pt") as f:

                            # Create a dictionary to hold the tensors
                            centroid_data = {}
                            for key in f.keys():

                                # Load each tensor and add it to the dictionary
                                # .clone() is often good practice to ensure you have an independent copy
                                centroid_data[key] = f.get_tensor(key).clone()

                        # Now, centroid_data is a dictionary with the loaded tensors
                        centroid_weights = centroid_data

                    except Exception as e:
                        logHandler.error_handler(e, "find_best_family_match", "Error loading safetensors file")
                else:
                    logHandler.error_handler(f"Centroid file does not exist for family {family_id}", "find_best_family_match")

                if centroid_weights is None:
                    continue
                
                distance = self.distance_calculator.calculate_distance(
                    model_weights, centroid_weights
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_family_id = family_id

            
            if best_family_id is None:
                return "", 0.0
        
            # Convert distance to confidence score
            print(f"{self.family_threshold}")

            value1 = best_distance / 4.2
            value2 = 1 - value1
            confidence = max (0.0, value2)
            confidence = min (1.0, confidence)
            
            return best_family_id, confidence
            
        except Exception as e:
            logHandler.error_handler(e, "find_best_family_match")
    
    def create_new_family(self, model: Model, model_weights: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new family for the model.
        
        Args:
            model: Model object to create family for
            model_weights: Pre-loaded model weights to use as initial centroid
        """
        try:
            family_id = f"family_{str(uuid.uuid4())[:8]}"
            
            # Create family in Neo4j
            family_data = {
                'id': family_id,
                'structural_pattern_hash': model.structural_hash,
                'member_count': 1,
                'avg_intra_distance': 0.0,
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
            neo4j_service.create_family(family_data)
            
            if model_weights:
                logger.info(f"Creating initial centroid for family {family_id} from model {model.id}")
                
                initial_centroid = {}
                for param_name, tensor in model_weights.items():
                    if isinstance(tensor, torch.Tensor):

                        # Create a copy of the tensor for the centroid
                        initial_centroid[param_name] = tensor.detach().clone()
                
                # Save the initial centroid
                if initial_centroid:
                    self.save_family_centroid(family_id, initial_centroid)
                    
                    # Update Neo4j centroid node
                    try:
                        if neo4j_service.is_connected():
                            
                            # Update enhanced Centroid node with metadata
                            self.update_centroid_metadata(neo4j_service, family_id, initial_centroid, 1)
                            
                            neo4j_service.create_has_centroid_relationship(family_id)
                    except Exception as neo4j_error:
                        logHandler.error_handler(neo4j_error, "create_new_family", "Failed to update Neo4j centroid for new family")
                    
                    logger.info(f"✅ Initial centroid created and saved for family {family_id}")
                else:
                    logHandler.error_handler(f"No valid weights found to create centroid for family {family_id}", "create_new_family")
            else:
                logHandler.error_handler(f"No model weights provided, family {family_id} could not be created", "create_new_family")

            logger.info(f"✅ Created new family {family_id} for model {model.id}")
            return family_id
            
        except Exception as e:
            logHandler.error_handler(e, "create_new_family", "Generic error creating new family")

    def normalize_safetensors_layers(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizza i nomi dei layer di un singolo file safetensors.

        Gestisce molteplici architetture Hugging Face (ViT, BERT, GPT, LLaMA, T5, 
        Whisper, CLIP, Swin, ecc.) per garantire compatibilità cross-framework.

        Operazioni applicate:
        1. LayerNorm TensorFlow → PyTorch (.beta → .bias, .gamma → .weight)
        2. Estrazione core strutturale (layer.X, block.X, h.X, etc.)
        3. Rimozione prefissi di wrapping del modello
        4. Normalizzazione abbreviazioni comuni
        5. Pulizia finale
        
        Args:
            weights: Dizionario {layer_name: tensor}

        Returns:
            Dizionario normalizzato {normalized_layer_name: tensor}
            
        Raises:
            Nessuna eccezione; logga warning per collisioni
        """
        if not weights:
            logger.warning("Input weights vuoto")
            return {}

        normalized_weights = {}

        # =========================================================================
        # PATTERN STRUTTURALI (ordinati per priorità/specificità)
        # =========================================================================
        structural_patterns = [
            # Encoder-Decoder specifici (più specifici prima)
            r'(encoder\.layer\.\d+)',
            r'(decoder\.layer\.\d+)',
            r'(encoder\.blocks\.\d+)',
            r'(decoder\.blocks\.\d+)',
            r'(encoder_block\.\d+)',
            r'(decoder_block\.\d+)',
            
            # Transformer generici
            r'(transformer\.h\.\d+)',
            r'(transformer\.layers\.\d+)',
            r'(transformer\.blocks\.\d+)',
            r'(transformer_block\.\d+)',
            
            # Layer/Block generici (LLaMA, GPT, Mistral, etc.)
            r'(model\.layers\.\d+)',
            r'(model\.decoder\.layers\.\d+)',
            r'(gpt_neox\.layers\.\d+)',
            r'(layers\.\d+)',
            r'(blocks\.\d+)',
            r'(layer\.\d+)',
            r'(block\.\d+)',
            r'(h\.\d+)',
            
            # Vision Models (CLIP, ViT, Swin, SAM, etc.)
            r'(vision_model\.encoder\.layers\.\d+)',
            r'(visual\.encoder\.blocks\.\d+)',
            r'(vision_tower\.blocks\.\d+)',
            r'(backbone\.stages\.\d+)',
            
            # Altri pattern
            r'(modules\.\d+)',
            r'(block_list\.\d+)',
        ]

        # =========================================================================
        # PREFISSI DA RIMUOVERE (ordinati alfabeticamente per manutenibilità)
        # =========================================================================
        known_prefixes = [
            # Generici
            'backbone.', 'base_model.', 'decoder.', 'discriminator.', 'encoder.',
            'generator.', 'head.', 'model.', 'proj.', 'student.', 'teacher.', 
            'transformer.', 'unet.', 'vae.',
            
            # Text Models
            'bart.', 'bert.', 'bloom.', 'deberta.', 'distilbert.', 'falcon.', 
            'gpt_neox.', 'gptj.', 'llama.', 'marian.', 'mistral.', 'mt5.', 
            'opt.', 'roberta.', 't5.',
            
            # Vision Models
            'beit.', 'clip.', 'clip_model.', 'clip_vision_model.', 'convnext.', 
            'dino.', 'dinov2.', 'open_clip.', 'sam.', 'swin.', 'vit.',
            
            # Multimodal
            'clip.', 'image_encoder.', 'speech_encoder_decoder.', 'text_encoder.', 
            'text_model.', 'vision_model.', 'vision_tower.', 'visual.', 'whisper.',
        ]

        # =========================================================================
        # ABBREVIAZIONI DA NORMALIZZARE
        # =========================================================================
        abbreviation_map = {
            '.ln_f.': '.layernorm_final.',
            '.ln_1.': '.layernorm_before.',
            '.ln_2.': '.layernorm_after.',
            '.ln.': '.layernorm.',
            '.attn.': '.attention.',
            '.mlp.': '.feedforward.',
            '.c_attn.': '.attention.combined.',  
            '.c_proj.': '.attention.projection.',
            '.c_fc.': '.feedforward.fc.',
        }

        # =========================================================================
        # NORMALIZZAZIONE
        # =========================================================================
        collision_count = 0
        
        for original_name, tensor in weights.items():
            normalized_name = original_name

            # Step 1: TensorFlow → PyTorch LayerNorm
            normalized_name = normalized_name.replace('.beta', '.bias')
            normalized_name = normalized_name.replace('.gamma', '.weight')

            # Step 2: Cerca pattern strutturali (dal più specifico al generico)
            core_name_found = False
            for pattern in structural_patterns:
                match = re.search(pattern, normalized_name)
                if match:
                    start_idx = match.start()
                    normalized_name = normalized_name[start_idx:]
                    core_name_found = True
                    break

            # Step 3: Se non trovato pattern, rimuovi prefissi noti
            if not core_name_found:
                for prefix in known_prefixes:
                    if normalized_name.startswith(prefix):
                        normalized_name = normalized_name[len(prefix):]
                        break

            # Step 4: Normalizza abbreviazioni comuni
            for abbrev, expanded in abbreviation_map.items():
                normalized_name = normalized_name.replace(abbrev, expanded)

            # Step 5: Pulizia finale - rimuovi prefissi residui
            # (potrebbe esserci dopo Step 2 se pattern non era all'inizio)
            normalized_name = re.sub(
                r'^(?:model|base_model|transformer)\.',
                '',
                normalized_name
            )

            # Step 6: Gestisci collisioni
            if normalized_name in normalized_weights:
                collision_count += 1
                logger.warning(
                    f"Collisione normalizzazione #{collision_count}: "
                    f"'{original_name}' → '{normalized_name}' (già esistente). "
                    f"Layer scartato."
                )
                continue

            # Step 7: Aggiungi al risultato
            normalized_weights[normalized_name] = tensor

        # =========================================================================
        # LOGGING FINALE
        # =========================================================================
        original_count = len(weights)
        normalized_count = len(normalized_weights)
        
        if collision_count > 0:
            logger.warning(
                f"Normalizzazione completata con {collision_count} collisioni: "
                f"{original_count} layer originali → {normalized_count} layer normalizzati "
                f"({original_count - normalized_count} scartati)"
            )
        else:
            logger.info(
                f"Normalizzazione completata: "
                f"{original_count} layer originali → {normalized_count} layer normalizzati"
            )

        return normalized_weights

    def calculate_weights_centroid(self, weights_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcola il centroide (media) dei pesi di modelli della stessa famiglia.
        
        Questa funzione:
        1. Trova i layer comuni presenti in TUTTI i modelli della famiglia relativa
        2. Verifica che abbiano shape identico
        3. Calcola la media elemento per elemento
        
        IMPORTANTE: Assume che i dizionari in weights_list siano già stati 
        normalizzati con normalize_safetensors_layers().
        
        Args:
            weights_list: Lista di dizionari normalizzati [{layer_name: tensor}, ...]
            
        Returns:
            Dizionario centroide {layer_name: averaged_tensor} o {} se nessun layer comune valido
        """
        try:
            # Step 1: Validazione input
            if not weights_list:
                logger.warning("Lista pesi vuota, impossibile calcolare centroide")
                return {}
            
            logger.info(f"Calcolo centroide per {len(weights_list)} modelli")
            
            normalized_weights = [self.normalize_safetensors_layers(w) for w in weights_list]
            
            # Step 2: Trova layer comuni (presenti in TUTTI i modelli)
            # Crea set di layer names per ogni modello
            layer_sets = [set(weights.keys()) for weights in normalized_weights]
            
            # Intersezione: layer presenti in TUTTI i modelli
            common_layers = set.intersection(*layer_sets) if layer_sets else set()
            
            logger.info(f"Layer comuni trovati: {len(common_layers)}")
            
            if not common_layers:
                logger.warning("Nessun layer comune trovato tra i modelli")
                return {}
            
            # Step 3: Verifica compatibilità shape e calcola media
            centroid = {}
            excluded_layers = []
            
            for layer_name in common_layers:
                # Raccogli tutti i tensori per questo layer
                tensors = []
                shapes = []
                
                for weights in normalized_weights:
                    tensor = weights[layer_name]
                    
                    # Assicurati che sia un tensore PyTorch
                    if isinstance(tensor, torch.Tensor):
                        tensors.append(tensor)
                        shapes.append(tuple(tensor.shape))
                    else:
                        logger.warning(
                            f"Layer '{layer_name}' in uno dei modelli non è un tensore PyTorch, "
                            f"tipo: {type(tensor)}"
                        )
                        excluded_layers.append(layer_name)
                        break
                
                # Se non abbiamo raccolto tensori da tutti i modelli, salta
                if len(tensors) != len(normalized_weights):
                    continue
                
                # Step 4: Verifica shape identico (strict)
                first_shape = shapes[0]
                if not all(shape == first_shape for shape in shapes):
                    logger.warning(
                        f"Layer '{layer_name}' escluso: shape incompatibili tra modelli. "
                        f"Shapes trovati: {set(shapes)}"
                    )
                    excluded_layers.append(layer_name)
                    continue
                
                # Step 5: Calcola media
                # Converti in numpy per calcolo efficiente
                numpy_tensors = [t.detach().cpu().numpy() for t in tensors]
                
                # Media elemento per elemento
                avg_tensor = np.mean(numpy_tensors, axis=0)
                
                # Converti back a torch tensor
                centroid[layer_name] = torch.from_numpy(avg_tensor)
            
            # Step 6: Logging finale
            logger.info(
                f"Centroide calcolato con successo: {len(centroid)} layer validi, "
                f"{len(excluded_layers)} layer esclusi"
            )
            
            if excluded_layers:
                logger.info(f"Layer esclusi: {excluded_layers[:10]}{'...' if len(excluded_layers) > 10 else ''}")
            
            if not centroid:
                logger.warning("Nessun layer compatibile trovato per il centroide")
                return {}
            
            return centroid
            
        except Exception as e:
            logger.error(f"Errore durante il calcolo del centroide: {e}", exc_info=True)
            return {}

    def update_centroid_metadata(self, neo4j_service, family_id: str, centroid: Dict[str, Any], model_count: int):
        """Update Centroid node metadata with enhanced attributes"""
        try:
            # Extract layer keys from centroid
            layer_keys = list(centroid.keys()) if centroid else []
            
            # Update the Centroid node with metadata
            with neo4j_service.driver.session(database='neo4j') as session:
                query = """
                MATCH (c:Centroid {family_id: $family_id})
                SET c.layer_keys = $layer_keys,
                    c.model_count = $model_count,
                    c.updated_at = $updated_at,
                    c.distance_metric = $distance_metric,
                    c.version = $version
                RETURN c
                """
                
                session.run(query, {
                    'family_id': family_id,
                    'layer_keys': layer_keys,
                    'model_count': model_count,
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'distance_metric': 'cosine',
                    'version': '1.1'  # Updated version
                })
                
            logger.info(f"✅ Updated Centroid metadata for family {family_id}: {len(layer_keys)} layers, {model_count} models")
            
        except Exception as e:
            logger.error(f"Failed to update centroid metadata for family {family_id}: {e}")