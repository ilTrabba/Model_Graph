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
from src.mother_algorithm.mother_utils import load_model_weights
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from enum import Enum
from datetime import datetime
from safetensors import safe_open
from ..db_entities.entity import Model
from src.services.neo4j_service import neo4j_service
from .distance_calculator import ModelDistanceCalculator
from src.utils.architecture_filtering import FilteringPatterns

logger = logging.getLogger(__name__)

#FIXME: capire bene quali soglie settare empiricamente e non
MIN_CONFIDENCE = 0.0
THRESHOLD_SINGLE_MEMBER = 50

# Costante per soglia chunking automatico
CENTROID_CHUNK_THRESHOLD = 10_000_000  # 40 MB in float32

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
                'distance_metric': 'L2',
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


    def calculate_adaptive_threshold(self, family_stats: Dict, k=2.0):

        try:
            member_count = family_stats["members"]
            avg_intra_distance = family_stats["avg_intra_distance"]
            std_intra_distance = family_stats["std_intra_distance"]

            if member_count == 1:
                # Famiglia con 1 solo membro: soglia conservativa fissa
                return THRESHOLD_SINGLE_MEMBER  # es. 1.5
            
            elif member_count == 2:
                # 1 sola relazione: std non affidabile, usa margine sulla media
                return avg_intra_distance * 20000 #1.5  # 50% margine
            
            else:
                # >= 3 membri: formula standard
                return 10000 + avg_intra_distance + k * std_intra_distance
            
        except Exception as e:
            logHandler.error_handler(e, "calculate_adaptive_threshold")
            return None

    def calculate_confidence(self, best_distance, avg_intra_distance, threshold):
        try:
            if best_distance <= avg_intra_distance:
                return 1.0
            elif best_distance <= threshold:
                # Scala lineare tra mean e threshold
                return 1.0 - (best_distance - avg_intra_distance) / (threshold - avg_intra_distance)
            else:
                return 0.0
        except Exception as e:
            logHandler.error_handler(e,"calculate_confidence")
            return 0.0

    def extract_family_metrics(self, best_family_id: str) -> Dict:
        try:
            models = neo4j_service.get_family_models(best_family_id)
            distances = neo4j_service.get_direct_relationship_distances(best_family_id)
            avg_intra_distance = neo4j_service.get_family_by_id(best_family_id).get('avg_intra_distance')
            #avg_intra_distance = self.distance_calculator.calculate_intra_family_distance(models)
            std_intra_distance = self.distance_calculator.calculate_std_intra_distance(distances, avg_intra_distance)
            members = len(models)

            return {
                'avg_intra_distance': avg_intra_distance,
                'std_intra_distance': std_intra_distance,
                'members': members
            }

        except Exception as e:
            logHandler.error_handler(e, "extract_family_metrics")
            return None
 
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
            
            logger.info(f"Model has this structural_hash:{model.structural_hash}")

            if (model.is_foundation_model): 
                candidate_centroids = neo4j_service.get_all_centroids_without_foundation(model.structural_hash)
            else:
                # Get all existing models with the same structural pattern
                candidate_centroids = neo4j_service.get_all_centroids(model.structural_hash)

            # If there aren't candidate centroids create a new one with the model weights for the centroid
            if not candidate_centroids:
                family_id = self.create_new_family(model, model_weights)
                confidence = 1.0
            else:
                
                # Calculate distances to family centroids
                best_family_id, best_distance = self.find_best_family_match(
                    model_weights, candidate_centroids
                )
                
                family_stats = self.extract_family_metrics(best_family_id)  # mean, std, member_count

                threshold = self.calculate_adaptive_threshold(family_stats, k=2.0)

                confidence = self.calculate_confidence(best_distance, family_stats["avg_intra_distance"], threshold)
                
                if best_distance < threshold and confidence > MIN_CONFIDENCE:
                    family_id = best_family_id
                    
                else:
                    family_id = self.create_new_family(model, model_weights)
                    confidence = 1.0
            
            # Update model with family assignment
            neo4j_service.update_model(model.id, {'family_id': family_id})

            if(model.is_foundation_model):
                updates = {
                    'has_foundation_model': True,
                    'updated_at': datetime.now(timezone.utc)
                    }
                neo4j_service.update_family(family_id, updates)

            return family_id, confidence
        except Exception as e:
            logHandler.error_handler(e, "assign_model_to_family")
    
    def calculate_family_centroid(self, family_id: str, new_model: Model) -> Optional[Dict[str, Any]]:
        """
        Calculate the centroid (average weights) for a family.
        
        Args:
            family_id: ID of the family
            
        Returns:
            Dictionary representing the family centroid weights
        """
        try:
            if not family_id:
                return None
            
            # Load weights for the new model
            new_model_weights = load_model_weights(new_model.file_path)
            if not new_model_weights:
                return None
            
            current_centroid = neo4j_service.get_centroid_by_family_id(family_id)

            # Calculate centroid by averaging weights
            updated_centroid = self.calculate_weights_centroid(current_centroid, new_model_weights)
            
            # Save centroid to file for incremental updates
            if updated_centroid:
                self.save_family_centroid(family_id, updated_centroid)
                
                # Update Neo4j centroid node with actual embedding and metadata
                try:
                    if neo4j_service.is_connected():

                        # Update enhanced Centroid node with metadata
                        neo4j_service.update_centroid_metadata(family_id)
                        
                except Exception as neo4j_error:
                    logHandler.warning_handler(f"Failed to update Neo4j centroid for family {family_id}: {neo4j_error}", "calculate_family_centroid")
            
            logger.info(f"Calculated centroid for {family_id}")
            return updated_centroid
            
        except Exception as e:
            logHandler.error_handler(f"{e}", "calculate_family_centroid")
            return None
    
    def find_best_family_match(self,
                               model_weights: Dict[str, Any],
                               candidate_centroids: List[Dict[str, Any]]) -> Tuple[str, float]:
        """
        Find the best family match for a model.
        
        Returns:
            Tuple of (best_family_id, confidence_score)
        """
        from src.clustering.distance_calculator import DistanceMetric
        try:
            distance_metric = DistanceMetric.L2_DISTANCE
            best_family_id = None
            best_distance = float('inf')
            
            for centroid in candidate_centroids:
                centroid_weights = None

                # prendere centroide già esistente della famiglia corrente con un if
                centroid_path = centroid["file_path"]
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
                    model_weights, centroid_weights, distance_metric, FilteringPatterns.FULL_MODEL
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_family_id = family_id

            
            if best_family_id is None:
                return "", 0.0
        
            # Convert distance to confidence score
            # print(f"{self.family_threshold}")

            # value1 = best_distance / 4.2
            # value2 = 1 - value1
            # confidence = max (0.0, value2)
            # confidence = min (1.0, confidence)
            
            return best_family_id, best_distance
            
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
                'member_count': 1,
                'avg_intra_distance': 0.0,
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'has_foundation_model': False
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
                            neo4j_service.create_centroid_with_metadata(family_id, model.structural_hash)
                            
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

    @torch.no_grad()
    def calculate_weights_centroid(
        self, 
        current_centroid: Dict[str, Any], 
        new_model_weights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcola centroide aggiornato aggiungendo un nuovo modello alla media esistente.
        
        Formula: centroid_new = (centroid_old * n + new_model) / (n + 1)
        
        Caratteristiche:
        - Aggiornamento in-place di current_centroid
        - Calcoli in float32 per massima precisione
        - Chunking automatico per tensori grandi (>10M elementi)
        - Solo layer comuni (nome esatto + shape identica)
        - Layer non comuni nel centroide rimangono invariati
        
        Args:
            current_centroid: Dizionario con centroide corrente (include 'model_count')
            new_model_weights: Dizionario con pesi del nuovo modello
            
        Returns:
            Centroide aggiornato (stesso oggetto di current_centroid, modificato in-place)
            
        Note:
            - I nomi dei layer devono essere già normalizzati
            - Gestisce automaticamente device mismatch
            - Tensori non compatibili vengono esclusi (con logging)
        """
        try:
            # Step 1: Validazione input
            if not new_model_weights:
                logger.warning("Pesi del nuovo modello vuoti, impossibile calcolare centroide")
                return current_centroid
            
            n = current_centroid.get('model_count', 0)
            
            # Non dovrebbe mai accadere
            if n <= 0:
                return logHandler.error_handler(f"model_count invalido nel centroide: {n}", "calculate_weights_centroid")
                 
            
            # Step 2: Trova layer comuni (intersezione per nome esatto)
            current_centroid_weights = load_model_weights(current_centroid['file_path'])

            centroid_layers = set(current_centroid_weights.keys()) 
            new_model_layers = set(new_model_weights.keys())

            common_layers = centroid_layers & new_model_layers
            
            logger.info(
                f"Layer: {len(centroid_layers)} nel centroide, "
                f"{len(new_model_layers)} nel nuovo modello, "
                f"{len(common_layers)} comuni"
            )
            
            if not common_layers:
                logger.warning("Nessun layer comune trovato tra centroide e nuovo modello")
                return current_centroid_weights
            
            # Step 3: Aggiorna layer comuni
            updated_count = 0
            excluded_count = 0
            chunked_layers = []
            
            for layer_name in common_layers:
                centroid_tensor = current_centroid_weights[layer_name]
                new_model_tensor = new_model_weights[layer_name]
                
                # Validazione 1: Entrambi devono essere tensori PyTorch
                if not (isinstance(centroid_tensor, torch.Tensor) and 
                        isinstance(new_model_tensor, torch.Tensor)):
                    logger.warning(
                        f"Layer '{layer_name}' escluso: non entrambi tensori PyTorch "
                        f"(centroid: {type(centroid_tensor)}, new: {type(new_model_tensor)})"
                    )
                    excluded_count += 1
                    continue
                
                # Validazione 2: Shape identiche
                if centroid_tensor.shape != new_model_tensor.shape:
                    logger.warning(
                        f"Layer '{layer_name}' escluso: shape mismatch "
                        f"({centroid_tensor.shape} vs {new_model_tensor.shape})"
                    )
                    excluded_count += 1
                    continue
                
                # Validazione 3: Device matching
                if centroid_tensor.device != new_model_tensor.device:
                    logger.debug(
                        f"Layer '{layer_name}': spostamento device "
                        f"{new_model_tensor.device} → {centroid_tensor.device}"
                    )
                    new_model_tensor = new_model_tensor.to(centroid_tensor.device)
                
                # Conversione a float32 per calcoli (massima precisione)
                original_dtype = centroid_tensor.dtype
                if centroid_tensor.dtype != torch.float32:
                    centroid_tensor = centroid_tensor.float()
                if new_model_tensor.dtype != torch.float32:
                    new_model_tensor = new_model_tensor.float()
                
                # Ottimizzazione: contiguous solo se necessario
                if not centroid_tensor.is_contiguous():
                    centroid_tensor = centroid_tensor.contiguous()
                if not new_model_tensor.is_contiguous():
                    new_model_tensor = new_model_tensor.contiguous()
                
                # Calcolo media con chunking automatico per tensori grandi
                numel = centroid_tensor.numel()
                
                if numel > CENTROID_CHUNK_THRESHOLD:
                    # CHUNKING: suddividi in blocchi per evitare OOM
                    chunk_size = CENTROID_CHUNK_THRESHOLD
                    num_chunks = (numel + chunk_size - 1) // chunk_size
                    
                    centroid_flat = centroid_tensor.view(-1)
                    new_model_flat = new_model_tensor.view(-1)
                    
                    for chunk_old, chunk_new in zip(
                        centroid_flat.split(chunk_size),
                        new_model_flat.split(chunk_size)
                    ):
                        # Formula incrementale: (old * n + new) / (n + 1)
                        chunk_old.mul_(n).add_(chunk_new).div_(n + 1)
                    
                    chunked_layers.append((layer_name, numel, num_chunks))
                    
                else:
                    # NORMALE: aggiornamento diretto (in-place)
                    centroid_tensor.mul_(n).add_(new_model_tensor).div_(n + 1)
                
                # Riconverti al dtype originale se necessario
                if centroid_tensor.dtype != original_dtype:
                    centroid_tensor = centroid_tensor.to(original_dtype)
                
                # Salva il layer aggiornato (in-place su current_centroid)
                current_centroid_weights[layer_name] = centroid_tensor
                updated_count += 1
            
            # Step 4: Logging finale
            logger.info(
                f"✅ Centroide aggiornato: {updated_count} layer modificati, "
                f"{excluded_count} layer esclusi, "
                f"{len(centroid_layers) - len(common_layers)} layer invariati"
            )
            
            if chunked_layers:
                logger.info(f"Chunking applicato a {len(chunked_layers)} layer grandi:")
                for layer_name, numel, num_chunks in chunked_layers[:5]:  # Primi 5
                    logger.info(
                        f"  - {layer_name}: {numel:,} elementi "
                        f"({num_chunks} chunk da {CENTROID_CHUNK_THRESHOLD:,})"
                    )
            
            if updated_count == 0:
                logHandler.warning_handler("Nessun layer aggiornato: centroide invariato", "calculate_weights_centroid")
            
            return current_centroid_weights
            
        except Exception as e:
            logHandler.error_handler(f"❌ Errore durante calcolo centroide: {e}", "calculate_weights_centroid")
            return current_centroid_weights