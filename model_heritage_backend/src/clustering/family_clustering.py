"""
FamilyClusteringSystem

This module manages model families and automatic clustering based on weight similarities.
It handles family assignment, centroid calculation, and family management operations.
"""

import logging
import numpy as np
import torch
import uuid
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
from enum import Enum
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import pairwise_distances
import safetensors.torch
from safetensors import safe_open
from ..models.model import Model, Family
from ..models.model import FamilyQuery, ModelQuery
from src.services.neo4j_service import neo4j_service
from .distance_calculator import ModelDistanceCalculator, DistanceMetric
from pathlib import Path

logger = logging.getLogger(__name__)

family_query = FamilyQuery()
model_query = ModelQuery()

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
        
        logger.info(f"Initialized FamilyClusteringSystem with threshold: {family_threshold}")
        logger.info(f"Centroids directory: {self.centroids_dir}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Absolute centroids path: {os.path.abspath(self.centroids_dir)}")
    
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
        #logger.info(f"Centroid keys: {list(centroid.keys()) if centroid else 'None'}")
        #logger.info(f"Current working directory: {os.getcwd()}")
        #logger.info(f"Centroids directory: {self.centroids_dir}")
        logger.info(f"Absolute centroids path: {os.path.abspath(self.centroids_dir)}")
        
        try:
            if not centroid:
                logger.warning(f"Cannot save empty centroid for family {family_id}")
                return False
            
            centroid_path = self.get_centroid_file_path(family_id)
            logger.info(f"Target path: {centroid_path}")
            logger.info(f"Absolute target path: {os.path.abspath(centroid_path)}")
            logger.info(f"Directory exists: {os.path.exists(os.path.dirname(centroid_path))}")
            
            # Ensure centroids directory exists
            os.makedirs(os.path.dirname(centroid_path), exist_ok=True)
            logger.info(f"Directory created/verified: {os.path.dirname(centroid_path)}")
            
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
                from src.algorithms.mother_utils import load_model_weights
                model_weights = load_model_weights(model.file_path)
                if model_weights is None:
                    logger.error(f"Failed to load weights for model {model.id}")
                    return self._create_new_family(model, None), 0.0
            
            # Get all existing models with the same structural pattern
            candidate_families = self._find_candidate_families(model)
            
            if not candidate_families:
                # No candidate families, create new one with the model weights
                family_id = self._create_new_family(model, model_weights)
                return family_id, 1.0
            
            # Calculate distances to family centroids
            best_family_id, confidence = self._find_best_family_match(
                model, model_weights, candidate_families
            )
            
            if confidence >= 0.2:
                # Assign to existing family
                self._add_model_to_family(model, best_family_id)
                #centroid2=self.calculate_family_centroid(family_id)  questa è sbagliata
                return best_family_id, confidence
            else:
                # Create new family with the model weights
                family_id = self._create_new_family(model, model_weights)
                return family_id, 1.0
                
        except Exception as e:
            logger.error(f"Error assigning model {model.id} to family: {e}")
            # Fallback: create new family
            family_id = self._create_new_family(model, model_weights)
            return family_id, 0.0
    
    def recluster_all_families(self, 
                             force_recalculate: bool = False) -> Dict[str, List[str]]:
        """
        Recluster all models into families using current settings.
        
        Args:
            force_recalculate: Whether to force recalculation of all distances
            
        Returns:
            Dictionary mapping family_id -> list of model_ids
        """
        try:
            logger.info("Starting complete family reclustering")
            
            # Get all processed models
            models = model_query.filter_by(status='ok').all()
            if len(models) < self.min_family_size:
                logger.warning("Not enough models for meaningful clustering")
                return {}
            
            # Load all model weights
            model_weights = {}
            valid_models = []
            
            for model in models:
                from src.algorithms.mother_utils import load_model_weights
                weights = load_model_weights(model.file_path)
                if weights is not None:
                    model_weights[model.id] = weights
                    valid_models.append(model)
                else:
                    logger.warning(f"Failed to load weights for model {model.id}")
            
            if len(valid_models) < self.min_family_size:
                logger.warning("Not enough valid models for clustering")
                return {}
            
            # Calculate pairwise distances
            distance_matrix = self.distance_calculator.calculate_pairwise_distances(
                model_weights
            )
            
            # Perform clustering
            cluster_labels = self._perform_clustering(distance_matrix, valid_models)
            
            # Update family assignments
            new_families = self._update_family_assignments(valid_models, cluster_labels, model_weights)
            
            logger.info(f"Reclustering complete. Created {len(new_families)} families")
            return new_families
            
        except Exception as e:
            logger.error(f"Error during family reclustering: {e}")
            return {}
    
    def calculate_family_centroid(self, family_id: str) -> Optional[Dict[str, Any]]:
        """
        Calculate the centroid (average weights) for a family.
        
        Args:
            family_id: ID of the family
            
        Returns:
            Dictionary representing the family centroid weights
        """
        try:
            # Get all models in the family
            family_models = model_query.filter_by(
                family_id=family_id, 
                status='ok'
            ).all()
            
            if not family_models:
                return None
            
            # Load weights for all family models
            family_weights = []
            for model in family_models:
                from src.algorithms.mother_utils import load_model_weights
                weights = load_model_weights(model.file_path)
                if weights is not None:
                    family_weights.append(weights)
            
            if not family_weights:
                return None
            
            # Calculate centroid by averaging weights
            centroid = self._calculate_weights_centroid(family_weights)
            
            # Save centroid to file for incremental updates
            if centroid:
                self.save_family_centroid(family_id, centroid)
                
                # Update Neo4j centroid node with actual embedding and metadata
                try:
                    if neo4j_service.is_connected():
                        # Update embedding in FamilyCentroid (for backward compatibility)
                        embedding = self.centroid_to_embedding(centroid)
                        #neo4j_service.create_or_update_family_centroid(family_id, embedding)
                        
                        # Update enhanced Centroid node with metadata
                        self._update_centroid_metadata(neo4j_service, family_id, centroid, len(family_weights))
                        
                        neo4j_service.create_has_centroid_relationship(family_id)
                except Exception as neo4j_error:
                    logger.warning(f"Failed to update Neo4j centroid for family {family_id}: {neo4j_error}")
            
            logger.info(f"Calculated centroid for family {family_id} with {len(family_weights)} models")
            return centroid
            
        except Exception as e:
            logger.error(f"Error calculating family centroid for {family_id}: {e}")
            return None
    
    def update_family_statistics(self, family_id: str) -> bool:
        """
        Update family statistics including member count and average intra-distance.
        
        Args:
            family_id: ID of the family to update
            
        Returns:
            True if successfully updated, False otherwise
        """
        try:
            family = family_query.get(family_id)
            if not family:
                return False
            
            # Get family models
            family_models = model_query.filter_by(
                family_id=family_id,
                status='ok'
            ).all()
            
            # Update member count
            member_count = len(family_models)
            
            # Calculate average intra-family distance
            if len(family_models) >= 2:
                avg_distance = self._calculate_intra_family_distance(family_models)
            else:
                avg_distance = 0.0
            
            # Update family in Neo4j
            updates = {
                'member_count': member_count,
                'avg_intra_distance': avg_distance,
                'updated_at': datetime.now(timezone.utc)
            }
            neo4j_service.create_or_update_family({
                'id': family_id,
                **updates
            })
            
            # Trigger centroid recalculation for incremental updates
            if len(family_models) >= 1:
                try:
                    self.calculate_family_centroid(family_id)
                except Exception as centroid_error:
                    logger.warning(f"Failed to update centroid for family {family_id}: {centroid_error}")
            
            logger.info(f"Updated statistics for family {family_id}: {member_count} members, avg_distance: {avg_distance:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating family statistics for {family_id}: {e}")
            return False
    
    def _find_candidate_families(self, model: Model) -> List[Family]:
        """
        Find candidate families for a model based on structural similarity.
        """
        try:
            # For now, consider all families as candidates
            # In future versions, could filter by structural_hash or other criteria
            return family_query.all()
            
        except Exception as e:
            logger.error(f"Error finding candidate families: {e}")
            return []
    
    def _find_best_family_match(self,
                               model: Model,
                               model_weights: Dict[str, Any],
                               candidate_families: List[Family]) -> Tuple[str, float]:
        """
        Find the best family match for a model.
        
        Returns:
            Tuple of (best_family_id, confidence_score)
        """
        try:
            best_family_id = None
            best_distance = float('inf')
            
            for family in candidate_families:
                #prendere centroide già esistente della famiglia corrente con un if
                centroid_path = os.path.join("weights", "centroids", f"{family.id}.safetensors")
                esiste_centroide = Path(centroid_path).exists()
                print(f"{esiste_centroide}")


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
                        centroid = centroid_data

                    except Exception as e:
                        print(f"Error loading safetensors file: {e}")
                        centroid = None
                else:
                    # Calculate distance to family centroid
                    #centroid = self.calculate_family_centroid(family.id)
                    print("hakuna matata")

                if centroid is None:
                    continue
                
                distance = self.distance_calculator.calculate_distance(
                    model_weights, centroid
                )
                
                if distance < best_distance:
                    best_distance = distance
                    best_family_id = family.id
            
            if best_family_id is None:
                return "", 0.0
        
            # Convert distance to confidence score
            print(f"{self.family_threshold}")
            value1 = best_distance / 2.0
            value2 = 1- value1
            confidence = max (0.0, value2)
            confidence = min (1.0, confidence)
            
            return best_family_id, confidence
            
        except Exception as e:
            logger.error(f"Error finding best family match: {e}")
            return "", 0.0
    
    def _create_new_family(self, model: Model, model_weights: Optional[Dict[str, Any]] = None) -> str:
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
            
            # Create initial centroid from the first model's weights if provided
            if model_weights:
                logger.info(f"Creating initial centroid for family {family_id} from model {model.id}")
                
                # Use the first model's weights as the initial centroid
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
                            # Create FamilyCentroid node (for backward compatibility)
                            embedding = self.centroid_to_embedding(initial_centroid)
                            #neo4j_service.create_or_update_family_centroid(family_id, embedding)
                            
                            # Update enhanced Centroid node with metadata
                            self._update_centroid_metadata(neo4j_service, family_id, initial_centroid, 1)
                            
                            neo4j_service.create_has_centroid_relationship(family_id)
                    except Exception as neo4j_error:
                        logger.warning(f"Failed to update Neo4j centroid for new family {family_id}: {neo4j_error}")
                    
                    logger.info(f"✅ Initial centroid created and saved for family {family_id}")
                else:
                    logger.warning(f"No valid weights found to create centroid for family {family_id}")
            else:
                logger.info(f"No model weights provided - centroid will be created later for family {family_id}")
            
            logger.info(f"Created new family {family_id} for model {model.id}")
            return family_id
            
        except Exception as e:
            logger.error(f"Error creating new family: {e}")
            return f"family_{str(uuid.uuid4())[:8]}"
    
    def _add_model_to_family(self, model: Model, family_id: str):
        """
        Add a model to an existing family and update statistics.
        """
        try:
            # Update family statistics
            self.update_family_statistics(family_id)
            logger.info(f"Added model {model.id} to family {family_id}")
            
        except Exception as e:
            logger.error(f"Error adding model to family: {e}")
    
    def _perform_clustering(self, 
                          distance_matrix: np.ndarray, 
                          models: List[Model]) -> np.ndarray:
        """
        Perform clustering on the distance matrix.
        
        Returns:
            Array of cluster labels for each model
        """
        try:
            if self.clustering_method == ClusteringMethod.DBSCAN:
                clustering = DBSCAN(
                    metric='precomputed',
                    eps=self.family_threshold,
                    min_samples=self.min_family_size
                )
                labels = clustering.fit_predict(distance_matrix)
                
            elif self.clustering_method == ClusteringMethod.KMEANS:
                # Estimate number of clusters
                n_clusters = max(1, len(models) // 5)  # Rough heuristic
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                # Convert distance to similarity for KMeans
                similarity_matrix = 1.0 / (1.0 + distance_matrix)
                labels = clustering.fit_predict(similarity_matrix)
                
            elif self.clustering_method == ClusteringMethod.THRESHOLD:
                # Simple threshold-based clustering
                labels = self._threshold_clustering(distance_matrix)
                
            else:
                # AUTO: choose best method based on data
                if len(models) < 10:
                    labels = self._threshold_clustering(distance_matrix)
                else:
                    clustering = DBSCAN(
                        metric='precomputed',
                        eps=self.family_threshold,
                        min_samples=self.min_family_size
                    )
                    labels = clustering.fit_predict(distance_matrix)
            
            return labels
            
        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
            # Fallback: each model in its own cluster
            return np.arange(len(models))
    
    def _threshold_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Simple threshold-based clustering.
        """
        try:
            n_models = distance_matrix.shape[0]
            labels = np.full(n_models, -1)  # -1 means unassigned
            current_label = 0
            
            for i in range(n_models):
                if labels[i] == -1:  # Unassigned
                    # Start new cluster
                    cluster_members = [i]
                    
                    # Find all models within threshold
                    for j in range(i + 1, n_models):
                        if labels[j] == -1 and distance_matrix[i, j] <= self.family_threshold:
                            cluster_members.append(j)
                    
                    # Assign cluster label if meets minimum size
                    if len(cluster_members) >= self.min_family_size:
                        for member in cluster_members:
                            labels[member] = current_label
                        current_label += 1
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in threshold clustering: {e}")
            return np.arange(distance_matrix.shape[0])
    
    def _update_family_assignments(self, 
                                 models: List[Model], 
                                 cluster_labels: np.ndarray,
                                 model_weights: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, List[str]]:
        """
        Update database with new family assignments.
        
        Args:
            models: List of models to assign
            cluster_labels: Cluster labels for each model 
            model_weights: Dictionary mapping model_id -> model weights for centroid creation
        """
        try:
            # Create new families and assign models
            families = {}
            label_to_family_id = {}
            model_updates = []  # Track model updates to batch them
            
            for i, (model, label) in enumerate(zip(models, cluster_labels)):
                if label == -1:  # Noise/unassigned
                    # Create individual family
                    weights = model_weights.get(model.id) if model_weights else None
                    family_id = self._create_new_family(model, weights)
                    model_updates.append((model.id, {'family_id': family_id}))
                    families[family_id] = [model.id]
                else:
                    # Get or create family for this cluster
                    if label not in label_to_family_id:
                        family_id = f"family_{str(uuid.uuid4())[:8]}"
                        label_to_family_id[label] = family_id
                        families[family_id] = []
                        
                        # Create family record in Neo4j
                        family_data = {
                            'id': family_id,
                            'structural_pattern_hash': model.structural_hash,
                            'member_count': 0,
                            'avg_intra_distance': 0.0,
                            'created_at': datetime.now(timezone.utc),
                            'updated_at': datetime.now(timezone.utc)
                        }
                        neo4j_service.create_or_update_family(family_data)
                    
                    family_id = label_to_family_id[label]
                    model_updates.append((model.id, {'family_id': family_id}))
                    families[family_id].append(model.id)
            
            # Update all models with their family assignments
            for model_id, updates in model_updates:
                neo4j_service.update_model(model_id, updates)
            
            # Update family statistics
            for family_id in families.keys():
                self.update_family_statistics(family_id)
            
            return families
            
        except Exception as e:
            logger.error(f"Error updating family assignments: {e}")
            return {}
    
    def _calculate_intra_family_distance(self, family_models: List[Model]) -> float:
        """
        Calculate average intra-family distance.
        """
        try:
            if len(family_models) < 2:
                return 0.0
            
            # Load weights for all models
            model_weights = {}
            for model in family_models:
                from src.algorithms.mother_utils import load_model_weights
                weights = load_model_weights(model.file_path)
                if weights is not None:
                    model_weights[model.id] = weights
            
            if len(model_weights) < 2:
                return 0.0
            
            # Calculate pairwise distances
            distances = []
            model_ids = list(model_weights.keys())
            
            for i in range(len(model_ids)):
                for j in range(i + 1, len(model_ids)):
                    dist = self.distance_calculator.calculate_distance(
                        model_weights[model_ids[i]],
                        model_weights[model_ids[j]]
                    )
                    distances.append(dist)
            
            return np.mean(distances) if distances else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating intra-family distance: {e}")
            return 0.0
    
    def _calculate_weights_centroid(self, weights_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate centroid by averaging model weights.
        """
        try:
            if not weights_list:
                return {}
            
            # Get common parameters across all models
            common_params = set(weights_list[0].keys())
            for weights in weights_list[1:]:
                common_params &= set(weights.keys())
            
            centroid = {}
            
            for param_name in common_params:
                # Collect all tensors for this parameter
                tensors = []
                for weights in weights_list:
                    tensor = weights[param_name]
                    if isinstance(tensor, torch.Tensor):
                        tensors.append(tensor.detach().cpu().numpy())
                
                if tensors:
                    # Average the tensors
                    avg_tensor = np.mean(tensors, axis=0)
                    centroid[param_name] = torch.from_numpy(avg_tensor)
            
            return centroid
            
        except Exception as e:
            logger.error(f"Error calculating weights centroid: {e}")
            return {}
    
    def _update_centroid_metadata(self, neo4j_service, family_id: str, centroid: Dict[str, Any], model_count: int):
        """Update Centroid node metadata with enhanced attributes"""
        try:
            from datetime import datetime
            
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
                
            logger.info(f"Updated Centroid metadata for family {family_id}: {len(layer_keys)} layers, {model_count} models")
            
        except Exception as e:
            logger.error(f"Failed to update centroid metadata for family {family_id}: {e}")