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

from src.models.model import Model, Family, db
from .distance_calculator import ModelDistanceCalculator, DistanceMetric

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
                 family_threshold: float = 0.5,
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
        
        # Centroid storage configuration
        self.centroids_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'weights', 'centroids')
        os.makedirs(self.centroids_dir, exist_ok=True)
        
        logger.info(f"Initialized FamilyClusteringSystem with threshold: {family_threshold}")
        logger.info(f"Centroids directory: {self.centroids_dir}")
    
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
        try:
            if not centroid:
                logger.warning(f"Cannot save empty centroid for family {family_id}")
                return False
            
            centroid_path = self.get_centroid_file_path(family_id)
            
            # Ensure centroids directory exists
            os.makedirs(os.path.dirname(centroid_path), exist_ok=True)
            
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
            logger.info(f"Saved centroid for family {family_id} to {centroid_path} (SafeTensors format)")
            return True
            
        except Exception as e:
            logger.error(f"Error saving centroid for family {family_id}: {e}")
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
                logger.debug(f"No centroid file found for family {family_id}")
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
        """
        Convert centroid weights to a single embedding vector for Neo4j.
        
        Args:
            centroid: Dictionary containing centroid weights
            
        Returns:
            List of floats representing the centroid as an embedding
        """
        try:
            if not centroid:
                return [0.0]
            
            # Flatten all weight tensors into a single vector
            embedding_parts = []
            
            for param_name, tensor in centroid.items():
                if isinstance(tensor, torch.Tensor):
                    # Flatten the tensor and convert to list
                    flattened = tensor.detach().cpu().numpy().flatten()
                    embedding_parts.extend(flattened.tolist())
            
            # If no valid tensors found, return placeholder
            if not embedding_parts:
                return [0.0]
            
            # Truncate if too long (Neo4j has practical limits)
            max_embedding_size = 1000  # Reasonable limit for Neo4j
            if len(embedding_parts) > max_embedding_size:
                # Sample evenly across the embedding
                step = len(embedding_parts) // max_embedding_size
                embedding_parts = embedding_parts[::step][:max_embedding_size]
            
            return embedding_parts
            
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
                    return self._create_new_family(model), 0.0
            
            # Get all existing models with the same structural pattern
            candidate_families = self._find_candidate_families(model)
            
            if not candidate_families:
                # No candidate families, create new one
                family_id = self._create_new_family(model)
                return family_id, 1.0
            
            # Calculate distances to family centroids
            best_family_id, confidence = self._find_best_family_match(
                model, model_weights, candidate_families
            )
            
            if confidence >= self.family_threshold:
                # Assign to existing family
                self._add_model_to_family(model, best_family_id)
                return best_family_id, confidence
            else:
                # Create new family
                family_id = self._create_new_family(model)
                return family_id, 1.0
                
        except Exception as e:
            logger.error(f"Error assigning model {model.id} to family: {e}")
            # Fallback: create new family
            family_id = self._create_new_family(model)
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
            models = Model.query.filter_by(status='ok').all()
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
            new_families = self._update_family_assignments(valid_models, cluster_labels)
            
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
            family_models = Model.query.filter_by(
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
                    from src.services.neo4j_service import Neo4jService
                    neo4j_service = Neo4jService()
                    if neo4j_service.is_connected():
                        # Update embedding in FamilyCentroid (for backward compatibility)
                        embedding = self.centroid_to_embedding(centroid)
                        neo4j_service.create_or_update_family_centroid(family_id, embedding)
                        
                        # Update enhanced Centroid node with metadata
                        self._update_centroid_metadata(neo4j_service, family_id, centroid, len(family_weights))
                        
                        neo4j_service.create_has_centroid_relationship(family_id)
                    neo4j_service.close()
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
            family = Family.query.get(family_id)
            if not family:
                return False
            
            # Get family models
            family_models = Model.query.filter_by(
                family_id=family_id,
                status='ok'
            ).all()
            
            # Update member count
            family.member_count = len(family_models)
            
            # Calculate average intra-family distance
            if len(family_models) >= 2:
                avg_distance = self._calculate_intra_family_distance(family_models)
                family.avg_intra_distance = avg_distance
            else:
                family.avg_intra_distance = 0.0
            
            family.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            
            # Trigger centroid recalculation for incremental updates
            if len(family_models) >= 1:
                try:
                    self.calculate_family_centroid(family_id)
                except Exception as centroid_error:
                    logger.warning(f"Failed to update centroid for family {family_id}: {centroid_error}")
            
            logger.info(f"Updated statistics for family {family_id}: {family.member_count} members, avg_distance: {family.avg_intra_distance:.4f}")
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
            return Family.query.all()
            
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
                # Calculate distance to family centroid
                centroid = self.calculate_family_centroid(family.id)
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
            confidence = max(0.0, 1.0 - (best_distance / (self.family_threshold * 2)))
            confidence = min(1.0, confidence)
            
            return best_family_id, confidence
            
        except Exception as e:
            logger.error(f"Error finding best family match: {e}")
            return "", 0.0
    
    def _create_new_family(self, model: Model) -> str:
        """
        Create a new family for the model.
        """
        try:
            family_id = f"family_{str(uuid.uuid4())[:8]}"
            
            family = Family(
                id=family_id,
                structural_pattern_hash=model.structural_hash,
                member_count=1,
                avg_intra_distance=0.0
            )
            
            db.session.add(family)
            db.session.commit()
            
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
                                 cluster_labels: np.ndarray) -> Dict[str, List[str]]:
        """
        Update database with new family assignments.
        """
        try:
            # Clear existing family assignments
            for model in models:
                model.family_id = None
            
            # Create new families and assign models
            families = {}
            label_to_family_id = {}
            
            for i, (model, label) in enumerate(zip(models, cluster_labels)):
                if label == -1:  # Noise/unassigned
                    # Create individual family
                    family_id = self._create_new_family(model)
                    model.family_id = family_id
                    families[family_id] = [model.id]
                else:
                    # Get or create family for this cluster
                    if label not in label_to_family_id:
                        family_id = f"family_{str(uuid.uuid4())[:8]}"
                        label_to_family_id[label] = family_id
                        families[family_id] = []
                        
                        # Create family record
                        family = Family(
                            id=family_id,
                            structural_pattern_hash=model.structural_hash,
                            member_count=0,
                            avg_intra_distance=0.0
                        )
                        db.session.add(family)
                    
                    family_id = label_to_family_id[label]
                    model.family_id = family_id
                    families[family_id].append(model.id)
            
            db.session.commit()
            
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
            with neo4j_service.driver.session(database=neo4j_service.driver._config.get('database', 'neo4j')) as session:
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