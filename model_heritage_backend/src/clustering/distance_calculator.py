"""
ModelDistanceCalculator

This module provides distance calculation between AI model weights using different metrics
optimized for various model types (fully fine-tuned models vs LoRA models).
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Import existing MoTHer utilities
from src.algorithms.mother_utils import (
    load_model_weights, 
    calculate_l2_distance,
    _get_layer_kinds
)

logger = logging.getLogger(__name__)

class DistanceMetric(Enum):
    """Available distance metrics for model comparison"""
    L2_DISTANCE = "l2_distance"
    MATRIX_RANK = "matrix_rank"
    COSINE_SIMILARITY = "cosine_similarity"
    AUTO = "auto"  # Automatically choose based on model type

class ModelType(Enum):
    """Model types for optimized distance calculation"""
    FULL_FINETUNED = "full_finetuned"
    LORA = "lora"
    AUTO = "auto"

class ModelDistanceCalculator:
    """
    Calculate distances between AI model weights using different metrics.
    
    Supports:
    - L2 distance for fully fine-tuned models
    - Matrix rank method for LoRA models  
    - Filtered layer analysis (attention, dense, linear layers)
    """
    
    def __init__(self, 
                 default_metric: DistanceMetric = DistanceMetric.AUTO,
                 layer_filter: Optional[List[str]] = None):
        """
        Initialize the distance calculator.
        
        Args:
            default_metric: Default distance metric to use
            layer_filter: List of layer patterns to include. If None, uses default patterns.
        """
        self.default_metric = default_metric
        self.layer_filter = layer_filter or _get_layer_kinds()
        logger.info(f"Initialized ModelDistanceCalculator with metric: {default_metric}")
        
    def calculate_distance(self, 
                          model1_weights: Dict[str, Any],
                          model2_weights: Dict[str, Any],
                          metric: Optional[DistanceMetric] = None,
                          model_type: ModelType = ModelType.AUTO) -> float:
        """
        Calculate distance between two model weight dictionaries.
        
        Args:
            model1_weights: First model's weights dictionary
            model2_weights: Second model's weights dictionary  
            metric: Distance metric to use (overrides default)
            model_type: Type of models being compared
            
        Returns:
            Distance value (lower means more similar)
        """
        try:
            metric = metric or self.default_metric
            
            # Auto-select metric based on model type
            if metric == DistanceMetric.AUTO:
                model_type_detected = self._detect_model_type(model1_weights, model2_weights)
                if model_type_detected == ModelType.LORA:
                    metric = DistanceMetric.MATRIX_RANK
                else:
                    metric = DistanceMetric.L2_DISTANCE
                    
            logger.debug(f"Using metric: {metric} for model comparison")
            
            if metric == DistanceMetric.L2_DISTANCE:
                return self._calculate_l2_distance(model1_weights, model2_weights)
            elif metric == DistanceMetric.MATRIX_RANK:
                return self._calculate_matrix_rank_distance(model1_weights, model2_weights)
            elif metric == DistanceMetric.COSINE_SIMILARITY:
                return self._calculate_cosine_distance(model1_weights, model2_weights)
            else:
                raise ValueError(f"Unsupported distance metric: {metric}")
                
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    #calulate distances of all pairs of models in a family
    def calculate_matrix_pairwise_distances(self,
                                   models_weights: Dict[str, Dict[str, Any]],
                                   metric: Optional[DistanceMetric] = None) -> np.ndarray:
        """
        Calculate pairwise distance matrix for multiple models.
        
        Args:
            models_weights: Dictionary mapping model_id -> weights_dict
            metric: Distance metric to use
            
        Returns:
            NxN distance matrix where N is number of models
        """
        try:
            model_ids = list(models_weights.keys())
            n_models = len(model_ids)
            
            if n_models < 2:
                logger.warning("Need at least 2 models for distance calculation")
                return np.array([[]])
                
            distance_matrix = np.zeros((n_models, n_models))
            
            for i in range(n_models):
                for j in range(n_models):
                    if i == j:
                        distance_matrix[i, j] = 0.0
                    elif i < j:  # Calculate only upper triangle
                        dist = self.calculate_distance(
                            models_weights[model_ids[i]],
                            models_weights[model_ids[j]],
                            metric
                        )
                        distance_matrix[i, j] = dist
                        distance_matrix[j, i] = dist  # Symmetric
                    
            logger.info(f"Calculated {n_models}x{n_models} distance matrix")
            return distance_matrix
            
        except Exception as e:
            logger.error(f"Error calculating pairwise distances: {e}")
            return np.array([[]])
    
    def load_and_calculate_distance(self,
                                  model1_path: str,
                                  model2_path: str,
                                  metric: Optional[DistanceMetric] = None) -> float:
        """
        Load model weights from files and calculate distance.
        
        Args:
            model1_path: Path to first model file
            model2_path: Path to second model file
            metric: Distance metric to use
            
        Returns:
            Distance value
        """
        try:
            weights1 = load_model_weights(model1_path)
            weights2 = load_model_weights(model2_path)
            
            if weights1 is None or weights2 is None:
                logger.error("Failed to load one or both model weights")
                return float('inf')
                
            return self.calculate_distance(weights1, weights2, metric)
            
        except Exception as e:
            logger.error(f"Error loading and calculating distance: {e}")
            return float('inf')
    
    def _calculate_l2_distance(self, 
                              weights1: Dict[str, Any], 
                              weights2: Dict[str, Any]) -> float:
        """
        Calculate L2 distance using existing MoTHer implementation.
        
        This leverages the existing calculate_l2_distance function from mother_utils
        which already handles layer filtering and tensor operations properly.
        """
        try:
            return calculate_l2_distance(weights1, weights2)
        except Exception as e:
            logger.error(f"Error in L2 distance calculation: {e}")
            return float('inf')
    
    def _calculate_matrix_rank_distance(self,
                                      weights1: Dict[str, Any],
                                      weights2: Dict[str, Any]) -> float:
        """
        Calculate distance based on matrix rank differences (optimized for LoRA models).
        
        For LoRA models, we analyze the rank of weight difference matrices
        as they typically have low-rank adaptations.
        """
        try:
            total_rank_diff = 0.0
            param_count = 0
            
            # Get common parameters
            common_params = set(weights1.keys()) & set(weights2.keys())
            
            for param_name in common_params:
                # Filter relevant layers
                if not self._should_include_layer(param_name):
                    continue
                    
                tensor1 = weights1[param_name]
                tensor2 = weights2[param_name]
                
                if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
                    if tensor1.shape != tensor2.shape:
                        continue
                        
                    # Calculate difference matrix
                    diff_matrix = tensor1.detach().cpu().numpy() - tensor2.detach().cpu().numpy()
                    
                    # For matrices, calculate rank
                    if len(diff_matrix.shape) == 2:
                        rank = np.linalg.matrix_rank(diff_matrix)
                        max_rank = min(diff_matrix.shape)
                        normalized_rank = rank / max_rank if max_rank > 0 else 0
                        total_rank_diff += normalized_rank
                    else:
                        # For other tensors, use frobenius norm as fallback
                        total_rank_diff += np.linalg.norm(diff_matrix)
                        
                    param_count += 1
            
            if param_count == 0:
                return float('inf')
                
            return total_rank_diff / param_count
            
        except Exception as e:
            logger.error(f"Error in matrix rank distance calculation: {e}")
            return float('inf')
    
    def _calculate_cosine_distance(self,
                                 weights1: Dict[str, Any],
                                 weights2: Dict[str, Any]) -> float:
        """
        Calculate cosine distance between model weights.
        """
        try:
            # Flatten all relevant weights into vectors
            vec1 = self._flatten_weights(weights1)
            vec2 = self._flatten_weights(weights2)
            
            if len(vec1) == 0 or len(vec2) == 0:
                return float('inf')
                
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return float('inf')
                
            cosine_sim = dot_product / (norm1 * norm2)
            
            # Convert to distance (1 - similarity)
            return 1.0 - cosine_sim
            
        except Exception as e:
            logger.error(f"Error in cosine distance calculation: {e}")
            return float('inf')
    
    def _flatten_weights(self, weights: Dict[str, Any]) -> np.ndarray:
        """
        Flatten relevant model weights into a single vector.
        """
        try:
            all_weights = []
            
            for param_name, tensor in weights.items():
                if not self._should_include_layer(param_name):
                    continue
                    
                if isinstance(tensor, torch.Tensor):
                    flattened = tensor.detach().cpu().numpy().flatten()
                    all_weights.extend(flattened)
                    
            return np.array(all_weights)
            
        except Exception as e:
            logger.error(f"Error flattening weights: {e}")
            return np.array([])
    
    def _should_include_layer(self, param_name: str) -> bool:
        """
        Check if a parameter should be included based on layer filtering.
        """
        for pattern in self.layer_filter:
            if pattern in param_name:
                return True
        return False
    
    def _detect_model_type(self, 
                          weights1: Dict[str, Any], 
                          weights2: Dict[str, Any]) -> ModelType:
        """
        Automatically detect model type based on weight patterns.
        
        LoRA models typically have specific naming patterns like 'lora_A', 'lora_B'
        """
        try:
            all_params = set(weights1.keys()) | set(weights2.keys())
            
            # Check for LoRA-specific patterns
            lora_patterns = ['lora_A', 'lora_B', '.lora.', '_lora_']
            for param in all_params:
                for pattern in lora_patterns:
                    if pattern in param:
                        return ModelType.LORA
                        
            return ModelType.FULL_FINETUNED
            
        except Exception as e:
            logger.error(f"Error detecting model type: {e}")
            return ModelType.FULL_FINETUNED