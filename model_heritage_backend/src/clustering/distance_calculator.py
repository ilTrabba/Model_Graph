"""
ModelDistanceCalculator

This module provides distance calculation between AI model weights using different metrics
optimized for various model types (fully fine-tuned models vs LoRA models).
"""

import logging
import numpy as np
import torch

from typing import Dict, List, Optional, Any
from src.log_handler import logHandler
from ..db_entities.entity import Model
from enum import Enum
from src.mother_algorithm.mother_utils import (
    load_model_weights,
    EXCLUDED_LAYER_PATTERNS
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
                 default_metric: DistanceMetric = DistanceMetric.COSINE_SIMILARITY,
                 layer_filter: Optional[List[str]] = None):
        """
        Initialize the distance calculator.
        
        Args:
            default_metric: Default distance metric to use
            layer_filter: List of layer patterns to include. If None, uses default patterns.
        """
        self.default_metric = default_metric
        # self.layer_filter = layer_filter or get_layer_kinds()

    def calculate_l2_distance(self, weights1: Dict[str, Any], weights2: Dict[str, Any]) -> float:
        """
        Calculate L2 distance between two sets of model weights.
        
        Only includes structural layers (attention, feedforward, convolutions, etc.)
        and excludes normalization, embedding, and head layers.
        
        Args:
            weights1: First model's normalized weights
            weights2: Second model's normalized weights
            
        Returns:
            Average L2 distance across common parameters, or inf if no valid layers
        """
        try:
            # Get common parameters (intersection)
            common_params = set(weights1.keys()) & set(weights2.keys())
            
            if not common_params:
                logger.warning("No common parameters found between models")
                return float('inf')
            
            logger.debug(f"Found {len(common_params)} common parameters")
            
            total_distance = 0.0
            param_count = 0
            excluded_count = 0
            
            for param_name in common_params:

                # Convert to lowercase once for case-insensitive matching
                param_lower = param_name.lower()
                
                # Exclude layers matching any pattern in blacklist
                if any(pattern in param_lower for pattern in EXCLUDED_LAYER_PATTERNS):
                    excluded_count += 1
                    continue
                
                tensor1 = weights1[param_name]
                tensor2 = weights2[param_name]
                
                # Verify both are tensors
                if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
                    logger.debug(f"Skipping {param_name}: not both tensors")
                    continue
                
                # Ensure same shape
                if tensor1.shape != tensor2.shape:
                    logger.warning(
                        f"Shape mismatch for {param_name}: "
                        f"{tensor1.shape} vs {tensor2.shape}"
                    )
                    continue
                
                # Calculate L2 distance
                diff = tensor1.detach().cpu().numpy() - tensor2.detach().cpu().numpy()
                l2_dist = np.linalg.norm(diff.flatten())
                total_distance += l2_dist
                param_count += 1
            
            # Log statistics
            logger.info(
                f"L2 distance calculation: {param_count} layers included, "
                f"{excluded_count} layers excluded (normalization/embedding/head)"
            )
            
            if param_count == 0:
                logger.warning("No valid layers found for distance calculation")
                return float('inf')
            
            # Return average L2 distance
            avg_distance = total_distance / param_count
            logger.debug(f"Average L2 distance: {avg_distance:.6f}")
            
            return avg_distance
            
        except Exception as e:
            logger.error(f"Failed to calculate L2 distance: {e}", exc_info=True)
            return float('inf')

    # Potenzialmente da eliminare poichè non usata (in pratica mai usata)
    def calculate_matrix_rank_distance(self,
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
                # if not self._should_include_layer(param_name):
                #     continue
                    
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
    
    def calculate_cosine_distance(self, weights1: Dict[str, Any], weights2: Dict[str, Any]) -> float:
        """
        Calculate Cosine distance between two sets of model weights.
        
        Only includes structural layers (attention, feedforward, convolutions, etc.)
        and excludes normalization, embedding, and head layers.
        
        Args:
            weights1: First model's normalized weights
            weights2: Second model's normalized weights
            
        Returns:
            Average Cosine distance across common parameters, or inf if no valid layers
        """
        try:
            # Get common parameters (intersection)
            common_params = set(weights1.keys()) & set(weights2.keys())
            
            if not common_params:
                logger.warning("No common parameters found between models")
                return float('inf')
            
            logger.debug(f"Found {len(common_params)} common parameters")
            
            total_distance = 0.0
            param_count = 0
            excluded_count = 0
            
            for param_name in common_params:

                # Convert to lowercase once for case-insensitive matching
                param_lower = param_name.lower()
                
                # Exclude layers matching any pattern in blacklist
                if any(pattern in param_lower for pattern in EXCLUDED_LAYER_PATTERNS):
                    excluded_count += 1
                    continue
                
                tensor1 = weights1[param_name]
                tensor2 = weights2[param_name]
                
                # Verify both are tensors
                if not (isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor)):
                    logger.debug(f"Skipping {param_name}: not both tensors")
                    continue
                
                # Ensure same shape
                if tensor1.shape != tensor2.shape:
                    logger.warning(
                        f"Shape mismatch for {param_name}: "
                        f"{tensor1.shape} vs {tensor2.shape}"
                    )
                    continue
                
                # Calculate Cosine distance
                # Flatten tensors to 1D vectors
                vec1 = tensor1.detach().cpu().numpy().flatten()
                vec2 = tensor2.detach().cpu().numpy().flatten()
                
                # Calculate norms
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                # Handle edge case: zero vectors (should be extremely rare in model weights)
                if norm1 == 0 or norm2 == 0:
                    logger.warning(f"Zero norm detected for {param_name}, skipping layer")
                    continue
                
                # Calculate cosine similarity
                dot_product = np.dot(vec1, vec2)
                cosine_similarity = dot_product / (norm1 * norm2)
                
                # Convert to cosine distance (1 - similarity)
                # Clamp similarity to [-1, 1] to handle numerical errors
                cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
                cosine_dist = 1.0 - cosine_similarity
                
                total_distance += cosine_dist
                param_count += 1
            
            # Log statistics
            logger.info(
                f"Cosine distance calculation: {param_count} layers included, "
                f"{excluded_count} layers excluded (normalization/embedding/head)"
            )
            
            if param_count == 0:
                logger.warning("No valid layers found for distance calculation")
                return float('inf')
            
            # Return average Cosine distance
            avg_distance = total_distance / param_count
            logger.debug(f"Average Cosine distance: {avg_distance:.6f}")
            
            return avg_distance
            
        except Exception as e:
            logger.error(f"Failed to calculate Cosine distance: {e}", exc_info=True)
            return float('inf')

    def calculate_distance(self, 
                          model1_weights: Dict[str, Any],
                          model2_weights: Dict[str, Any],
                          metric: Optional[DistanceMetric] = None,
                          model_type: ModelType = ModelType.FULL_FINETUNED) -> float:
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
                if model_type == ModelType.LORA:
                    metric = DistanceMetric.MATRIX_RANK
                else:
                    metric = DistanceMetric.L2_DISTANCE
                    
            logger.debug(f"Using metric: {metric} for model comparison")
            
            if metric == DistanceMetric.L2_DISTANCE:
                return self.calculate_l2_distance(model1_weights, model2_weights)
            elif metric == DistanceMetric.MATRIX_RANK:
                return self.calculate_matrix_rank_distance(model1_weights, model2_weights)
            elif metric == DistanceMetric.COSINE_SIMILARITY:
                return self.calculate_cosine_distance(model1_weights, model2_weights)
            else:
                raise Exception(f"Unsupported distance metric: {metric}")
                
        except Exception as e:
            logHandler.error_handler(e, "calculate_distance")
            return float('inf')

    # Funzione potenzialmente utile per la realizzazione della soglia adattiva (non usata ma per ora lasciarla)
    def calculate_intra_family_distance(self, family_models: List[Model]) -> float:
        """
        Calculate average intra-family distance.
        """
        try:
            if len(family_models) < 2:
                return 0.0
            
            # Load weights for all models
            model_weights = {}
            for model in family_models:
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
                    dist = self.calculate_distance(
                        model_weights[model_ids[i]],
                        model_weights[model_ids[j]]
                    )
                    distances.append(dist)
            
            return np.mean(distances) if distances else 0.0
            
        except Exception as e:
            logHandler.error_handler(f"Error calculating intra-family distance: {e}", "calculate_intra_family_distance")
            return 0.0