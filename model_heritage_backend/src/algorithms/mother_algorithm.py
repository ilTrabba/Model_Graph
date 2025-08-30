"""
MoTHer Algorithm Main Implementation
Replaces the find_parent_stub function with sophisticated heritage detection
"""

import logging
import numpy as np
import networkx as nx
from typing import Tuple, Optional, List, Dict, Any

from src.models.model import Model
from src.algorithms.mother_utils import (
    load_model_weights, calc_ku, calculate_l2_distance, build_tree
)

logger = logging.getLogger(__name__)

def find_model_parent_mother(model: Model, family_id: str) -> Tuple[Optional[str], float]:
    """
    Find parent model using MoTHer algorithm
    
    Args:
        model: The model to find parent for
        family_id: Family ID to search within
        
    Returns:
        Tuple of (parent_id, confidence_score) or (None, 0.0)
    """
    try:
        # Get all models in the family (excluding the current model)
        family_models = Model.query.filter_by(family_id=family_id, status='ok').all()
        
        if not family_models:
            logger.info(f"No models found in family {family_id}")
            return None, 0.0
            
        # Filter out the current model and get candidates
        candidates = [m for m in family_models if m.id != model.id]
        
        if not candidates:
            logger.info(f"No candidate parents found for model {model.id}")
            return None, 0.0
            
        if len(candidates) == 1:
            # Only one candidate, return it with moderate confidence
            return candidates[0].id, 0.7
        
        # Load weights for all models (including current model)
        all_models = [model] + candidates
        model_weights = {}
        valid_models = []
        
        for m in all_models:
            weights = load_model_weights(m.file_path)
            if weights is not None:
                model_weights[m.id] = weights
                valid_models.append(m)
            else:
                logger.warning(f"Failed to load weights for model {m.id}")
        
        if len(valid_models) < 2:
            logger.warning(f"Insufficient valid models for MoTHer analysis. Falling back to parameter similarity.")
            return _fallback_parameter_similarity(model, candidates)
        
        # Calculate kurtosis for each model
        kurtosis_values = []
        model_ids = []
        
        for m in valid_models:
            ku = calc_ku(model_weights[m.id])
            kurtosis_values.append(ku)
            model_ids.append(m.id)
            logger.info(f"Model {m.id} kurtosis: {ku:.4f}")
        
        # Build distance matrix
        n_models = len(valid_models)
        distance_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    dist = calculate_l2_distance(
                        model_weights[model_ids[i]], 
                        model_weights[model_ids[j]]
                    )
                    distance_matrix[i, j] = dist
                else:
                    distance_matrix[i, j] = float('inf')
        
        # Apply MoTHer algorithm
        tree, confidence_scores = build_tree(
            ku_values=kurtosis_values,
            distance_matrix=distance_matrix,
            lambda_param=0.5  # Balance between kurtosis and distance
        )
        
        # Find the current model's index
        try:
            current_model_idx = model_ids.index(model.id)
        except ValueError:
            logger.error(f"Current model {model.id} not found in valid models")
            return _fallback_parameter_similarity(model, candidates)
        
        # Get parent from tree
        predecessors = list(tree.predecessors(current_model_idx))
        
        if predecessors:
            parent_idx = predecessors[0]
            parent_id = model_ids[parent_idx]
            confidence = confidence_scores.get(current_model_idx, 0.5)
            
            logger.info(f"MoTHer found parent {parent_id} for model {model.id} with confidence {confidence:.3f}")
            return parent_id, confidence
        else:
            # Current model is likely a root node
            logger.info(f"Model {model.id} appears to be a root model (no parent found)")
            return None, 0.0
            
    except Exception as e:
        logger.error(f"MoTHer algorithm failed for model {model.id}: {e}")
        # Fallback to original parameter similarity method
        return _fallback_parameter_similarity(model, family_models)

def _fallback_parameter_similarity(model: Model, candidates: List[Model]) -> Tuple[Optional[str], float]:
    """
    Fallback method using parameter count similarity
    This is the original stub logic
    """
    try:
        if not candidates:
            return None, 0.0
            
        # Find closest by parameter count
        closest_model = None
        min_diff = float('inf')
        
        for candidate in candidates:
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
            
            logger.info(f"Fallback method found parent {closest_model.id} for model {model.id} with confidence {confidence:.3f}")
            return closest_model.id, confidence
        
        return None, 0.0
        
    except Exception as e:
        logger.error(f"Fallback method failed: {e}")
        return None, 0.0