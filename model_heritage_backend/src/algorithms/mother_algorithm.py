"""
MoTHer Algorithm Main Implementation
Replaces the find_parent_stub function with sophisticated heritage detection
"""

import logging
import numpy as np
import networkx as nx
from typing import Tuple, Optional, List, Dict, Any

from src.services.neo4j_service import neo4j_service
from src.algorithms.mother_utils import (
    load_model_weights, calc_ku, calculate_l2_distance, build_tree
)

logger = logging.getLogger(__name__)

def find_model_parent_mother(model_proxy, family_id: str) -> Tuple[Optional[str], float]:
    """
    Find parent model using MoTHer algorithm
    
    Args:
        model_proxy: The ModelProxy object to find parent for
        family_id: Family ID to search within
        
    Returns:
        Tuple of (parent_id, confidence_score) or (None, 0.0)
    """
    try:
        # Get all models in the family (excluding the current model)
        family_models_data = neo4j_service.get_family_models(family_id)
        
        # Filter to only OK status models and exclude current model
        family_models_data = [
            m for m in family_models_data 
            if m.get('status') == 'ok' and m.get('id') != model_proxy.id
        ]
        
        if not family_models_data:
            logger.info(f"No models found in family {family_id}")
            return None, 0.0
            
        if len(family_models_data) == 1:
            # Only one candidate, return it with moderate confidence
            return family_models_data[0]['id'], 0.7
        
        # Load weights for all models (including current model)
        from src.models.model import ModelProxy
        all_models_data = [model_proxy.to_dict()] + family_models_data
        model_weights = {}
        valid_models = []
        
        for m_data in all_models_data:
            file_path = m_data.get('file_path')
            if file_path:
                weights = load_model_weights(file_path)
                if weights is not None:
                    model_weights[m_data['id']] = weights
                    valid_models.append(m_data)
                else:
                    logger.warning(f"Failed to load weights for model {m_data['id']}")
            else:
                logger.warning(f"No file_path for model {m_data['id']}")
        
        if len(valid_models) < 2:
            logger.warning(f"Insufficient valid models for MoTHer analysis. Falling back to parameter similarity.")
            return _fallback_parameter_similarity(model_proxy.to_dict(), family_models_data)
        
        # Calculate kurtosis for each model
        kurtosis_values = []
        model_ids = []
        
        for m_data in valid_models:
            ku = calc_ku(model_weights[m_data['id']])
            kurtosis_values.append(ku)
            model_ids.append(m_data['id'])
            logger.info(f"Model {m_data['id']} kurtosis: {ku:.4f}")
        
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
            current_model_idx = model_ids.index(model_proxy.id)
        except ValueError:
            logger.error(f"Current model {model_proxy.id} not found in valid models")
            return _fallback_parameter_similarity(model_proxy.to_dict(), family_models_data)
        
        # Get parent from tree
        predecessors = list(tree.predecessors(current_model_idx))
        
        if predecessors:
            parent_idx = predecessors[0]
            parent_id = model_ids[parent_idx]
            confidence = confidence_scores.get(current_model_idx, 0.5)
            
            logger.info(f"MoTHer found parent {parent_id} for model {model_proxy.id} with confidence {confidence:.3f}")
            return parent_id, confidence
        else:
            # Current model is likely a root node
            logger.info(f"Model {model_proxy.id} appears to be a root model (no parent found)")
            return None, 0.0
            
    except Exception as e:
        logger.error(f"MoTHer algorithm failed for model {model_proxy.id}: {e}")
        # Fallback to original parameter similarity method
        return _fallback_parameter_similarity(model_proxy.to_dict(), family_models_data)

def _fallback_parameter_similarity(model_data: dict, candidates_data: List[dict]) -> Tuple[Optional[str], float]:
    """
    Fallback method using parameter count similarity
    This is the original stub logic
    """
    try:
        if not candidates_data:
            return None, 0.0
            
        # Find closest by parameter count
        closest_model = None
        min_diff = float('inf')
        model_params = model_data.get('total_parameters', 0)
        
        for candidate in candidates_data:
            if candidate.get('id') != model_data.get('id'):
                candidate_params = candidate.get('total_parameters', 0)
                diff = abs(candidate_params - model_params)
                if diff < min_diff:
                    min_diff = diff
                    closest_model = candidate
        
        if closest_model:
            # Mock confidence based on parameter similarity
            closest_params = closest_model.get('total_parameters', 0)
            max_params = max(model_params, closest_params)
            confidence = 1.0 - (min_diff / max_params) if max_params > 0 else 0.0
            confidence = max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
            
            logger.info(f"Fallback method found parent {closest_model['id']} for model {model_data['id']} with confidence {confidence:.3f}")
            return closest_model['id'], confidence
        
        return None, 0.0
        
    except Exception as e:
        logger.error(f"Fallback method failed: {e}")
        return None, 0.0