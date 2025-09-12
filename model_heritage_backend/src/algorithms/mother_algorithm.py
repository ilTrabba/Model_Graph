"""
MoTHer Algorithm Main Implementation
Full integration of reconstruct_family_tree_mother replacing direct single-parent logic
"""

import logging
import numpy as np
import networkx as nx
from typing import Tuple, Optional, List, Dict, Any

from src.models.model import Model, ModelQuery
from src.algorithms.mother_utils import (
    load_model_weights, calc_ku, calculate_l2_distance, build_tree
)

logger = logging.getLogger(__name__)
model_query = ModelQuery()

def reconstruct_family_tree_mother(family_id: str) -> Tuple[nx.DiGraph, Dict[str, float]]:
    """
    Reconstruct complete family tree using MoTHer algorithm
    
    Args:
        family_id: Family ID to reconstruct tree for
        
    Returns:
        Tuple of (tree, confidence_scores)
        - tree: NetworkX DiGraph with model IDs as nodes
        - confidence_scores: Dictionary mapping model_id -> confidence score
    """
    try:
        # Get all models in the family
        family_models = model_query.filter_by(family_id=family_id, status='ok').all()
        
        if len(family_models) < 2:
            logger.info(f"Family {family_id} has insufficient models for tree reconstruction")
            return nx.DiGraph(), {}
        
        logger.info(f"Reconstructing tree for family {family_id} with {len(family_models)} models")
        
        # Load weights for all models
        model_weights = {}
        valid_models = []
        
        for model in family_models:
            weights = load_model_weights(model.file_path)
            if weights is not None:
                model_weights[model.id] = weights
                valid_models.append(model)
            else:
                logger.warning(f"Failed to load weights for model {model.id}")
        
        if len(valid_models) < 2:
            logger.warning(f"Insufficient valid models for tree reconstruction. Using fallback.")
            return _fallback_tree_reconstruction(family_id)
        
        # Calculate kurtosis for each model
        kurtosis_values = []
        model_ids = []
        
        for model in valid_models:
            ku = calc_ku(model_weights[model.id])
            kurtosis_values.append(ku)
            model_ids.append(model.id)
            logger.debug(f"Model {model.id} kurtosis: {ku:.4f}")
        
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
        
        # Convert tree with indices to tree with model IDs
        tree_with_ids = nx.DiGraph()
        confidence_with_ids = {}
        
        for i, model_id in enumerate(model_ids):
            tree_with_ids.add_node(model_id)
            confidence_with_ids[model_id] = confidence_scores.get(i, 0.5)
        
        for edge in tree.edges():
            source_id = model_ids[edge[0]]
            target_id = model_ids[edge[1]]
            tree_with_ids.add_edge(source_id, target_id)
        
        logger.info(f"Successfully reconstructed tree for family {family_id} with {tree_with_ids.number_of_nodes()} nodes")
        return tree_with_ids, confidence_with_ids
        
    except Exception as e:
        logger.error(f"Tree reconstruction failed for family {family_id}: {e}")
        return _fallback_tree_reconstruction(family_id)

def get_family_tree_info(family_id: str) -> Dict[str, Any]:
    """
    Get complete family tree information including structure and statistics
    
    Args:
        family_id: Family ID to get info for
        
    Returns:
        Dictionary with tree structure, confidence scores, and statistics
    """
    try:
        tree, confidence_scores = reconstruct_family_tree_mother(family_id)
        
        if tree.number_of_nodes() == 0:
            return {
                'family_id': family_id,
                'tree': {},
                'confidence_scores': {},
                'statistics': {
                    'num_nodes': 0,
                    'num_edges': 0,
                    'num_roots': 0,
                    'num_leaves': 0,
                    'max_depth': 0,
                    'avg_confidence': 0.0
                }
            }
        
        # Calculate statistics
        roots = [node for node in tree.nodes() if tree.in_degree(node) == 0]
        leaves = [node for node in tree.nodes() if tree.out_degree(node) == 0]
        
        # Calculate max depth
        max_depth = 0
        if roots:
            for root in roots:
                try:
                    depths = nx.single_source_shortest_path_length(tree, root)
                    max_depth = max(max_depth, max(depths.values()) if depths else 0)
                except nx.NetworkXError:
                    pass
        
        # Calculate average confidence
        confidences = list(confidence_scores.values())
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Convert tree to serializable format
        tree_data = {
            'nodes': list(tree.nodes()),
            'edges': [(u, v) for u, v in tree.edges()]
        }
        
        return {
            'family_id': family_id,
            'tree': tree_data,
            'confidence_scores': confidence_scores,
            'statistics': {
                'num_nodes': tree.number_of_nodes(),
                'num_edges': tree.number_of_edges(),
                'num_roots': len(roots),
                'num_leaves': len(leaves),
                'max_depth': max_depth,
                'avg_confidence': float(avg_confidence)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get tree info for family {family_id}: {e}")
        return {
            'family_id': family_id,
            'tree': {},
            'confidence_scores': {},
            'statistics': {},
            'error': str(e)
        }

def find_model_parent_mother(model: Model, family_id: str) -> Tuple[Optional[str], float]:
    """
    Find parent model using MoTHer algorithm by deriving from reconstructed tree
    This is the adapter function that maintains backward compatibility
    
    Args:
        model: The model to find parent for
        family_id: Family ID to search within
        
    Returns:
        Tuple of (parent_id, confidence_score) or (None, 0.0)
    """
    try:
        # Reconstruct the full family tree
        tree, confidence_scores = reconstruct_family_tree_mother(family_id)
        
        if tree.number_of_nodes() == 0:
            logger.info(f"No tree reconstructed for family {family_id}")
            return _fallback_parameter_similarity(model, family_id)
        
        if model.id not in tree.nodes():
            logger.warning(f"Model {model.id} not found in reconstructed tree")
            return _fallback_parameter_similarity(model, family_id)
        
        # Get parent from tree
        predecessors = list(tree.predecessors(model.id))
        
        if predecessors:
            parent_id = predecessors[0]  # Should only be one parent in tree
            confidence = confidence_scores.get(model.id, 0.5)
            
            logger.info(f"MoTHer found parent {parent_id} for model {model.id} with confidence {confidence:.3f}")
            return parent_id, confidence
        else:
            # Current model is a root node
            logger.info(f"Model {model.id} appears to be a root model (no parent found)")
            return None, 0.0
            
    except Exception as e:
        logger.error(f"MoTHer algorithm failed for model {model.id}: {e}")
        # Fallback to parameter similarity method
        return _fallback_parameter_similarity(model, family_id)

def _fallback_tree_reconstruction(family_id: str) -> Tuple[nx.DiGraph, Dict[str, float]]:
    """
    Fallback tree reconstruction using parameter similarity when MoTHer fails
    
    Args:
        family_id: Family ID to create fallback tree for
        
    Returns:
        Tuple of (tree, confidence_scores)
    """
    try:
        family_models = model_query.filter_by(family_id=family_id, status='ok').all()
        
        if len(family_models) < 2:
            return nx.DiGraph(), {}
        
        # Sort models by parameter count (higher = likely parent)
        sorted_models = sorted(family_models, key=lambda m: m.total_parameters or 0, reverse=True)
        
        tree = nx.DiGraph()
        confidence_scores = {}
        
        # Add all models as nodes
        for model in sorted_models:
            tree.add_node(model.id)
        
        # Create simple parent-child relationships based on parameter similarity
        for i, model in enumerate(sorted_models[1:], 1):  # Skip first (root)
            # Find most similar model with higher parameter count
            best_parent = None
            best_similarity = 0.0
            
            for potential_parent in sorted_models[:i]:  # Only consider models with higher param count
                similarity = _calculate_parameter_similarity(model, potential_parent)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_parent = potential_parent
            
            if best_parent:
                tree.add_edge(best_parent.id, model.id)
                confidence_scores[model.id] = max(0.1, min(0.9, best_similarity))
            else:
                confidence_scores[model.id] = 0.1
        
        # Root model has no parent
        if sorted_models:
            confidence_scores[sorted_models[0].id] = 0.0
        
        logger.info(f"Fallback tree reconstruction completed for family {family_id}")
        return tree, confidence_scores
        
    except Exception as e:
        logger.error(f"Fallback tree reconstruction failed for family {family_id}: {e}")
        return nx.DiGraph(), {}

def _fallback_parameter_similarity(model: Model, family_id: str) -> Tuple[Optional[str], float]:
    """
    Fallback method using parameter count similarity
    This is the original stub logic adapted for family_id input
    """
    try:
        family_models = model_query.filter_by(family_id=family_id, status='ok').all()
        candidates = [m for m in family_models if m.id != model.id]
        
        if not candidates:
            return None, 0.0
            
        # Find closest by parameter count
        closest_model = None
        min_diff = float('inf')
        
        for candidate in candidates:
            if candidate.id != model.id:
                diff = abs((candidate.total_parameters or 0) - (model.total_parameters or 0))
                if diff < min_diff:
                    min_diff = diff
                    closest_model = candidate
        
        if closest_model:
            # Mock confidence based on parameter similarity
            max_params = max(model.total_parameters or 0, closest_model.total_parameters or 0)
            confidence = 1.0 - (min_diff / max_params) if max_params > 0 else 0.0
            confidence = max(0.1, min(0.9, confidence))  # Clamp between 0.1 and 0.9
            
            logger.info(f"Fallback method found parent {closest_model.id} for model {model.id} with confidence {confidence:.3f}")
            return closest_model.id, confidence
        
        return None, 0.0
        
    except Exception as e:
        logger.error(f"Fallback method failed: {e}")
        return None, 0.0

def _calculate_parameter_similarity(model1: Model, model2: Model) -> float:
    """
    Calculate similarity between two models based on parameters
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        params1 = model1.total_parameters or 0
        params2 = model2.total_parameters or 0
        
        if params1 == 0 and params2 == 0:
            return 1.0
        
        max_params = max(params1, params2)
        if max_params == 0:
            return 1.0
        
        diff = abs(params1 - params2)
        similarity = 1.0 - (diff / max_params)
        
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logger.error(f"Parameter similarity calculation failed: {e}")
        return 0.0