"""
MoTHer Algorithm Utilities
Based on the paper "Unsupervised Model Tree Heritage Recovery" (ICLR 2025)
"""

import os
import logging
import numpy as np
import torch
import safetensors
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx

logger = logging.getLogger(__name__)

def _get_layer_kinds() -> List[str]:
    """Get layer types for analysis"""
    return [
        # Transformer attention layers
        'self_attn.q_proj',
        'self_attn.k_proj', 
        'self_attn.v_proj',
        'self_attn.o_proj',
        'mlp.gate_proj',
        'mlp.up_proj',
        'mlp.down_proj',
        'attention.q_proj',
        'attention.k_proj',
        'attention.v_proj',
        'attention.out_proj',
        'feed_forward.w1',
        'feed_forward.w2',
        'feed_forward.w3',
        # Common dense/linear layers
        'linear1',
        'linear2',
        'dense',
        'fc1',
        'fc2',
        # Generic layer patterns (for simple models)
        '.weight',  # Any layer with weights
        '.bias'     # Any layer with bias
    ]

def load_model_weights(file_path: str) -> Optional[Dict[str, Any]]:
    """Load model weights from file"""
    try:
        if file_path.endswith('.safetensors'):
            with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
                weights = {key: f.get_tensor(key) for key in f.keys()}
        elif file_path.endswith(('.pt', '.pth', '.bin')):
            weights = torch.load(file_path, map_location='cpu')
            # Handle state_dict format
            if isinstance(weights, dict) and 'state_dict' in weights:
                weights = weights['state_dict']
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return None
            
        return weights
    except Exception as e:
        logger.error(f"Failed to load model weights from {file_path}: {e}")
        return None

def calc_ku(weights: Dict[str, Any], layer_kind: Optional[str] = None) -> float:
    """Calculate kurtosis of model weights"""
    try:
        layer_kinds = _get_layer_kinds()
        all_weights = []
        
        for param_name, param_tensor in weights.items():
            # Filter by layer kind if specified
            if layer_kind:
                if not any(lk in param_name for lk in [layer_kind]):
                    continue
            else:
                # Include all weight and bias parameters for analysis
                # More inclusive approach for simple models
                if not any(lk in param_name for lk in layer_kinds) and 'weight' not in param_name and 'bias' not in param_name:
                    continue
                    
            if isinstance(param_tensor, torch.Tensor):
                # Flatten and convert to numpy
                param_weights = param_tensor.detach().cpu().numpy().flatten()
                all_weights.extend(param_weights)
        
        if not all_weights:
            logger.warning(f"No weights found for analysis in model parameters: {list(weights.keys())}")
            return 0.0
            
        all_weights = np.array(all_weights)
        
        # Calculate kurtosis (using Fisher definition, excess kurtosis)
        kurt = stats.kurtosis(all_weights, fisher=True)
        
        # Handle NaN/inf cases
        if np.isnan(kurt) or np.isinf(kurt):
            return 0.0
            
        return float(kurt)
        
    except Exception as e:
        logger.error(f"Error calculating kurtosis: {e}")
        return 0.0

def calculate_l2_distance(weights1: Dict[str, Any], weights2: Dict[str, Any]) -> float:
    """Calculate L2 distance between two sets of model weights"""
    try:
        layer_kinds = _get_layer_kinds()
        total_distance = 0.0
        param_count = 0
        
        # Get common parameters
        common_params = set(weights1.keys()) & set(weights2.keys())
        
        for param_name in common_params:
            # Include all weight and bias parameters for distance calculation
            should_include = False
            for lk in layer_kinds:
                if lk in param_name:
                    should_include = True
                    break
            
            # Also include if it's a weight or bias parameter
            if 'weight' in param_name or 'bias' in param_name:
                should_include = True
                
            if not should_include:
                continue
                
            tensor1 = weights1[param_name]
            tensor2 = weights2[param_name]
            
            if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
                # Ensure same shape
                if tensor1.shape != tensor2.shape:
                    continue
                    
                # Calculate L2 distance
                diff = tensor1.detach().cpu().numpy() - tensor2.detach().cpu().numpy()
                l2_dist = np.linalg.norm(diff.flatten())
                total_distance += l2_dist
                param_count += 1
        
        if param_count == 0:
            return float('inf')
            
        # Return average L2 distance
        return total_distance / param_count
        
    except Exception as e:
        logger.error(f"Error calculating L2 distance: {e}")
        return float('inf')

def _find_min_weighted_directed_tree(distance_matrix: np.ndarray) -> nx.DiGraph:
    """Find minimum weighted directed spanning tree"""
    try:
        n = distance_matrix.shape[0]
        if n <= 1:
            return nx.DiGraph()
            
        # Create directed graph from distance matrix
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n):
            G.add_node(i)
            
        # Add edges with weights (distances)
        for i in range(n):
            for j in range(n):
                if i != j and not np.isinf(distance_matrix[i, j]):
                    G.add_edge(i, j, weight=distance_matrix[i, j])
        
        # Find minimum spanning arborescence (directed MST)
        try:
            mst = nx.minimum_spanning_arborescence(G)
            return mst
        except nx.NetworkXException:
            # Fallback: create a simple tree structure
            mst = nx.DiGraph()
            for i in range(n):
                mst.add_node(i)
            
            # Connect to nearest neighbor for each node (except first)
            for i in range(1, n):
                min_dist = float('inf')
                best_parent = 0
                for j in range(i):
                    if distance_matrix[j, i] < min_dist:
                        min_dist = distance_matrix[j, i]
                        best_parent = j
                mst.add_edge(best_parent, i, weight=min_dist)
            
            return mst
            
    except Exception as e:
        logger.error(f"Error finding minimum directed tree: {e}")
        return nx.DiGraph()

def build_tree(ku_values: List[float], distance_matrix: np.ndarray, 
               lambda_param: float = 0.5, ground_truth: Optional[List] = None, 
               rev: bool = False) -> Tuple[nx.DiGraph, Dict[int, float]]:
    """Build heritage tree from kurtosis and distance matrices"""
    try:
        n = len(ku_values)
        if n <= 1:
            return nx.DiGraph(), {}
            
        # Create combined cost matrix
        ku_array = np.array(ku_values)
        
        # Normalize kurtosis values to [0, 1] range
        ku_min, ku_max = ku_array.min(), ku_array.max()
        if ku_max > ku_min:
            ku_normalized = (ku_array - ku_min) / (ku_max - ku_min)
        else:
            ku_normalized = np.zeros_like(ku_array)
            
        # Normalize distance matrix
        dist_flat = distance_matrix[~np.isinf(distance_matrix)]
        if len(dist_flat) > 0:
            dist_min, dist_max = dist_flat.min(), dist_flat.max()
            if dist_max > dist_min:
                dist_normalized = (distance_matrix - dist_min) / (dist_max - dist_min)
                dist_normalized[np.isinf(distance_matrix)] = 1.0
            else:
                dist_normalized = np.zeros_like(distance_matrix)
        else:
            dist_normalized = np.ones_like(distance_matrix)
            
        # Combine kurtosis and distance with lambda weighting
        combined_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Parent should have higher kurtosis (less trained)
                    ku_cost = 1.0 - abs(ku_normalized[i] - ku_normalized[j])
                    dist_cost = dist_normalized[i, j]
                    combined_matrix[i, j] = lambda_param * ku_cost + (1 - lambda_param) * dist_cost
                else:
                    combined_matrix[i, j] = float('inf')
        
        # Find minimum spanning tree
        mst = _find_min_weighted_directed_tree(combined_matrix)
        
        # Calculate confidence scores based on edge weights
        confidence_scores = {}
        for node in mst.nodes():
            predecessors = list(mst.predecessors(node))
            if predecessors:
                edge_data = mst.get_edge_data(predecessors[0], node)
                if edge_data:
                    # Convert cost to confidence (lower cost = higher confidence)
                    confidence = 1.0 - min(1.0, edge_data.get('weight', 1.0))
                    confidence_scores[node] = max(0.1, min(0.9, confidence))
                else:
                    confidence_scores[node] = 0.5
            else:
                confidence_scores[node] = 0.0  # Root node
                
        return mst, confidence_scores
        
    except Exception as e:
        logger.error(f"Error building tree: {e}")
        return nx.DiGraph(), {}