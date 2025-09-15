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
    
def build_tree(ku_values: List[float], 
               distance_matrix: np.ndarray, 
               lambda_param: float) -> Tuple[nx.DiGraph, Dict[int, float]]:
    """
    Build directed tree using MoTHer algorithm with Chu-Liu-Edmonds MDST
    
    Args:
        ku_values: Kurtosis values for each model
        distance_matrix: Distance matrix between models (n x n)
        lambda_param: Balance parameter (0.0 = distance only, 1.0 = kurtosis only)
        
    Returns:
        Tuple of:
        - directed_tree: NetworkX DiGraph with node indices
        - confidence_scores: Dict mapping node_index -> confidence score
    """
    n = len(ku_values)
    
    if n < 2:
        logger.warning("Cannot build tree with fewer than 2 models")
        return nx.DiGraph(), {}
    
    if n == 2:
        # Special case: only two models, create simple parent->child based on kurtosis
        if ku_values[0] < ku_values[1]:
            parent, child = 0, 1
        else:
            parent, child = 1, 0
            
        tree = nx.DiGraph()
        tree.add_edge(parent, child)
        return tree, {parent: 0.8, child: 0.7}
    
    logger.debug(f"Building tree with {n} models using Chu-Liu-Edmonds algorithm")
    
    # Create weighted directed graph
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    # Calculate edge weights combining distance and kurtosis
    for i in range(n):
        for j in range(n):
            if i != j:
                # Distance component (normalized)
                distance_cost = distance_matrix[i, j]
                
                # Kurtosis component - lower kurtosis models are better parents
                kurtosis_diff = ku_values[j] - ku_values[i]  # Cost for i->j edge
                
                # If i has lower kurtosis than j, this is a good parent->child relationship
                if kurtosis_diff > 0:
                    kurtosis_cost = -abs(kurtosis_diff)  # Negative cost = preferred
                else:
                    kurtosis_cost = abs(kurtosis_diff) * 2  # Penalty for bad direction
                
                # Combine costs using lambda parameter
                edge_weight = lambda_param * kurtosis_cost + (1 - lambda_param) * distance_cost
                
                G.add_edge(i, j, weight=edge_weight)
    
    # Apply Chu-Liu-Edmonds algorithm for Minimum Directed Spanning Tree
    try:
        mdst = chu_liu_edmonds_algorithm(G)
        logger.debug(f"Chu-Liu-Edmonds completed: {mdst.number_of_nodes()} nodes, {mdst.number_of_edges()} edges")
    except Exception as e:
        logger.warning(f"Chu-Liu-Edmonds failed ({e}), using fallback")
        mdst = fallback_directed_mst(G)
    
    # Calculate confidence scores
    confidence_scores = calculate_confidence_scores(mdst, G, ku_values)
    
    return mdst, confidence_scores


def chu_liu_edmonds_algorithm(G: nx.DiGraph) -> nx.DiGraph:
    """
    Chu-Liu-Edmonds algorithm for Minimum Directed Spanning Tree
    
    Implements the algorithm as described in the paper:
    1. Find minimum incoming edge for each node
    2. Detect cycles in the resulting graph
    3. Contract cycles by removing heaviest edge
    4. Repeat until no cycles remain
    """
    if G.number_of_nodes() <= 1:
        return G.copy()
    
    # Step 1: Find minimum incoming edge for each node
    def find_min_incoming_edges(graph):
        min_edges = {}
        for node in graph.nodes():
            min_weight = float('inf')
            min_edge = None
            
            for pred in graph.predecessors(node):
                weight = graph[pred][node]['weight']
                if weight < min_weight:
                    min_weight = weight
                    min_edge = (pred, node)
            
            if min_edge is not None:
                min_edges[node] = min_edge
        
        return min_edges
    
    # Step 2: Build candidate tree with minimum incoming edges
    def build_candidate_tree(graph, min_edges):
        candidate = nx.DiGraph()
        candidate.add_nodes_from(graph.nodes())
        
        for target_node, (source_node, _) in min_edges.items():
            if graph.has_edge(source_node, target_node):
                candidate.add_edge(source_node, target_node, 
                                 weight=graph[source_node][target_node]['weight'])
        
        return candidate
    
    # Step 3: Detect and resolve cycles
    def resolve_cycles(candidate_tree):
        max_iterations = len(candidate_tree.nodes()) * 2
        iteration = 0
        
        while iteration < max_iterations:
            try:
                # Find simple cycles
                cycles = list(nx.simple_cycles(candidate_tree))
                
                if not cycles:
                    break  # No more cycles
                
                # Process the first cycle found
                cycle = cycles[0]
                if len(cycle) < 2:
                    iteration += 1
                    continue
                
                # Find the heaviest edge in the cycle
                max_weight = float('-inf')
                heaviest_edge = None
                
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    
                    if candidate_tree.has_edge(u, v):
                        weight = candidate_tree[u][v]['weight']
                        if weight > max_weight:
                            max_weight = weight
                            heaviest_edge = (u, v)
                
                # Remove the heaviest edge
                if heaviest_edge is not None:
                    candidate_tree.remove_edge(heaviest_edge[0], heaviest_edge[1])
                    logger.debug(f"Removed edge {heaviest_edge} with weight {max_weight}")
                
                iteration += 1
                
            except (nx.NetworkXError, nx.NetworkXNoCycle):
                break
        
        return candidate_tree
    
    # Step 4: Ensure connectivity
    def ensure_connectivity(tree, original_graph):
        """Add edges to ensure weak connectivity if needed"""
        components = list(nx.weakly_connected_components(tree))
        
        if len(components) <= 1:
            return tree
        
        # Connect components with minimum weight edges
        connected_tree = tree.copy()
        
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            
            min_weight = float('inf')
            best_edge = None
            
            # Find minimum weight edge between components
            for u in comp1:
                for v in comp2:
                    if original_graph.has_edge(u, v):
                        weight = original_graph[u][v]['weight']
                        if weight < min_weight:
                            min_weight = weight
                            best_edge = (u, v)
                    
                    if original_graph.has_edge(v, u):
                        weight = original_graph[v][u]['weight']
                        if weight < min_weight:
                            min_weight = weight
                            best_edge = (v, u)
            
            # Add the best connecting edge
            if best_edge is not None:
                u, v = best_edge
                connected_tree.add_edge(u, v, weight=original_graph[u][v]['weight'])
        
        return connected_tree
    
    # Execute the algorithm
    current_graph = G.copy()
    
    # Main iteration
    for main_iteration in range(len(G.nodes())):
        min_edges = find_min_incoming_edges(current_graph)
        
        if not min_edges:
            break
            
        candidate_tree = build_candidate_tree(current_graph, min_edges)
        resolved_tree = resolve_cycles(candidate_tree)
        
        # Check if we have a valid tree
        if resolved_tree.number_of_edges() >= len(G.nodes()) - 1:
            final_tree = ensure_connectivity(resolved_tree, G)
            return final_tree
        
        current_graph = resolved_tree
    
    # Fallback: ensure we return something reasonable
    return ensure_connectivity(current_graph, G)


def fallback_directed_mst(G: nx.DiGraph) -> nx.DiGraph:
    """
    Fallback algorithm using greedy approach for directed MST
    """
    logger.debug("Using fallback directed MST algorithm")
    
    # Sort edges by weight
    edges = [(u, v, data['weight']) for u, v, data in G.edges(data=True)]
    edges.sort(key=lambda x: x[2])
    
    result = nx.DiGraph()
    result.add_nodes_from(G.nodes())
    
    # Greedily add edges, avoiding cycles
    for u, v, weight in edges:
        # Temporarily add the edge
        result.add_edge(u, v, weight=weight)
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(result))
            if cycles:
                # Remove the edge if it creates a cycle
                result.remove_edge(u, v)
        except:
            # Keep the edge if cycle detection fails
            pass
        
        # Stop when we have enough edges for a spanning tree
        if result.number_of_edges() >= len(G.nodes()) - 1:
            break
    
    return result


def calculate_confidence_scores(tree: nx.DiGraph, original_graph: nx.DiGraph, 
                              ku_values: List[float]) -> Dict[int, float]:
    """
    Calculate confidence scores for each node based on tree structure and kurtosis
    """
    confidence_scores = {}
    
    if tree.number_of_edges() == 0:
        # No edges, return default confidence for all nodes
        for node in tree.nodes():
            confidence_scores[node] = 0.5
        return confidence_scores
    
    # Get weight statistics for normalization
    all_weights = [data['weight'] for _, _, data in original_graph.edges(data=True)]
    if not all_weights:
        for node in tree.nodes():
            confidence_scores[node] = 0.5
        return confidence_scores
    
    min_weight = min(all_weights)
    max_weight = max(all_weights)
    weight_range = max_weight - min_weight
    
    for node in tree.nodes():
        predecessors = list(tree.predecessors(node))
        
        if not predecessors:
            # Root node - high confidence
            confidence_scores[node] = 0.85
        else:
            # Child node - confidence based on parent relationship quality
            parent = predecessors[0]  # Should only have one parent in a tree
            
            if tree.has_edge(parent, node):
                edge_weight = tree[parent][node]['weight']
                
                # Normalize weight to confidence (lower weight = higher confidence)
                if weight_range > 0:
                    normalized_weight = (edge_weight - min_weight) / weight_range
                    weight_confidence = 1.0 - normalized_weight
                else:
                    weight_confidence = 0.5
                
                # Kurtosis confidence (parent should have lower kurtosis)
                kurtosis_diff = ku_values[node] - ku_values[parent]
                if kurtosis_diff > 0:
                    kurtosis_confidence = min(1.0, kurtosis_diff * 2)  # Good parent-child relationship
                else:
                    kurtosis_confidence = 0.3  # Questionable relationship
                
                # Combined confidence
                confidence = (weight_confidence + kurtosis_confidence) / 2
                confidence_scores[node] = max(0.1, min(0.95, confidence))
            else:
                confidence_scores[node] = 0.4  # Default for orphaned nodes
    
    return confidence_scores

'''
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
                    kurtosis_diff = ku_normalized[j] - ku_normalized[i]  # i->j
                    ku_cost = kurtosis_diff if kurtosis_diff > 0 else abs(kurtosis_diff) * 2
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

'''