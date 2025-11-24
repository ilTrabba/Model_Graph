"""
MoTHer Algorithm Utilities
Based on the paper "Unsupervised Model Tree Heritage Recovery" (ICLR 2025)
"""

from datetime import datetime, timezone
import logging
import numpy as np
import torch
import safetensors
import networkx as nx

from numpy.typing import NDArray
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple
from src.log_handler import logHandler
from src.services.neo4j_service import neo4j_service

logger = logging.getLogger(__name__)

# Aggiungi questa costante a livello di modulo (top del file)
# Lista completa di pattern per layer da ESCLUDERE dal calcolo L2

EXCLUDED_LAYER_PATTERNS_FULL_MODEL = frozenset([
    # Solo normalization layers
    'layernorm', 'layer_norm', 'ln_', '.ln.', 
    'batchnorm', 'batch_norm', 'bn', '.bn.',
    'groupnorm', 'group_norm', 'gn',
    'instancenorm', 'instance_norm',
    'rmsnorm', 'rms_norm',
])

EXCLUDED_LAYER_PATTERNS = frozenset([
    # Normalization layers
    'layernorm', 'layer_norm', 'ln_', '.ln.', 
    'batchnorm', 'batch_norm', 'bn', '.bn.',
    'groupnorm', 'group_norm', 'gn',
    'instancenorm', 'instance_norm',
    'rmsnorm', 'rms_norm',
    
    # Embedding layers
    'embed', 'embedding', 'embeddings',
    'position_embed', 'positional_embed', 'pos_embed',
    'token_embed', 'word_embed',
    'patch_embed',
    'wte', 'wpe',  # GPT-style embeddings
    
    # Head/Output layers
    'lm_head', 'language_model_head',
    'classifier', 'classification_head',
    'head', 
    'output_projection', 'output_proj',
    'score', 'scorer',
    'cls', 'pooler',
    'prediction_head', 'pred_head',
    'qa_outputs', 
    'seq_relationship',
    
    # Additional output-specific patterns
    'logits',
    'final_layer',
    'output_layer'
])

EXCLUDED_LAYER_PATTERNS_BACKBONE_EMBEDDING = frozenset([
    # Normalization layers
    'layernorm', 'layer_norm', 'ln_', '.ln.', 
    'batchnorm', 'batch_norm', 'bn', '.bn.',
    'groupnorm', 'group_norm', 'gn',
    'instancenorm', 'instance_norm',
    'rmsnorm', 'rms_norm',
    
    # Head/Output layers
    'lm_head', 'language_model_head',
    'classifier', 'classification_head',
    'head', 
    'output_projection', 'output_proj',
    'score', 'scorer',
    'cls', 'pooler',
    'prediction_head', 'pred_head',
    'qa_outputs', 
    'seq_relationship',
    'logits',
    'final_layer',
    'output_layer',
])

EXCLUDED_LAYER_PATTERNS_BACKBONE_HEAD = frozenset([
    # Normalization layers
    'layernorm', 'layer_norm', 'ln_', '.ln.', 
    'batchnorm', 'batch_norm', 'bn', '.bn.',
    'groupnorm', 'group_norm', 'gn',
    'instancenorm', 'instance_norm',
    'rmsnorm', 'rms_norm',
    
    # Embedding layers
    'embed', 'embedding', 'embeddings',
    'position_embed', 'positional_embed', 'pos_embed',
    'token_embed', 'word_embed',
    'patch_embed',
    'wte', 'wpe',
])

EXCLUDED_LAYER_PATTERNS_EMBEDDING_ONLY = frozenset([
    # Normalization layers
    'layernorm', 'layer_norm', 'ln_', '.ln.', 
    'batchnorm', 'batch_norm', 'bn', '.bn.',
    'groupnorm', 'group_norm', 'gn',
    'instancenorm', 'instance_norm',
    'rmsnorm', 'rms_norm',
    
    # Head/Output layers
    'lm_head', 'language_model_head',
    'classifier', 'classification_head',
    'head', 
    'output_projection', 'output_proj',
    'score', 'scorer',
    'cls', 'pooler',
    'prediction_head', 'pred_head',
    'qa_outputs', 
    'seq_relationship',
    'logits',
    'final_layer',
    'output_layer',
    
    # Backbone/Attention layers
    'attention', 'attn', 'self_attn', 'cross_attn',
    'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj',
    'dense', 'intermediate', 'output',
    'mlp', 'ffn', 'feed_forward',
    'conv', 'convolution',
    'linear',
    'layer.', 'layers.',
])

EXCLUDED_LAYER_PATTERNS_HEAD_ONLY = frozenset([
    # Normalization layers
    'layernorm', 'layer_norm', 'ln_', '.ln.', 
    'batchnorm', 'batch_norm', 'bn', '.bn.',
    'groupnorm', 'group_norm', 'gn',
    'instancenorm', 'instance_norm',
    'rmsnorm', 'rms_norm',
    
    # Embedding layers
    'embed', 'embedding', 'embeddings',
    'position_embed', 'positional_embed', 'pos_embed',
    'token_embed', 'word_embed',
    'patch_embed',
    'wte', 'wpe',
    
    # Backbone/Attention layers
    'attention', 'attn', 'self_attn', 'cross_attn',
    'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj',
    'dense', 'intermediate', 'output',
    'mlp', 'ffn', 'feed_forward',
    'conv', 'convolution',
    'linear',
    'layer.', 'layers.',  # generic layer indexing
])

# da spulciare bene (forse da eliminare)
def normalize_parent_child_orientation(tree: nx.DiGraph) -> nx.DiGraph:
    """
    Ensure edges are oriented parent -> child.
    If the tree has no nodes with in_degree == 0 but has sinks (out_degree == 0),
    it likely means edges are child -> parent; in that case, reverse it.
    """
    if tree is None or tree.number_of_nodes() == 0:
        return tree
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    sinks = [n for n in tree.nodes if tree.out_degree(n) == 0]
    if len(roots) == 0 and len(sinks) >= 1:
        return nx.reverse(tree, copy=True)
    return tree

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
        logHandler.error_handler(e, "load_model_weights", {"file_path": file_path})

def calc_ku(weights: Dict[str, Any]) -> float:
    """
    Calculate kurtosis of model weights (only 2D square tensors).
    
    Computes the kurtosis (excess kurtosis using Fisher definition) of weight
    distributions, excluding normalization, embedding, and head layers.
    Only considers square 2D matrices (e.g., attention projection matrices).
    
    Args:
        weights: Dictionary of layer names to tensors (already normalized names)
        
    Returns:
        Kurtosis value (float), or 0.0 if no valid weights found or error occurs
        
    Note:
        - Excludes layers matching EXCLUDED_LAYER_PATTERNS (case-insensitive)
        - Only includes 2D square matrices (shape[0] == shape[1])
        - Uses Fisher definition (excess kurtosis, normal distribution = 0)
    """
    try:
        all_weights = []
        excluded_count = 0
        shape_filtered_count = 0
        
        for param_name, param_tensor in weights.items():

            # Convert to lowercase for case-insensitive matching
            param_lower = param_name.lower()
            
            # Exclude normalization, embedding, and head layers
            if any(pattern in param_lower for pattern in EXCLUDED_LAYER_PATTERNS):
                excluded_count += 1
                continue
            
            # Verify it's a tensor
            if not isinstance(param_tensor, torch.Tensor):
                continue
            
            # Only include 2D square matrices
            if not (param_tensor.ndim == 2 and param_tensor.shape[0] == param_tensor.shape[1]):
                shape_filtered_count += 1
                continue
            
            # Flatten and add to collection
            param_weights = param_tensor.detach().cpu().numpy().flatten()
            all_weights.extend(param_weights)
        
        if not all_weights:
            logger.warning(
                f"No valid weights found for kurtosis calculation. "
                f"Excluded: {excluded_count}, Shape filtered: {shape_filtered_count}"
            )
            return 0.0
        
        # Convert to numpy array
        all_weights = np.array(all_weights)
        
        # Calculate kurtosis (using Fisher definition, excess kurtosis)
        kurt = stats.kurtosis(all_weights, fisher=True)
        
        # Handle NaN/inf cases
        if np.isnan(kurt) or np.isinf(kurt):
            logger.warning("Kurtosis calculation resulted in NaN or Inf")
            return 0.0
        
        logger.debug(
            f"Kurtosis calculated: {kurt:.4f} "
            f"({len(all_weights)} weights from square matrices, "
            f"{excluded_count} layers excluded, {shape_filtered_count} shape filtered)"
        )
        
        return float(kurt)
        
    except Exception as e:
        logger.error(f"Error calculating kurtosis: {e}", exc_info=True)
        return 0.0

def compute_lambda(distance_matrix: np.ndarray, c: float = 0.3) -> float:
    """
    Compute lambda as defined in the MoTHer paper:
    
        λ = c * (1/n^2) * Σ_{i,j} D_ij
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Matrix of pairwise distances between models (n x n).
    c : float, optional
        Scaling constant, default = 0.3 as in the paper.
    
    Returns
    -------
    float
        Value of λ.
    """
    n = distance_matrix.shape[0]
    mean_distance = np.sum(distance_matrix) / (n * n)
    lam = c * mean_distance
    return lam

def update_family_statistics(family_id: str, distance_matrix: NDArray[np.float64], edge_list: List[Tuple[int, int]]) -> None:
    """
    Update family statistics based on the current distance matrix and selected edges.
    """
    try:
        
        total_distance = 0.0
        num_nodes = distance_matrix.shape[0]

        for i, j in edge_list:
            total_distance += distance_matrix[i, j]

        avg_distance = total_distance / (num_nodes - 1) if num_nodes > 1 else 0.0

        # Update family in Neo4j
        updates = {
            'member_count': num_nodes,
            'avg_intra_distance': avg_distance,
            'updated_at': datetime.now(timezone.utc)
        }
        neo4j_service.create_or_update_family({
            'id': family_id,
            **updates
        })

        logger.info(f"Updated statistics for family {family_id}: {num_nodes} members, avg_distance: {avg_distance:.4f}")

    except Exception as e:
        logHandler.error_handler(f"Error updating distance matrix: {e}", "distance_matrix_updates")

############################# FUNZIONI DI SUPPORTO CHE ANDRANNO FIXATE (NON IMPORTANTI) ##################################

def fallback_directed_mst(G: nx.DiGraph) -> nx.DiGraph:
    """
    Fallback algorithm using greedy approach for directed MST
    RIPRISTINO IMPLEMENTAZIONE ORIGINALE
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
    RIPRISTINO IMPLEMENTAZIONE ORIGINALE
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
                
                # CORREZIONE: Parent should have HIGHER kurtosis than child
                kurtosis_diff = ku_values[parent] - ku_values[node]
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