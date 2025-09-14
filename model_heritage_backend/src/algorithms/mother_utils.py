"""
MoTHer Algorithm Utilities
Based on the paper "Unsupervised Model Tree Heritage Recovery" (ICLR 2025)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import numpy as np
import safetensors
import torch
from scipy import stats

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
                    G.add_edge(i, j, weight=float(distance_matrix[i, j]))

        # Find minimum spanning arborescence (directed MST)
        mst = nx.minimum_spanning_arborescence(G)
        return mst

    except Exception as e:
        # Fallback: create a simple tree structure if Edmonds fails
        logger.warning(f"Edmonds algorithm failed ({e}); falling back to greedy tree.")
        try:
            n = distance_matrix.shape[0]
            mst = nx.DiGraph()
            for i in range(n):
                mst.add_node(i)
            for i in range(1, n):
                min_dist = float('inf')
                best_parent = 0
                for j in range(i):
                    if distance_matrix[j, i] < min_dist:
                        min_dist = distance_matrix[j, i]
                        best_parent = j
                mst.add_edge(best_parent, i, weight=min_dist)
            return mst
        except Exception as e2:
            logger.error(f"Error finding minimum directed tree (fallback failed): {e2}")
            return nx.DiGraph()


def build_tree(
    ku_values: List[float],
    distance_matrix: np.ndarray,
    lambda_param: float = 0.5,
    ground_truth: Optional[List] = None,  # kept for backward-compat, unused
    rev: bool = False,
    labels: Optional[List[str]] = None,
    sort_desc: bool = True,
) -> Tuple[nx.DiGraph, Dict[int, float]]:
    """
    Build heritage tree from kurtosis and distance matrices.

    Changes:
    - MoTHer-style direction penalty: prefer edges i->j where ku[i] > ku[j].
      We add a penalty to edges that violate this (ku[i] < ku[j]).
    - Deterministic ordering: if `labels` is provided, rows/cols are reordered
      by sorted(labels, reverse=sort_desc) before running MDST (Edmonds).
      The exact order used is stored in mst.graph['sorted_labels'] and
      mst.graph['order'] for safe post-hoc mapping of indices to labels.

    Parameters:
    - ku_values: list of kurtosis values, aligned with distance_matrix rows/columns.
    - distance_matrix: square (n x n) numpy array of distances (base costs).
    - lambda_param: lambda controlling the strength of direction penalty term.
    - ground_truth: unused placeholder (compatibility).
    - rev: if True, reverse the preferred direction (penalize ku[i] > ku[j]).
    - labels: optional list of labels (e.g., model IDs) aligned to inputs.
    - sort_desc: when labels provided, sort order is descending if True, else ascending.

    Returns:
    - mst: directed minimum arborescence over indices (0..n-1) of the possibly
           reordered matrix. If labels were provided, use mst.graph['sorted_labels']
           to map indices back to labels safely.
    - confidence_scores: per-node confidence based on inbound edge cost (lower cost -> higher confidence).
    """
    try:
        n = len(ku_values)
        if n <= 1:
            mst = nx.DiGraph()
            if n == 1:
                mst.add_node(0)
            return mst, {}

        # Ensure numpy arrays
        ku = np.array(ku_values, dtype=float)
        dist = np.array(distance_matrix, dtype=float)
        if dist.shape != (n, n):
            raise ValueError("distance_matrix must be square with shape (n, n)")

        # Direction penalty mask T:
        # Not rev: prefer i->j if ku[i] > ku[j]; penalize when ku[i] < ku[j] (T=1).
        # Rev: inverse.
        if not rev:
            T = (ku.reshape(-1, 1) < ku.reshape(1, -1)).astype(float)
        else:
            T = (ku.reshape(-1, 1) > ku.reshape(1, -1)).astype(float)

        # Mean of finite distances to scale penalty (as in MoTHer)
        finite = dist[np.isfinite(dist)]
        mean_dist = float(np.mean(finite)) if finite.size > 0 else 1.0

        # Combined cost KD = dist + lambda * mean_dist * T + inf on diagonal
        KD = dist + (lambda_param * mean_dist * T)
        KD = KD.copy()
        np.fill_diagonal(KD, float('inf'))

        # Deterministic ordering if labels provided
        used_labels = None
        used_order = None
        if labels is not None:
            labels_arr = np.array(labels)
            order = np.argsort(labels_arr)
            if sort_desc:
                order = order[::-1]
            KD = KD[order][:, order]
            ku = ku[order]
            used_order = order.tolist()
            used_labels = [labels[i] for i in order]

        # Compute MDST on KD
        mst = _find_min_weighted_directed_tree(KD)

        # Attach metadata for safe mapping (does not change return type)
        if used_labels is not None:
            mst.graph['sorted_labels'] = used_labels
        if used_order is not None:
            mst.graph['order'] = used_order

        # Confidence per node: 1 - normalized inbound cost (clamped)
        denom = mean_dist + 1e-8
        confidence_scores: Dict[int, float] = {}
        for node in mst.nodes():
            preds = list(mst.predecessors(node))
            if preds:
                w = float(mst.get_edge_data(preds[0], node).get('weight', mean_dist))
                conf = 1.0 - max(0.0, min(1.0, w / denom))
                confidence_scores[node] = max(0.05, min(0.95, conf))
            else:
                confidence_scores[node] = 0.0  # root

        return mst, confidence_scores

    except Exception as e:
        logger.error(f"Error building tree: {e}")
        return nx.DiGraph(), {}