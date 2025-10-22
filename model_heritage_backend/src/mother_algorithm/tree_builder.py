"""
TreeBuilder

This module builds genealogical relationships within model families using algorithm approaches (MoTHer).
It leverages the existing MoTHer implementation to create heritage trees within families.
"""

import logging
import numpy as np
import networkx as nx

from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from src.clustering.distance_calculator import ModelDistanceCalculator
from src.db_entities.entity import Model
from src.db_entities.entity import ModelQuery
from src.mother_algorithm.mdst import MDST
from src.mother_algorithm.mother_utils import (
    load_model_weights,
    calc_ku, 
    compute_lambda,
    fallback_directed_mst,
    calculate_confidence_scores,
    _normalize_parent_child_orientation
)

logger = logging.getLogger(__name__)
model_query = ModelQuery()
mdst = MDST()

class TreeBuildingMethod(Enum):
    """Available tree building methods"""
    MOTHER = "mother"          # Full MoTHer algorithm with kurtosis
    DISTANCE_ONLY = "distance" # Distance-based minimum spanning tree only
    KURTOSIS_ONLY = "kurtosis" # Kurtosis-based ordering only

class MoTHerTreeBuilder:
    """
    Build genealogical relationships within model families using the MoTHer algorithm.
    
    Features:
    - Calculate distance matrices between family models
    - Use kurtosis-based direction determination
    - Apply Edmonds' algorithm for minimum directed spanning tree
    - Leverage existing MoTHer algorithm implementation
    """
    
    def __init__(self,
                 lambda_param: float = 0.3, # parametro c, non lambda, da rinominare ovunque
                 method: TreeBuildingMethod = TreeBuildingMethod.MOTHER,
                 distance_calculator: ModelDistanceCalculator = ModelDistanceCalculator()):
        """
        Initialize the tree builder.
        
        Args:
            distance_calculator: Calculator for model distances
            lambda_param: Balance between kurtosis and distance (0=distance only, 1=kurtosis only)
            method: Tree building method to use
        """
        self.lambda_param = lambda_param
        self.method = method
        self.distance_calculator = distance_calculator

        logger.info(f"Initialized MoTHerTreeBuilder with method: {method}, lambda: {lambda_param}")
    
    def build_family_tree(self, 
                         family_id: str,
                         models: Optional[List[Model]] = None) -> Tuple[nx.DiGraph, Dict[str, float]]:
        """
        Build a genealogical tree for all models in a family.
        
        Args:
            family_id: ID of the family to build tree for
            models: Optional list of models (will query if None)
            
        Returns:
            Tuple of (directed_tree, confidence_scores)
            - directed_tree: NetworkX DiGraph with model IDs as nodes (parent -> child)
            - confidence_scores: Dictionary mapping model_id -> confidence score
        """
        try:
            # Get family models if not provided
            if models is None:
                models = model_query.filter_by(
                    family_id=family_id,
                    status='ok'
                ).all()

            # Deterministic ordering independent of insertion timing
            models = sorted(models, key=lambda m: m.id)
            
            if len(models) < 2:
                logger.warning(f"Family {family_id} has insufficient models for tree building")
                return nx.DiGraph(), {}
            
            logger.info(f"Building tree for family {family_id} with {len(models)} models")
            
            # Load model weights
            model_weights: Dict[str, Any] = {}
            valid_models: List[Model] = []
            
            for model in models:
                weights = load_model_weights(model.file_path)
                if weights is not None:
                    model_weights[model.id] = weights
                    valid_models.append(model)
                else:
                    logger.warning(f"Failed to load weights for model {model.id}")
            
            if len(valid_models) < 2:
                logger.warning(f"Insufficient valid models for tree building in family {family_id}")
                return nx.DiGraph(), {}

            # Deterministic ordering also for valid-only set
            valid_models.sort(key=lambda m: m.id)
            
            # Build tree using selected method
            if self.method == TreeBuildingMethod.MOTHER:
                tree, confidence_scores = self._build_mother_tree(valid_models, model_weights)
            elif self.method == TreeBuildingMethod.DISTANCE_ONLY:
                tree, confidence_scores = self._build_distance_tree(valid_models, model_weights)
            elif self.method == TreeBuildingMethod.KURTOSIS_ONLY:
                tree, confidence_scores = self._build_kurtosis_tree(valid_models, model_weights)
            else:
                raise ValueError(f"Unknown tree building method: {self.method}")
            
            # Convert node indices to model IDs
            tree_with_ids = self._convert_tree_to_model_ids(tree, valid_models)
            confidence_with_ids = self._convert_confidence_to_model_ids(confidence_scores, valid_models)

            # Normalize orientation to parent -> child for end-to-end consistency
            tree_with_ids = _normalize_parent_child_orientation(tree_with_ids)
            
            logger.info(f"Built tree for family {family_id}: {tree_with_ids.number_of_nodes()} nodes, {tree_with_ids.number_of_edges()} edges")
            return tree_with_ids, confidence_with_ids
            
        except Exception as e:
            logger.error(f"Error building family tree for {family_id}: {e}")
            return nx.DiGraph(), {}
    
    def _build_mother_tree(self, 
                          models: List[Model], 
                          model_weights: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict[int, float]]:
        """
        Build tree using full MoTHer algorithm (kurtosis + distance).
        """
        try:
            # Calculate kurtosis for each model
            kurtosis_values = []
            model_ids = []
            
            for model in models:
                ku = calc_ku(model_weights[model.id])
                kurtosis_values.append(ku)
                model_ids.append(model.id)
                logger.debug(f"Model {model.id} kurtosis: {ku:.4f}")
                print(f"Modello: {model.id} -> kurtosi: {ku:.4f}")
            
            # Build distance matrix
            n_models = len(models)
            distance_matrix = np.zeros((n_models, n_models))
            
            for i in range(n_models):
                for j in range(i, n_models):
                    if i != j:
                        dist = self.distance_calculator.calculate_l2_distance(
                            model_weights[model_ids[i]], 
                            model_weights[model_ids[j]]
                        )
                        distance_matrix[i, j] = dist
                        distance_matrix[j, i] = dist

                    else:
                        distance_matrix[i, j] = 0
                        
            
            # Apply MoTHer algorithm using existing implementation
            tree, confidence_scores = self.build_tree(
                ku_values=kurtosis_values,
                distance_matrix=distance_matrix,
                lambda_param=self.lambda_param
            )
            
            logger.info(f"MoTHer tree built with {tree.number_of_nodes()} nodes")
            return tree, confidence_scores
            
        except Exception as e:
            logger.error(f"Error building MoTHer tree: {e}")
            return nx.DiGraph(), {}
    
    def build_tree(self, ku_values: List[float], 
               distance_matrix: np.ndarray, 
               lambda_param: float) -> Tuple[nx.DiGraph, Dict[int, float]]:
        """
        Build directed tree using MoTHer algorithm with Chu-Liu-Edmonds MDST
        """
        n = len(ku_values)
        
        if n < 2:
            logger.warning("Cannot build tree with fewer than 2 models")
            return nx.DiGraph(), {}
        
        if n == 2:

            # Higher kurtosis = parent (original model, less fine-tuned)
            if ku_values[0] > ku_values[1]:
                parent, child = 0, 1
            else:
                parent, child = 1, 0
                
            tree = nx.DiGraph()
            tree.add_edge(parent, child)
            return tree, {parent: 0.8, child: 0.7}
        
        logger.debug(f"Building tree with {n} models using Chu-Liu-Edmonds algorithm")

        true_lambda = compute_lambda(distance_matrix)
        
        # Create weighted directed graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        
        # Calculate edge weights combining distance and kurtosis
        for i in range(n):
            for j in range(n):
                if i != j:

                    # Distance component (normalized)
                    distance_cost = distance_matrix[i, j]
                    
                    # Higher kurtosis models should be parents
                    # Based on paper: fine-tuning reduces kurtosis (fewer outliers)
                    # A (high kurtosis, original) -> B (low kurtosis, fine-tuned)
                    kurtosis_diff = ku_values[i] - ku_values[j]  # parent_ku - child_ku
                    
                    # If i has higher kurtosis than j, this is a good parent->child relationship
                    if kurtosis_diff > 0:
                        kurtosis_cost = 0 #-abs(kurtosis_diff)  # Negative cost = preferred
                    else:
                        kurtosis_cost = abs(kurtosis_diff) * 2  # Penalty for bad direction
                    
                    # Combine costs using lambda parameter
                    edge_weight = true_lambda * kurtosis_cost + (1 - true_lambda) * distance_cost
                    
                    G.add_edge(i, j, weight=edge_weight)

        # Apply Chu-Liu-Edmonds algorithm for Minimum Directed Spanning Tree
        try:
            spanning_tree = mdst.chu_liu_edmonds_algorithm(G,np.argmax(ku_values))
            logger.debug(f"Chu-Liu-Edmonds completed: {spanning_tree.number_of_nodes()} nodes, {spanning_tree.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"Chu-Liu-Edmonds failed ({e}), using fallback")
            spanning_tree = fallback_directed_mst(G)
        
        # Calculate confidence scores
        confidence_scores = calculate_confidence_scores(spanning_tree, G, ku_values)
        
        return spanning_tree, confidence_scores

    def _build_distance_tree(self, 
                           models: List[Model], 
                           model_weights: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict[int, float]]:
        """
        Build tree using only distance information (minimum spanning tree).
        """
        try:
            model_ids = [model.id for model in models]
            n_models = len(models)
            
            # Build distance matrix
            distance_matrix = np.zeros((n_models, n_models))
            
            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        dist = self.distance_calculator.calculate_l2_distance(
                            model_weights[model_ids[i]], 
                            model_weights[model_ids[j]]
                        )
                        distance_matrix[i, j] = dist
                    else:
                        distance_matrix[i, j] = float('inf')
            
            # Use MoTHer tree builder with lambda=0 (distance only)
            tree, confidence_scores = self.build_tree(
                ku_values=[0.0] * n_models,  # Dummy kurtosis values
                distance_matrix=distance_matrix,
                lambda_param=0.0  # Distance only
            )
            
            logger.info(f"Distance-only tree built with {tree.number_of_nodes()} nodes")
            return tree, confidence_scores
            
        except Exception as e:
            logger.error(f"Error building distance tree: {e}")
            return nx.DiGraph(), {}
    
    def _build_kurtosis_tree(self, 
                           models: List[Model], 
                           model_weights: Dict[str, Any]) -> Tuple[nx.DiGraph, Dict[int, float]]:
        """
        Build tree using only kurtosis information (highest kurtosis as root).
        """
        try:
            # Calculate kurtosis for each model
            kurtosis_values = []
            model_ids = []
            
            for model in models:
                ku = calc_ku(model_weights[model.id])
                kurtosis_values.append(ku)
                model_ids.append(model.id)
            
            n_models = len(models)
            
            # Create dummy distance matrix (uniform distances)
            distance_matrix = np.ones((n_models, n_models))
            np.fill_diagonal(distance_matrix, float('inf'))
            
            # Use MoTHer tree builder with lambda=1 (kurtosis only)
            tree, confidence_scores = self.build_tree(
                ku_values=kurtosis_values,
                distance_matrix=distance_matrix,
                lambda_param=1.0  # Kurtosis only
            )
            
            logger.info(f"Kurtosis-only tree built with {tree.number_of_nodes()} nodes")
            return tree, confidence_scores
            
        except Exception as e:
            logger.error(f"Error building kurtosis tree: {e}")
            return nx.DiGraph(), {}
    
    def _convert_tree_to_model_ids(self, 
                                  tree: nx.DiGraph, 
                                  models: List[Model]) -> nx.DiGraph:
        """
        Convert tree with integer node indices to tree with model IDs.
        """
        try:
            if tree.number_of_nodes() == 0:
                return nx.DiGraph()
            
            # Create mapping from index to model ID
            index_to_id = {i: models[i].id for i in range(len(models))}
            
            # Create new tree with model IDs
            id_tree = nx.DiGraph()
            
            # Add nodes
            for node in tree.nodes():
                if node < len(models):
                    id_tree.add_node(index_to_id[node])
            
            # Add edges
            for source, target in tree.edges():
                if source < len(models) and target < len(models):
                    source_id = index_to_id[source]
                    target_id = index_to_id[target]
                    
                    # Copy edge attributes if they exist
                    edge_data = tree.get_edge_data(source, target)
                    if edge_data:
                        id_tree.add_edge(source_id, target_id, **edge_data)
                    else:
                        id_tree.add_edge(source_id, target_id)
            
            return id_tree
            
        except Exception as e:
            logger.error(f"Error converting tree to model IDs: {e}")
            return nx.DiGraph()
    
    def _convert_confidence_to_model_ids(self, 
                                       confidence_scores: Dict[int, float], 
                                       models: List[Model]) -> Dict[str, float]:
        """
        Convert confidence scores with integer indices to model IDs.
        """
        try:
            id_confidence = {}
            
            for index, confidence in confidence_scores.items():
                if index < len(models):
                    model_id = models[index].id
                    id_confidence[model_id] = confidence
            
            return id_confidence
            
        except Exception as e:
            logger.error(f"Error converting confidence to model IDs: {e}")
            return {}

    def get_tree_statistics(self, tree: nx.DiGraph) -> Dict[str, Any]:
        """
        Calculate statistics for a genealogical tree.
        
        Args:
            tree: Tree to analyze
            
        Returns:
            Dictionary with tree statistics
        """
        try:
            if tree.number_of_nodes() == 0:
                return {
                    'num_nodes': 0,
                    'num_edges': 0,
                    'num_roots': 0,
                    'num_leaves': 0,
                    'max_depth': 0,
                    'avg_branching_factor': 0.0
                }
            
            # Basic counts
            num_nodes = tree.number_of_nodes()
            num_edges = tree.number_of_edges()
            
            # Root nodes (no predecessors)
            roots = [node for node in tree.nodes() if tree.in_degree(node) == 0]
            num_roots = len(roots)
            
            # Leaf nodes (no successors)
            leaves = [node for node in tree.nodes() if tree.out_degree(node) == 0]
            num_leaves = len(leaves)
            
            # Calculate depths from each root
            max_depth = 0
            if roots:
                for root in roots:
                    depths = nx.single_source_shortest_path_length(tree, root)
                    max_depth = max(max_depth, max(depths.values()) if depths else 0)
            
            # Average branching factor
            out_degrees = [tree.out_degree(node) for node in tree.nodes()]
            avg_branching = np.mean(out_degrees) if out_degrees else 0.0
            
            return {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_roots': num_roots,
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'avg_branching_factor': float(avg_branching),
                'roots': roots,
                'leaves': leaves
            }
            
        except Exception as e:
            logger.error(f"Error calculating tree statistics: {e}")
            return {}
    
    def validate_tree(self, tree: nx.DiGraph) -> Tuple[bool, List[str]]:
        """
        Validate that the tree is a proper directed acyclic graph (DAG).
        
        Args:
            tree: Tree to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        try:
            issues = []
            
            # Check if it's a DAG
            if not nx.is_directed_acyclic_graph(tree):
                issues.append("Tree contains cycles")
            
            # Check that each node has at most one parent
            for node in tree.nodes():
                in_degree = tree.in_degree(node)
                if in_degree > 1:
                    issues.append(f"Node {node} has multiple parents ({in_degree})")
            
            # Check connectivity
            if tree.number_of_nodes() > 1:
                undirected = tree.to_undirected()
                if not nx.is_connected(undirected):
                    issues.append("Tree is not connected")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                logger.info("Tree validation passed")
            else:
                logger.warning(f"Tree validation failed with issues: {issues}")
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating tree: {e}")
            return False, [f"Validation error: {e}"]