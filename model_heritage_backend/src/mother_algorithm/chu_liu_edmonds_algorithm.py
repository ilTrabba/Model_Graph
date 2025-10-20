"""
MoTHer Algorithm Main Implementation
Replaces the find_parent_stub function with sophisticated heritage detection
"""

import re
import logging
import networkx as nx
from typing import Tuple, Dict
from src.db_entities.entity import ModelQuery
from src.mother_algorithm.mother_utils import (
    load_model_weights, calc_ku, calculate_l2_distance, build_tree
)

logger = logging.getLogger(__name__)
model_query = ModelQuery()

def chu_liu_edmonds_algorithm(G: nx.DiGraph, root: int) -> nx.DiGraph:
    """
    Chu-Liu-Edmonds algorithm for Minimum Directed Spanning Tree (Arborescence)
    
    Args:
        G: Directed graph with 'weight' attribute on edges
        root: Root node for the arborescence
        
    Returns:
        Minimum spanning arborescence rooted at 'root'
    """
    if G.number_of_nodes() <= 1:
        return G.copy()
    
    if root not in G.nodes():
        raise ValueError(f"Root node {root} not in graph")
    
    # Step 1: Find minimum incoming edges for each node (except root)
    min_edges = find_min_incoming_edges(G, root)
    
    if not min_edges:

        # No incoming edges found, return just the root
        result = nx.DiGraph()
        result.add_node(root)
        return result
    
    # Step 2: Check for cycles in the minimum edge set
    cycles = find_cycles_in_min_edges(min_edges, root)
    
    if not cycles:

        # No cycles: we have our arborescence
        return build_arborescence_from_edges(G, min_edges, root)
    
    # Step 3: Contract cycles and solve recursively
    return contract_cycles_and_recurse(G, root, min_edges, cycles)

def find_min_incoming_edges(graph: nx.DiGraph, root: int) -> Dict[int, Tuple[int, int, float]]:
    """Find minimum incoming edge for each node (except root)"""
    min_edges = {}
    
    for node in graph.nodes():
        if node == root:
            continue
            
        min_weight = float('inf')
        min_edge = None
        
        for pred in graph.predecessors(node):
            if graph.has_edge(pred, node):
                weight = graph[pred][node]['weight']
                if weight < min_weight:
                    min_weight = weight
                    min_edge = (pred, node, weight)
        
        if min_edge is not None:
            min_edges[node] = min_edge
    
    return min_edges

def find_cycles_in_min_edges(min_edges: Dict[int, Tuple[int, int, float]], root: int) -> list:
    """Find cycles formed by minimum incoming edges"""
    # Build a graph with only the minimum edges
    temp_graph = nx.DiGraph()
    
    # Add all nodes that appear in min_edges
    nodes = set([root])
    for target, (source, _, _) in min_edges.items():
        nodes.add(source)
        nodes.add(target)
    
    temp_graph.add_nodes_from(nodes)
    
    # Add minimum edges
    for target, (source, _, weight) in min_edges.items():
        temp_graph.add_edge(source, target, weight=weight)
    
    # Find cycles
    try:
        cycles = list(nx.simple_cycles(temp_graph))
        return cycles
    except:
        return []

def build_arborescence_from_edges(graph: nx.DiGraph, min_edges: Dict[int, Tuple[int, int, float]], root: int) -> nx.DiGraph:
    """Build arborescence from minimum edges (when no cycles exist)"""
    result = nx.DiGraph()
    result.add_node(root)
    
    for target, (source, _, weight) in min_edges.items():
        result.add_edge(source, target, weight=weight)
    
    return result

def contract_cycles_and_recurse(graph: nx.DiGraph, root: int, min_edges: Dict[int, Tuple[int, int, float]], cycles: list) -> nx.DiGraph:
    """Contract cycles into super-nodes and solve recursively"""
    
    # Take the first cycle to contract
    cycle = cycles[0]
    cycle_nodes = set(cycle)
    
    # Create contracted graph
    contracted_graph = nx.DiGraph()
    
    # Create mapping: original_node -> contracted_node
    super_node = f"super_{min(cycle)}"  # Name for the super-node
    node_mapping = {}
    
    # Map cycle nodes to super-node, others to themselves
    for node in graph.nodes():
        if node in cycle_nodes:
            node_mapping[node] = super_node
        else:
            node_mapping[node] = node
            contracted_graph.add_node(node)
    
    contracted_graph.add_node(super_node)
    
    # Calculate cycle weight (sum of minimum edges in cycle)
    cycle_weight = 0
    for node in cycle:
        if node in min_edges:
            cycle_weight += min_edges[node][2]
    
    # Add edges to contracted graph with adjusted weights
    for u, v, data in graph.edges(data=True):
        u_mapped = node_mapping[u]
        v_mapped = node_mapping[v]
        
        # Skip self-loops in contracted graph
        if u_mapped == v_mapped:
            continue
        
        weight = data['weight']
        
        # If edge enters the super-node, adjust weight
        if v in cycle_nodes and u not in cycle_nodes:
            # Adjust weight by subtracting the minimum incoming edge weight for v
            if v in min_edges:
                weight = weight - min_edges[v][2]
        
        # Add edge to contracted graph (keep minimum if multiple edges exist)
        if contracted_graph.has_edge(u_mapped, v_mapped):
            current_weight = contracted_graph[u_mapped][v_mapped]['weight']
            if weight < current_weight:
                contracted_graph[u_mapped][v_mapped]['weight'] = weight
        else:
            contracted_graph.add_edge(u_mapped, v_mapped, weight=weight)
    
    # Recursively solve on contracted graph
    contracted_root = node_mapping[root]
    contracted_solution = chu_liu_edmonds_algorithm(contracted_graph, contracted_root)
    
    # Expand solution back to original graph
    return expand_solution(graph, contracted_solution, cycle, min_edges, super_node, node_mapping)

def expand_solution(original_graph: nx.DiGraph, contracted_solution: nx.DiGraph, 
                   cycle: list, min_edges: Dict[int, Tuple[int, int, float]], 
                   super_node: str, node_mapping: Dict[int, str]) -> nx.DiGraph:
    """Expand the solution from contracted graph back to original graph"""
    
    result = nx.DiGraph()
    
    # Add all original nodes
    result.add_nodes_from(original_graph.nodes())
    
    # Add edges from contracted solution, mapping back to original nodes
    for u, v, data in contracted_solution.edges(data=True):
        if u == super_node:
            # Edge from super-node: shouldn't happen in a valid arborescence
            continue
        elif v == super_node:
            # Edge to super-node: need to find which cycle node it connects to
            # Find the cycle node that this edge should connect to
            original_u = None
            for orig_node, mapped_node in node_mapping.items():
                if mapped_node == u:
                    original_u = orig_node
                    break
            
            if original_u is not None:
                # Find best entry point to cycle
                min_weight = float('inf')
                best_target = None
                for cycle_node in cycle:
                    if original_graph.has_edge(original_u, cycle_node):
                        weight = original_graph[original_u][cycle_node]['weight']
                        if weight < min_weight:
                            min_weight = weight
                            best_target = cycle_node
                
                if best_target is not None:
                    result.add_edge(original_u, best_target, weight=min_weight)
        else:
            # Regular edge: map back to original nodes
            original_u = None
            original_v = None
            for orig_node, mapped_node in node_mapping.items():
                if mapped_node == u:
                    original_u = orig_node
                if mapped_node == v:
                    original_v = orig_node
            
            if original_u is not None and original_v is not None:
                result.add_edge(original_u, original_v, weight=data['weight'])
    
    # Add cycle edges (except the one that was "broken" by external connection)
    external_entry = None
    for node in cycle:
        for pred in result.predecessors(node):
            if pred not in cycle:
                external_entry = node
                break
        if external_entry:
            break
    
    # Add minimum edges within cycle, except for external entry point
    for node in cycle:
        if node != external_entry and node in min_edges:
            source, target, weight = min_edges[node]
            if source in cycle:  # Internal cycle edge
                result.add_edge(source, target, weight=weight)
    
    return result
