"""
MoTHer Algorithm Main Implementation
Replaces the find_parent_stub function with sophisticated heritage detection
"""

import logging
import networkx as nx

from typing import Tuple, Dict

logger = logging.getLogger(__name__)

# Minimum Directed Spanning Tree (MDST) using Chu-Liu-Edmonds algorithm
class MDST:
    
    def __init__(self):
        pass

    def chu_liu_edmonds_algorithm(self, G: nx.DiGraph, root: int) -> nx.DiGraph:
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
        min_edges = self.find_min_incoming_edges(G, root)
        
        if not min_edges:

            # No incoming edges found, return just the root
            result = nx.DiGraph()
            result.add_node(root)
            return result
        
        # Step 2: Check for cycles in the minimum edge set
        cycles = self.find_cycles_in_min_edges(min_edges, root)
        
        if not cycles:

            # No cycles: we have our arborescence
            return self.build_arborescence_from_edges(G, min_edges, root)
        
        # Step 3: Contract cycles and solve recursively
        return self.contract_cycles_and_recurse(G, root, min_edges, cycles)

    def find_min_incoming_edges(self, graph: nx.DiGraph, root: int) -> Dict[int, Tuple[int, int, float, float]]:
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
                    distance = graph[pred][node]['distance']  # ← AGGIUNGI
                    if weight < min_weight:
                        min_weight = weight
                        min_edge = (pred, node, weight, distance)  # ← AGGIUNGI distance
            
            if min_edge is not None:
                min_edges[node] = min_edge
        
        return min_edges

    def find_cycles_in_min_edges(self, min_edges: Dict[int, Tuple[int, int, float, float]], root: int) -> list:
        """Find cycles formed by minimum incoming edges"""
        temp_graph = nx.DiGraph()
        
        nodes = set([root])
        for target, (source, _, _, _) in min_edges.items():  # ← 4 elementi ora
            nodes.add(source)
            nodes.add(target)
        
        temp_graph.add_nodes_from(nodes)
        
        for target, (source, _, weight, distance) in min_edges. items():  # ← AGGIUNGI distance
            temp_graph.add_edge(source, target, weight=weight, distance=distance)  # ← AGGIUNGI
        
        try:
            cycles = list(nx.simple_cycles(temp_graph))
            return cycles
        except:
            return []

    def build_arborescence_from_edges(self, graph: nx.DiGraph, min_edges: Dict[int, Tuple[int, int, float, float]], root: int) -> nx.DiGraph:
        """Build arborescence from minimum edges (when no cycles exist)"""
        result = nx.DiGraph()
        result.add_node(root)
        
        for target, (source, _, weight, distance) in min_edges.items():  # ← AGGIUNGI distance
            result.add_edge(source, target, weight=weight, distance=distance)  # ← AGGIUNGI
        
        return result
    
    def contract_cycles_and_recurse(self, graph: nx.DiGraph, root: int, 
                                min_edges: Dict[int, Tuple[int, int, float, float]], cycles: list) -> nx.DiGraph:
        """Contract cycles into super-nodes and solve recursively"""
        
        cycle = cycles[0]
        cycle_nodes = set(cycle)
        
        contracted_graph = nx.DiGraph()
        
        super_node = f"super_{min(cycle)}"
        node_mapping = {}
        
        for node in graph.nodes():
            if node in cycle_nodes:
                node_mapping[node] = super_node
            else:
                node_mapping[node] = node
                contracted_graph.add_node(node)
        
        contracted_graph.add_node(super_node)
        
        cycle_weight = 0
        for node in cycle:
            if node in min_edges:
                cycle_weight += min_edges[node][2]
        
        # Add edges to contracted graph with adjusted weights
        for u, v, data in graph.edges(data=True):
            u_mapped = node_mapping[u]
            v_mapped = node_mapping[v]
            
            if u_mapped == v_mapped:
                continue
            
            weight = data['weight']
            distance = data.get('distance', 0.0)  # ← AGGIUNGI
            
            if v in cycle_nodes and u not in cycle_nodes:
                if v in min_edges:
                    weight = weight - min_edges[v][2]
            
            if contracted_graph.has_edge(u_mapped, v_mapped):
                current_weight = contracted_graph[u_mapped][v_mapped]['weight']
                if weight < current_weight:
                    contracted_graph[u_mapped][v_mapped]['weight'] = weight
                    contracted_graph[u_mapped][v_mapped]['distance'] = distance  # ← AGGIUNGI
            else:
                contracted_graph.add_edge(u_mapped, v_mapped, weight=weight, distance=distance)  # ← AGGIUNGI
        
        contracted_root = node_mapping[root]
        contracted_solution = self.chu_liu_edmonds_algorithm(contracted_graph, contracted_root)
        
        return self.expand_solution(graph, contracted_solution, cycle, min_edges, super_node, node_mapping)

    def expand_solution(self, original_graph: nx.DiGraph, contracted_solution: nx.DiGraph,
                cycle: list, min_edges: Dict[int, Tuple[int, int, float, float]], 
                super_node: str, node_mapping: Dict[int, str]) -> nx.DiGraph:
        """Expand the solution from contracted graph back to original graph"""
        
        result = nx.DiGraph()
        result.add_nodes_from(original_graph.nodes())
        
        for u, v, data in contracted_solution.edges(data=True):
            if u == super_node:
                continue
            elif v == super_node:
                original_u = None
                for orig_node, mapped_node in node_mapping.items():
                    if mapped_node == u:
                        original_u = orig_node
                        break
                
                if original_u is not None:
                    min_weight = float('inf')
                    best_target = None
                    best_distance = None
                    for cycle_node in cycle:
                        if original_graph. has_edge(original_u, cycle_node):
                            weight = original_graph[original_u][cycle_node]['weight']
                            if weight < min_weight:
                                min_weight = weight
                                best_target = cycle_node
                                best_distance = original_graph[original_u][cycle_node]['distance']  # ✅ CORRETTO
                    
                    if best_target is not None:
                        result.add_edge(original_u, best_target, weight=min_weight, distance=best_distance)
            else:
                original_u = None
                original_v = None
                for orig_node, mapped_node in node_mapping. items():
                    if mapped_node == u:
                        original_u = orig_node
                    if mapped_node == v:
                        original_v = orig_node
                
                if original_u is not None and original_v is not None:
                    result.add_edge(original_u, original_v, 
                                weight=data['weight'], 
                                distance=data['distance'])  # ✅ OK anche questo
        
        external_entry = None
        for node in cycle:
            for pred in result.predecessors(node):
                if pred not in cycle:
                    external_entry = node
                    break
            if external_entry:
                break
        
        for node in cycle:
            if node != external_entry and node in min_edges:
                source, target, weight, distance = min_edges[node]
                if source in cycle:
                    result.add_edge(source, target, weight=weight, distance=distance)
        
        return result
