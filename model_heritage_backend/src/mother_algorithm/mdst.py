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
        """
        if G.number_of_nodes() <= 1:
            return G.copy()
        
        if root not in G.nodes():
            raise ValueError(f"Root node {root} not in graph")
        
        # Step 1: Find minimum incoming edges for each node (except root)
        min_edges = self.find_min_incoming_edges(G, root)
        
        # Se non ci sono archi entranti sufficienti, restituisci quello che puoi (o solo la root)
        if not min_edges and G.number_of_nodes() > 1:
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

    def find_min_incoming_edges(self, graph: nx.DiGraph, root: int) -> Dict[int, Tuple[int, int, float, dict]]:
        """
        Find minimum incoming edge for each node (except root).
        Returns dict: target -> (source, target, weight, full_attributes_dict)
        """
        min_edges = {}
        
        for node in graph.nodes():
            if node == root:
                continue
            
            candidates = []
            for pred in graph.predecessors(node):
                if graph.has_edge(pred, node):
                    data = graph[pred][node]
                    weight = data.get('weight', 0.0)
                    candidates.append((pred, node, weight, data))
            
            if candidates:
                # Seleziona quello con peso minore
                best_edge = min(candidates, key=lambda x: x[2])
                min_edges[node] = best_edge
        
        return min_edges

    def find_cycles_in_min_edges(self, min_edges: Dict[int, Tuple[int, int, float, dict]], root: int) -> list:
        """Find cycles formed by minimum incoming edges"""
        temp_graph = nx.DiGraph()
        
        # Costruiamo il grafo temporaneo delle selezioni greedy
        for target, (source, _, _, _) in min_edges.items():
            temp_graph.add_edge(source, target)
        
        try:
            cycles = list(nx.simple_cycles(temp_graph))
            return cycles
        except:
            return []

    def build_arborescence_from_edges(self, graph: nx.DiGraph, min_edges: Dict[int, Tuple[int, int, float, dict]], root: int) -> nx.DiGraph:
        """Build arborescence from minimum edges (when no cycles exist)"""
        result = nx.DiGraph()
        result.add_nodes_from(graph.nodes()) # Aggiungi tutti i nodi, anche isolati se ce ne fossero
        
        for target, (source, _, _, data) in min_edges.items():
            result.add_edge(source, target, **data)
        
        return result
    
    def contract_cycles_and_recurse(self, graph: nx.DiGraph, root: int, 
                                min_edges: Dict[int, Tuple[int, int, float, dict]], cycles: list) -> nx.DiGraph:
        """Contract cycles into super-nodes and solve recursively"""
        
        cycle = cycles[0]
        cycle_nodes = set(cycle)
        
        contracted_graph = nx.DiGraph()
        super_node = f"super_{min(cycle)}"
        
        # Mappa dei nodi: Nodo Originale -> Nodo Contratto (o se stesso)
        node_mapping = {}
        for node in graph.nodes():
            if node in cycle_nodes:
                node_mapping[node] = super_node
            else:
                node_mapping[node] = node
                contracted_graph.add_node(node)
        contracted_graph.add_node(super_node)
        
        # Iteriamo su TUTTI gli archi del grafo originale
        for u, v, data in graph.edges(data=True):
            u_mapped = node_mapping[u]
            v_mapped = node_mapping[v]
            
            # 1. Ignora self-loops sul super-nodo (archi interni al ciclo)
            if u_mapped == v_mapped:
                continue
            
            new_weight = data['weight']
            # Copiamo tutti i dati originali per non perderli (distance, etc.)
            new_data = data.copy()
            
            # --- SALVATAGGIO FONDAMENTALE PER L'ESPANSIONE ---
            # Salviamo chi erano u e v nel grafo originale. 
            # Questo evita di doverli "indovinare" dopo.
            new_data['original_u'] = u
            new_data['original_v'] = v
            
            # 2. Gestione pesi per archi entranti nel ciclo
            if v in cycle_nodes:
                # Formula: w' = w - w(edge_in_cycle_pointing_to_v)
                # min_edges[v] è la tupla (source, target, weight, data)
                cycle_edge_weight = min_edges[v][2]
                new_weight = new_weight - cycle_edge_weight
                new_data['weight'] = new_weight
            
            # 3. Gestione archi uscenti (es. 9 -> 5)
            # Se u in cycle_nodes e v fuori, l'arco diventa super_node -> v
            # Il peso rimane invariato. Il 'bug' precedente era qui (venivano ignorati dopo).
            
            # Aggiunta al grafo contratto (gestione multi-archi: teniamo il migliore)
            if contracted_graph.has_edge(u_mapped, v_mapped):
                current_weight = contracted_graph[u_mapped][v_mapped]['weight']
                if new_weight < current_weight:
                    contracted_graph.add_edge(u_mapped, v_mapped, **new_data)
            else:
                contracted_graph.add_edge(u_mapped, v_mapped, **new_data)
        
        # Ricorsione
        contracted_root = node_mapping[root]
        contracted_solution = self.chu_liu_edmonds_algorithm(contracted_graph, contracted_root)
        
        return self.expand_solution(graph, contracted_solution, cycle, min_edges, super_node, node_mapping)

    def expand_solution(self, original_graph: nx.DiGraph, contracted_solution: nx.DiGraph,
                cycle: list, min_edges: Dict[int, Tuple[int, int, float, dict]], 
                super_node: str, node_mapping: Dict[int, str]) -> nx.DiGraph:
        """Expand the solution from contracted graph back to original graph"""
        
        result = nx.DiGraph()
        result.add_nodes_from(original_graph.nodes())
        
        cycle_nodes = set(cycle)
        entry_node_in_cycle = None
        
        # 1. Ripristina gli archi dalla soluzione contratta
        for u, v, data in contracted_solution.edges(data=True):
            # Recuperiamo gli ID veri grazie al salvataggio fatto in contract_cycles
            # .get('original_u', u) serve per gli archi che non sono stati toccati (esterni)
            real_u = data.get('original_u', u)
            real_v = data.get('original_v', v)
            
            # Puliamo i dati rimuovendo le chiavi di servizio
            clean_data = {k: val for k, val in data.items() 
                          if k not in ['original_u', 'original_v', 'weight']}
            
            # Ripristiniamo il peso originale dal grafo originale per sicurezza
            # (perché 'weight' in data potrebbe essere quello modificato)
            original_weight = original_graph[real_u][real_v]['weight']
            clean_data['weight'] = original_weight
            
            # Aggiungiamo l'arco "vero"
            result.add_edge(real_u, real_v, **clean_data)
            
            # Se questo arco entra nel ciclo, segniamo il punto di ingresso
            if real_v in cycle_nodes and real_u not in cycle_nodes:
                entry_node_in_cycle = real_v

        # 2. Ripristina il ciclo interno
        # Se la radice era parte del ciclo, non c'è un entry_node esterno, 
        # ma la root funge da entry.
        
        # Aggiungiamo tutti gli archi del ciclo originale tranne quello che punta all'entry_node
        for node in cycle:
            if node == entry_node_in_cycle:
                continue
            
            # Recupera l'arco del ciclo originale (salvato in min_edges)
            source, target, weight, edge_data = min_edges[node]
            result.add_edge(source, target, **edge_data)
            
        return result