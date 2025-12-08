import numpy as np
import networkx as nx

def chu_liu_edmonds_robust(graph_input, root):
    """
    Implementazione robusta di Chu-Liu-Edmonds.
    1. Sanitizza l'input rimuovendo numpy/views.
    2. Esegue l'algoritmo.
    3. Verifica che il risultato sia un albero valido.
    """
    
    # --- STEP 0: SANITIZZAZIONE INPUT ---
    # Convertiamo tutto in {u: {v: float}} puro per evitare errori di tipo
    graph = {}
    nodes = set()
    for u, neighbors in graph_input.items():
        nodes.add(u)
        if u not in graph: graph[u] = {}
        for v, attrs in neighbors.items():
            nodes.add(v)
            # Gestione sicura del peso (estrae il valore se è un dict o un numpy obj)
            w = attrs.get('weight') if isinstance(attrs, dict) else attrs
            graph[u][v] = float(w) # Forza float nativo Python

    nodes = list(nodes)
    if root not in nodes:
        raise ValueError(f"La radice {root} non è presente nei nodi del grafo.")

    # --- STEP 1: SELEZIONE GREEDY (Miglior arco entrante) ---
    # Per ogni nodo v != root, troviamo l'arco u->v con peso minore
    min_in_edges = {} # Key: v (figlio), Value: (u (padre), peso)
    
    for v in nodes:
        if v == root:
            continue
        
        best_u = None
        min_w = float('inf')
        
        # Scansioniamo tutti i possibili padri u
        for u in nodes:
            if u in graph and v in graph[u]:
                w = graph[u][v]
                if w < min_w:
                    min_w = w
                    best_u = u
        
        if best_u is not None:
            min_in_edges[v] = (best_u, min_w)
        else:
            # Se un nodo non è la radice e non ha padri, l'arborescenza è impossibile
            raise ValueError(f"Il nodo {v} è irraggiungibile (non ha archi entranti).")

    # --- STEP 2: RILEVAMENTO CICLI ---
    # Costruiamo il grafo temporaneo delle scelte greedy
    # cycle_detect_graph[u] = [v]
    temp_adj = {u: [] for u in nodes}
    for v, (u, w) in min_in_edges.items():
        temp_adj[u].append(v)
        
    cycle = _find_cycle_iterative(nodes, temp_adj)
    
    # CASO A: Nessun ciclo -> Abbiamo finito!
    if not cycle:
        # Costruiamo il dizionario risultato
        arborescence = {u: {} for u in nodes}
        for v, (u, w) in min_in_edges.items():
            arborescence[u][v] = {'weight': w}
        _validate_result(arborescence, root, len(nodes))
        return arborescence

    # CASO B: Ciclo trovato -> Contrazione (Recursion)
    cycle_nodes = set(cycle)
    new_node = max(nodes) + 1 # Super-nodo
    
    # Creiamo il grafo contratto
    contracted_graph = {}
    
    # Calcolo costo del ciclo (per la normalizzazione)
    cycle_weight = 0.0
    for v in cycle:
        # Trova chi punta a v nel ciclo
        # Nota: min_in_edges[v] è l'arco scelto. Se c'è un ciclo, questo arco FA PARTE del ciclo.
        cycle_weight += min_in_edges[v][1]

    for u in nodes:
        if u not in graph: continue
        for v in graph[u]:
            w = graph[u][v]
            
            # Caso 1: Esterno -> Esterno (Copia)
            if u not in cycle_nodes and v not in cycle_nodes:
                _add_edge(contracted_graph, u, v, w)
            
            # Caso 2: Esterno -> Ciclo (Entra nel Super-Nodo)
            elif u not in cycle_nodes and v in cycle_nodes:
                # Edmonds: w' = w - w_ciclo(v)
                w_in_cycle_v = min_in_edges[v][1]
                new_w = w - w_in_cycle_v
                # Salviamo quale era il vero nodo 'v' target per l'unpacking
                _add_edge(contracted_graph, u, new_node, new_w, original_target=v)
            
            # Caso 3: Ciclo -> Esterno (Esce dal Super-Nodo)
            elif u in cycle_nodes and v not in cycle_nodes:
                # Salviamo quale era il vero nodo 'u' source
                _add_edge(contracted_graph, new_node, v, w, original_source=u)
            
            # Caso 4: Interno al ciclo (Ignorato nel grafo contratto)
            pass

    # Chiamata Ricorsiva
    # Se la radice era nel ciclo, il super-nodo diventa la nuova radice
    next_root = new_node if root in cycle_nodes else root
    mst_contracted = chu_liu_edmonds_robust(contracted_graph, next_root)
    
    # --- STEP 3: UNPACKING (Espansione) ---
    final_adj = {u: {} for u in nodes}
    
    # 1. Aggiungiamo tutti gli archi scelti nel ciclo (uno verrà rimosso dopo)
    for v in cycle:
        u, w = min_in_edges[v]
        final_adj[u][v] = {'weight': w}
        
    # 2. Integriamo gli archi dal risultato ricorsivo
    node_to_break_in_cycle = None
    
    for u_cont, children in mst_contracted.items():
        for v_cont, props in children.items():
            w_cont = props['weight']
            
            if v_cont == new_node: 
                # Arco che entra nel ciclo (u -> Ciclo)
                # Recuperiamo il vero target dal grafo contratto (che abbiamo salvato prima)
                # Nota: qui dobbiamo ritrovare i metadati salvati. 
                # Poiché mst_contracted è pulito, dobbiamo riguardare contracted_graph per i metadati.
                # Ma 'w_cont' corrisponde al minimo scelto.
                
                # Strategia: Ricalcoliamo quale nodo del ciclo u_cont sta puntando con quel peso
                real_v = _find_original_target(contracted_graph, u_cont, new_node, w_cont)
                real_w = graph[u_cont][real_v] # Peso originale
                
                final_adj[u_cont][real_v] = {'weight': real_w}
                node_to_break_in_cycle = real_v
                
            elif u_cont == new_node:
                # Arco che esce dal ciclo (Ciclo -> v)
                real_u = _find_original_source(contracted_graph, new_node, v_cont, w_cont)
                real_w = graph[real_u][v_cont]
                final_adj[real_u][v_cont] = {'weight': real_w}
                
            else:
                # Arco normale
                final_adj[u_cont][v_cont] = {'weight': w_cont}

    # 3. Rompiamo il ciclo nel punto di ingresso
    if node_to_break_in_cycle is not None:
        # Dobbiamo rimuovere l'arco INTERNO al ciclo che puntava a node_to_break_in_cycle
        parent_in_cycle, _ = min_in_edges[node_to_break_in_cycle]
        if parent_in_cycle in final_adj and node_to_break_in_cycle in final_adj[parent_in_cycle]:
            del final_adj[parent_in_cycle][node_to_break_in_cycle]

    _validate_result(final_adj, root, len(nodes))
    return final_adj

# --- HELPER FUNCTIONS ---

def _add_edge(g, u, v, w, original_target=None, original_source=None):
    if u not in g: g[u] = {}
    # Se esiste già un arco multiplo (dalla contrazione), teniamo il minimo
    if v not in g[u] or w < g[u][v]['weight']:
        g[u][v] = {'weight': w, 'orig_t': original_target, 'orig_s': original_source}

def _find_original_target(contracted_g, u, new_node, w_looked_for):
    # Recupera il metadato salvato
    return contracted_g[u][new_node]['orig_t']

def _find_original_source(contracted_g, new_node, v, w_looked_for):
    return contracted_g[new_node][v]['orig_s']

def _find_cycle_iterative(nodes, adj):
    """Rileva cicli usando DFS iterativa (più sicura della ricorsione)."""
    visited = set()
    recursion_stack = set()
    parent_map = {} # Per ricostruire il ciclo
    
    for start_node in nodes:
        if start_node in visited: continue
        
        stack = [(start_node, iter(adj.get(start_node, [])))]
        visited.add(start_node)
        recursion_stack.add(start_node)
        
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child in recursion_stack:
                    # CICLO TROVATO
                    cycle = [child]
                    curr = parent
                    while curr != child:
                        cycle.append(curr)
                        # Dobbiamo trovare chi punta a curr nello stack corrente
                        # Metodo semplice: guardiamo lo stack
                        found_parent = False
                        for i in range(len(stack)-1):
                             if stack[i+1][0] == curr:
                                 curr = stack[i][0]
                                 found_parent = True
                                 break
                        if not found_parent: break # Should not happen
                    return cycle
                
                if child not in visited:
                    visited.add(child)
                    recursion_stack.add(child)
                    stack.append((child, iter(adj.get(child, []))))
            except StopIteration:
                stack.pop()
                recursion_stack.remove(parent)
    return None

def _validate_result(adj, root, num_nodes):
    """Verifica di sanità mentale sul risultato."""
    parents = {}
    count = 0
    for u, children in adj.items():
        for v in children:
            if v in parents:
                raise ValueError(f"ERRORE GRAVE: Il nodo {v} ha due padri: {parents[v]} e {u}")
            parents[v] = u
            count += 1
    
    if count != num_nodes - 1:
        # Nota: questo può succedere se il grafo originale è disconnesso.
        # Ma per questo problema specifico ci aspettiamo un albero completo.
        print(f"ATTENZIONE: Trovati {count} archi, attesi {num_nodes-1}. Il grafo potrebbe essere disconnesso.")

# --- TUOI DATI (INCOLLATI DA TE) ---
raw_data = {
    0: {1: {'weight': np.float64(2.8316802978515625)}, 2: {'weight': np.float64(2.9032514095306396)}, 3: {'weight': np.float64(3.0643889904022217)}, 4: {'weight': np.float64(3.3813138253023833)}, 5: {'weight': np.float64(4.048594737701675)}, 6: {'weight': np.float64(3.2695980072021484)}, 7: {'weight': np.float64(3.5079813248446197)}, 8: {'weight': np.float64(2.082582473754883)}, 9: {'weight': np.float64(3.813126564025879)}, 10: {'weight': np.float64(3.8807041889956206)}, 11: {'weight': np.float64(2.379507541656494)}, 12: {'weight': np.float64(3.537630081176758)}, 13: {'weight': np.float64(3.548272132873535)}, 14: {'weight': np.float64(3.5042112118532867)}, 15: {'weight': np.float64(2.177345037460327)}, 16: {'weight': np.float64(3.7174148559570312)}, 18: {'weight': np.float64(3.347879409790039)}, 19: {'weight': np.float64(3.5538976192474365)}, 20: {'weight': np.float64(3.3446314334869385)}}, 
    1: {0: {'weight': np.float64(3.8219580895235747)}, 2: {'weight': np.float64(3.3586056232452393)}, 3: {'weight': np.float64(2.6118900775909424)}, 4: {'weight': np.float64(2.7569777018358916)}, 5: {'weight': np.float64(4.428071523361465)}, 6: {'weight': np.float64(3.6572341918945312)}, 7: {'weight': np.float64(4.019087339096329)}, 8: {'weight': np.float64(3.5194928646087646)}, 9: {'weight': np.float64(5.1637096649935454)}, 10: {'weight': np.float64(4.353056455307266)}, 11: {'weight': np.float64(3.686918020248413)}, 12: {'weight': np.float64(3.912600517272949)}, 13: {'weight': np.float64(4.910794521026871)}, 14: {'weight': np.float64(4.770794177704117)}, 15: {'weight': np.float64(3.558980703353882)}, 16: {'weight': np.float64(4.077112197875977)}, 18: {'weight': np.float64(2.9365909099578857)}, 19: {'weight': np.float64(3.9293062686920166)}, 20: {'weight': np.float64(2.941840410232544)}}, 
    2: {0: {'weight': np.float64(3.893529201202652)}, 1: {'weight': np.float64(4.3488834149172515)}, 3: {'weight': np.float64(4.4518428093722076)}, 4: {'weight': np.float64(3.862120176010391)}, 5: {'weight': np.float64(4.4291892296602935)}, 6: {'weight': np.float64(1.7963958978652954)}, 7: {'weight': np.float64(3.9573529011538238)}, 8: {'weight': np.float64(3.575862169265747)}, 9: {'weight': np.float64(3.5295641667177886)}, 10: {'weight': np.float64(4.41677882736232)}, 11: {'weight': np.float64(3.5172274112701416)}, 12: {'weight': np.float64(4.63490512435939)}, 13: {'weight': np.float64(4.860763335876724)}, 14: {'weight': np.float64(4.811910892181656)}, 15: {'weight': np.float64(3.6149704456329346)}, 16: {'weight': np.float64(2.3389668464660645)}, 18: {'weight': np.float64(3.5155580043792725)}, 19: {'weight': np.float64(2.035360336303711)}, 20: {'weight': np.float64(4.690716291122696)}}, 
    3: {0: {'weight': np.float64(4.054666782074234)}, 1: {'weight': np.float64(3.6021678692629546)}, 2: {'weight': np.float64(3.4615650177001953)}, 4: {'weight': np.float64(2.9007755762865752)}, 5: {'weight': np.float64(4.4436803108980865)}, 6: {'weight': np.float64(3.8790578842163086)}, 7: {'weight': np.float64(4.0348632580569)}, 8: {'weight': np.float64(3.5519320964813232)}, 9: {'weight': np.float64(5.26118948524501)}, 10: {'weight': np.float64(4.484116817169449)}, 11: {'weight': np.float64(3.8728437423706055)}, 12: {'weight': np.float64(3.9389827251434326)}, 13: {'weight': np.float64(4.934636378937027)}, 14: {'weight': np.float64(4.965388084106705)}, 15: {'weight': np.float64(3.7436344623565674)}, 16: {'weight': np.float64(4.167294025421143)}, 18: {'weight': np.float64(3.049647331237793)}, 19: {'weight': np.float64(3.864743232727051)}, 20: {'weight': np.float64(4.036382938080093)}}, 
    4: {0: {'weight': np.float64(2.391036033630371)}, 1: {'weight': np.float64(1.7666999101638794)}, 2: {'weight': np.float64(2.871842384338379)}, 3: {'weight': np.float64(1.910497784614563)}, 5: {'weight': np.float64(3.948250079803726)}, 6: {'weight': np.float64(3.3647749423980713)}, 7: {'weight': np.float64(3.463053250961563)}, 8: {'weight': np.float64(3.1717755794525146)}, 9: {'weight': np.float64(3.7979509830474854)}, 10: {'weight': np.float64(3.0066299438476562)}, 11: {'weight': np.float64(3.363482713699341)}, 12: {'weight': np.float64(3.5109872817993164)}, 13: {'weight': np.float64(3.5184600353240967)}, 14: {'weight': np.float64(3.4660117626190186)}, 15: {'weight': np.float64(3.2195379734039307)}, 16: {'weight': np.float64(3.6927826404571533)}, 18: {'weight': np.float64(2.3564231395721436)}, 19: {'weight': np.float64(3.52471661567688)}, 20: {'weight': np.float64(2.3687031269073486)}}, 
    5: {0: {'weight': np.float64(3.058316946029663)}, 1: {'weight': np.float64(3.437793731689453)}, 2: {'weight': np.float64(3.4389114379882812)}, 3: {'weight': np.float64(3.453402519226074)}, 4: {'weight': np.float64(2.957972288131714)}, 6: {'weight': np.float64(3.8581669330596924)}, 7: {'weight': np.float64(2.7550431734850616)}, 8: {'weight': np.float64(3.6033639907836914)}, 9: {'weight': np.float64(4.249019145965576)}, 10: {'weight': np.float64(2.4549882411956787)}, 11: {'weight': np.float64(3.8587734699249268)}, 12: {'weight': np.float64(3.053086996078491)}, 13: {'weight': np.float64(3.0514824390411377)}, 14: {'weight': np.float64(3.962582588195801)}, 15: {'weight': np.float64(3.73256516456604)}, 16: {'weight': np.float64(4.142317295074463)}, 18: {'weight': np.float64(3.772383451461792)}, 19: {'weight': np.float64(3.9072580337524414)}, 20: {'weight': np.float64(3.764801263809204)}}, 
    6: {0: {'weight': np.float64(4.259875798874161)}, 1: {'weight': np.float64(4.6475119835665435)}, 2: {'weight': np.float64(2.7866736895373077)}, 3: {'weight': np.float64(4.869335675888321)}, 4: {'weight': np.float64(4.3550527340700835)}, 5: {'weight': np.float64(4.848444724731705)}, 7: {'weight': np.float64(4.444839263611099)}, 8: {'weight': np.float64(4.872080350570938)}, 9: {'weight': np.float64(4.096048618011734)}, 10: {'weight': np.float64(4.701820159607193)}, 11: {'weight': np.float64(3.830977201461792)}, 12: {'weight': np.float64(5.0423555619051665)}, 13: {'weight': np.float64(5.24080374305751)}, 14: {'weight': np.float64(5.097804094009659)}, 15: {'weight': np.float64(4.905258918457291)}, 16: {'weight': np.float64(2.937891721725464)}, 18: {'weight': np.float64(4.921354556732437)}, 19: {'weight': np.float64(2.716374635696411)}, 20: {'weight': np.float64(5.076747918777725)}}, 
    7: {0: {'weight': np.float64(2.5177035331726074)}, 1: {'weight': np.float64(3.0288095474243164)}, 2: {'weight': np.float64(2.9670751094818115)}, 3: {'weight': np.float64(3.0445854663848877)}, 4: {'weight': np.float64(2.472775459289551)}, 5: {'weight': np.float64(1.7647653818130493)}, 6: {'weight': np.float64(3.454561471939087)}, 8: {'weight': np.float64(3.156979560852051)}, 9: {'weight': np.float64(3.8709497451782227)}, 10: {'weight': np.float64(1.7403239011764526)}, 11: {'weight': np.float64(3.453615427017212)}, 12: {'weight': np.float64(2.501788854598999)}, 13: {'weight': np.float64(2.508254051208496)}, 14: {'weight': np.float64(3.557210922241211)}, 15: {'weight': np.float64(3.3148839473724365)}, 16: {'weight': np.float64(3.767610549926758)}, 18: {'weight': np.float64(3.4102773666381836)}, 19: {'weight': np.float64(3.499701738357544)}, 20: {'weight': np.float64(3.4064254760742188)}}, 
    8: {0: {'weight': np.float64(3.072860265426895)}, 1: {'weight': np.float64(4.509770656280777)}, 2: {'weight': np.float64(4.566139960937759)}, 3: {'weight': np.float64(4.5422098881533355)}, 4: {'weight': np.float64(4.162053371124527)}, 5: {'weight': np.float64(4.593641782455704)}, 6: {'weight': np.float64(3.881802558898926)}, 7: {'weight': np.float64(4.147257352524063)}, 9: {'weight': np.float64(5.347576165848038)}, 10: {'weight': np.float64(4.456708694153091)}, 11: {'weight': np.float64(3.1634809970855713)}, 12: {'weight': np.float64(5.0074291474154204)}, 13: {'weight': np.float64(5.01625540321376)}, 14: {'weight': np.float64(4.290280605011246)}, 15: {'weight': np.float64(3.0093679428100586)}, 16: {'weight': np.float64(4.26025915145874)}, 18: {'weight': np.float64(4.931106353454849)}, 19: {'weight': np.float64(3.9284560680389404)}, 20: {'weight': np.float64(4.923726821594498)}}, 
    9: {0: {'weight': np.float64(4.803404355697891)}, 1: {'weight': np.float64(4.173431873321533)}, 2: {'weight': np.float64(2.5392863750457764)}, 3: {'weight': np.float64(4.270911693572998)}, 4: {'weight': np.float64(4.788228774719498)}, 5: {'weight': np.float64(5.239296937637588)}, 6: {'weight': np.float64(3.1057708263397217)}, 7: {'weight': np.float64(4.861227536850235)}, 8: {'weight': np.float64(4.357298374176025)}, 10: {'weight': np.float64(5.219926858597061)}, 11: {'weight': np.float64(4.311219692230225)}, 12: {'weight': np.float64(4.419049263000488)}, 13: {'weight': np.float64(4.605196475982666)}, 14: {'weight': np.float64(5.331615472488663)}, 15: {'weight': np.float64(4.384064197540283)}, 16: {'weight': np.float64(3.4631128311157227)}, 18: {'weight': np.float64(4.312011241912842)}, 19: {'weight': np.float64(3.2882540225982666)}, 20: {'weight': np.float64(4.461357116699219)}}, 
    10: {0: {'weight': np.float64(2.8904263973236084)}, 1: {'weight': np.float64(3.362778663635254)}, 2: {'weight': np.float64(3.4265010356903076)}, 3: {'weight': np.float64(3.4938390254974365)}, 4: {'weight': np.float64(3.9969077355196685)}, 5: {'weight': np.float64(3.445266032867691)}, 6: {'weight': np.float64(3.7115423679351807)}, 7: {'weight': np.float64(2.730601692848465)}, 8: {'weight': np.float64(3.466430902481079)}, 9: {'weight': np.float64(4.229649066925049)}, 11: {'weight': np.float64(3.7306268215179443)}, 12: {'weight': np.float64(3.038696527481079)}, 13: {'weight': np.float64(3.0434722900390625)}, 14: {'weight': np.float64(3.827712297439575)}, 15: {'weight': np.float64(3.6054837703704834)}, 16: {'weight': np.float64(4.131654262542725)}, 18: {'weight': np.float64(3.8060085773468018)}, 19: {'weight': np.float64(3.8990767002105713)}, 20: {'weight': np.float64(3.8016881942749023)}}, 
    11: {0: {'weight': np.float64(3.3697853333285064)}, 1: {'weight': np.float64(4.677195811920425)}, 2: {'weight': np.float64(4.507505202942154)}, 3: {'weight': np.float64(4.863121534042618)}, 4: {'weight': np.float64(4.353760505371353)}, 5: {'weight': np.float64(4.849051261596939)}, 6: {'weight': np.float64(4.821254993133804)}, 7: {'weight': np.float64(4.443893218689224)}, 8: {'weight': np.float64(4.1537587887575835)}, 9: {'weight': np.float64(5.301497483902237)}, 10: {'weight': np.float64(4.720904613189957)}, 12: {'weight': np.float64(5.03663590019252)}, 13: {'weight': np.float64(5.240166211776993)}, 14: {'weight': np.float64(4.4387059456637115)}, 15: {'weight': np.float64(4.203486228637955)}, 16: {'weight': np.float64(5.2062630898287505)}, 18: {'weight': np.float64(4.911695266418716)}, 19: {'weight': np.float64(5.065674806289932)}, 20: {'weight': np.float64(5.075596833877823)}}, 
    12: {0: {'weight': np.float64(4.52790787284877)}, 1: {'weight': np.float64(4.9028783089449615)}, 2: {'weight': np.float64(3.644627332687378)}, 3: {'weight': np.float64(4.929260516815445)}, 4: {'weight': np.float64(4.501265073471329)}, 5: {'weight': np.float64(4.0433647877505035)}, 6: {'weight': np.float64(4.052077770233154)}, 7: {'weight': np.float64(3.4920666462710113)}, 8: {'weight': np.float64(4.017151355743408)}, 9: {'weight': np.float64(5.4093270546725005)}, 10: {'weight': np.float64(4.028974319153091)}, 11: {'weight': np.float64(4.046358108520508)}, 13: {'weight': np.float64(4.518657231979629)}, 14: {'weight': np.float64(5.315854573898575)}, 15: {'weight': np.float64(4.134026050567627)}, 16: {'weight': np.float64(4.324892044067383)}, 18: {'weight': np.float64(4.0504984855651855)}, 19: {'weight': np.float64(4.096662998199463)}, 20: {'weight': np.float64(5.19774487083461)}}, 
    13: {0: {'weight': np.float64(4.538549924545547)}, 1: {'weight': np.float64(3.9205167293548584)}, 2: {'weight': np.float64(3.870485544204712)}, 3: {'weight': np.float64(3.9443585872650146)}, 4: {'weight': np.float64(4.508737826996109)}, 5: {'weight': np.float64(4.04176023071315)}, 6: {'weight': np.float64(4.250525951385498)}, 7: {'weight': np.float64(3.4985318428805083)}, 8: {'weight': np.float64(4.025977611541748)}, 9: {'weight': np.float64(5.595474267654678)}, 10: {'weight': np.float64(4.033750081711075)}, 11: {'weight': np.float64(4.2498884201049805)}, 12: {'weight': np.float64(3.528379440307617)}, 14: {'weight': np.float64(5.3411484009554595)}, 15: {'weight': np.float64(4.00761604309082)}, 16: {'weight': np.float64(4.36177921295166)}, 18: {'weight': np.float64(4.217453479766846)}, 19: {'weight': np.float64(4.292541027069092)}, 20: {'weight': np.float64(4.053243637084961)}}, 
    14: {0: {'weight': np.float64(2.5139334201812744)}, 1: {'weight': np.float64(3.7805163860321045)}, 2: {'weight': np.float64(3.8216331005096436)}, 3: {'weight': np.float64(3.9751102924346924)}, 4: {'weight': np.float64(4.456289554291031)}, 5: {'weight': np.float64(4.952860379867813)}, 6: {'weight': np.float64(4.1075263023376465)}, 7: {'weight': np.float64(4.547488713913223)}, 8: {'weight': np.float64(3.3000028133392334)}, 9: {'weight': np.float64(4.34133768081665)}, 10: {'weight': np.float64(4.817990089111587)}, 11: {'weight': np.float64(3.448428153991699)}, 12: {'weight': np.float64(4.3255767822265625)}, 13: {'weight': np.float64(4.350870609283447)}, 15: {'weight': np.float64(3.336210250854492)}, 16: {'weight': np.float64(4.475723743438721)}, 18: {'weight': np.float64(4.168687343597412)}, 19: {'weight': np.float64(4.345520496368408)}, 20: {'weight': np.float64(4.18206262588501)}}, 
    15: {0: {'weight': np.float64(3.1676228291323394)}, 1: {'weight': np.float64(4.549258495025894)}, 2: {'weight': np.float64(4.605248237304947)}, 3: {'weight': np.float64(4.73391225402858)}, 4: {'weight': np.float64(4.209815765075943)}, 5: {'weight': np.float64(4.722842956238052)}, 6: {'weight': np.float64(3.9149811267852783)}, 7: {'weight': np.float64(4.305161739044449)}, 8: {'weight': np.float64(3.999645734482071)}, 9: {'weight': np.float64(5.3743419892122954)}, 10: {'weight': np.float64(4.595761562042496)}, 11: {'weight': np.float64(3.2132084369659424)}, 12: {'weight': np.float64(5.124303842239639)}, 13: {'weight': np.float64(4.9978938347628326)}, 14: {'weight': np.float64(4.326488042526504)}, 16: {'weight': np.float64(4.132482051849365)}, 18: {'weight': np.float64(4.963712001495621)}, 19: {'weight': np.float64(4.14988374710083)}, 20: {'weight': np.float64(4.797836089782974)}}, 
    16: {0: {'weight': np.float64(4.7076926476290435)}, 1: {'weight': np.float64(5.067389989547989)}, 2: {'weight': np.float64(3.3292446381380767)}, 3: {'weight': np.float64(5.157571817093155)}, 4: {'weight': np.float64(4.683060432129166)}, 5: {'weight': np.float64(5.132595086746475)}, 6: {'weight': np.float64(3.928169513397476)}, 7: {'weight': np.float64(4.75788834159877)}, 8: {'weight': np.float64(5.2505369431307525)}, 9: {'weight': np.float64(4.453390622787735)}, 10: {'weight': np.float64(5.121932054214737)}, 11: {'weight': np.float64(4.215985298156738)}, 12: {'weight': np.float64(5.315169835739395)}, 13: {'weight': np.float64(5.352057004623672)}, 14: {'weight': np.float64(5.466001535110733)}, 15: {'weight': np.float64(5.1227598435213775)}, 18: {'weight': np.float64(5.202398324661514)}, 19: {'weight': np.float64(3.100804567337036)}, 20: {'weight': np.float64(5.202718759231827)}}, 
    17: {0: {'weight': np.float64(1.734190583229065)}, 1: {'weight': np.float64(2.423665761947632)}, 2: {'weight': np.float64(2.336991786956787)}, 3: {'weight': np.float64(2.538810968399048)}, 4: {'weight': np.float64(1.670309066772461)}, 5: {'weight': np.float64(2.533151865005493)}, 6: {'weight': np.float64(2.9365034103393555)}, 7: {'weight': np.float64(1.8230060338974)}, 8: {'weight': np.float64(2.713411331176758)}, 9: {'weight': np.float64(3.405182123184204)}, 10: {'weight': np.float64(2.513216733932495)}, 11: {'weight': np.float64(2.9368534088134766)}, 12: {'weight': np.float64(3.0923404693603516)}, 13: {'weight': np.float64(3.0997724533081055)}, 14: {'weight': np.float64(3.0401039123535156)}, 15: {'weight': np.float64(2.7731130123138428)}, 16: {'weight': np.float64(3.2985804080963135)}, 18: {'weight': np.float64(2.884599447250366)}, 19: {'weight': np.float64(3.1095800399780273)}, 20: {'weight': np.float64(2.8831002712249756)}}, 
    18: {0: {'weight': np.float64(4.338157201462051)}, 1: {'weight': np.float64(3.926868701629898)}, 2: {'weight': np.float64(4.505835796051285)}, 3: {'weight': np.float64(4.039925122909805)}, 4: {'weight': np.float64(3.346700931244156)}, 5: {'weight': np.float64(4.762661243133804)}, 6: {'weight': np.float64(3.931076765060425)}, 7: {'weight': np.float64(4.400555158310196)}, 8: {'weight': np.float64(3.940828561782837)}, 9: {'weight': np.float64(5.302289033584854)}, 10: {'weight': np.float64(4.796286369018814)}, 11: {'weight': np.float64(3.921417474746704)}, 12: {'weight': np.float64(5.040776277237198)}, 13: {'weight': np.float64(5.207731271438858)}, 14: {'weight': np.float64(5.158965135269424)}, 15: {'weight': np.float64(3.9734342098236084)}, 16: {'weight': np.float64(4.212120532989502)}, 19: {'weight': np.float64(4.0717620849609375)}, 20: {'weight': np.float64(4.314014697723648)}}, 
    19: {0: {'weight': np.float64(4.544175410919449)}, 1: {'weight': np.float64(4.919584060364029)}, 2: {'weight': np.float64(3.025638127975723)}, 3: {'weight': np.float64(4.855021024399063)}, 4: {'weight': np.float64(4.514994407348892)}, 5: {'weight': np.float64(4.897535825424454)}, 6: {'weight': np.float64(3.7066524273684234)}, 7: {'weight': np.float64(4.489979530029556)}, 8: {'weight': np.float64(4.918733859710953)}, 9: {'weight': np.float64(4.278531814270279)}, 10: {'weight': np.float64(4.8893544918825835)}, 11: {'weight': np.float64(4.07539701461792)}, 12: {'weight': np.float64(5.086940789871475)}, 13: {'weight': np.float64(5.282818818741104)}, 14: {'weight': np.float64(5.3357982880404204)}, 15: {'weight': np.float64(5.140161538772842)}, 16: {'weight': np.float64(4.091082359009048)}, 18: {'weight': np.float64(5.06203987663295)}, 20: {'weight': np.float64(5.210628057174942)}}, 
    20: {0: {'weight': np.float64(4.334909225158951)}, 1: {'weight': np.float64(3.932118201904556)}, 2: {'weight': np.float64(3.7004384994506836)}, 3: {'weight': np.float64(3.046105146408081)}, 4: {'weight': np.float64(3.358980918579361)}, 5: {'weight': np.float64(4.755079055481216)}, 6: {'weight': np.float64(4.086470127105713)}, 7: {'weight': np.float64(4.396703267746231)}, 8: {'weight': np.float64(3.9334490299224854)}, 9: {'weight': np.float64(5.451634908371231)}, 10: {'weight': np.float64(4.791965985946915)}, 11: {'weight': np.float64(4.0853190422058105)}, 12: {'weight': np.float64(4.207467079162598)}, 13: {'weight': np.float64(5.043521428756973)}, 14: {'weight': np.float64(5.172340417557022)}, 15: {'weight': np.float64(3.807558298110962)}, 16: {'weight': np.float64(4.2124409675598145)}, 18: {'weight': np.float64(3.3237369060516357)}, 19: {'weight': np.float64(4.22035026550293)}}
}

# --- ESECUZIONE ---
try:
    ROOT = 17
    arb = chu_liu_edmonds_robust(raw_data, root=ROOT)
    
    # --- STAMPA RISULTATI PULITA ---
    print("\nRISULTATO DELL'ARBORESCENZA:")
    print(f"{'Padre':<6} -> {'Figlio':<6} {'Peso':<10}")
    print("-" * 30)
    
    edges_list = []
    tot = 0
    for u in arb:
        for v, d in arb[u].items():
            edges_list.append((u, v, d['weight']))
            tot += d['weight']
    
    # Ordiniamo per una lettura facile
    edges_list.sort(key=lambda x: x[1]) # Ordina per figlio (così vedi che ogni figlio ha 1 solo padre)
    
    for u, v, w in edges_list:
        print(f"{u:<6} -> {v:<6} {w:.4f}")
        
    print("-" * 30)
    print(f"Peso Totale: {tot:.4f}")

    import networkx as nx

# --- PARTE DA INSERIRE DOPO LA DEFINIZIONE DI raw_data ---

    print("\n" + "="*40)
    print(" VERIFICA CON NETWORKX (LIBRERIA STANDARD) ")
    print("="*40)

    # 1. Convertiamo il dizionario raw_data in un Grafo NetworkX
    # NetworkX è intelligente e capisce la struttura dict-of-dicts
    G = nx.DiGraph(raw_data)

    # 2. Calcoliamo l'Arborescenza Minima
    # NOTA: Usiamo 'minimum_spanning_arborescence' per confrontarlo con il nostro calcolo.
    # Se usassi 'maximum', otterresti l'albero più pesante, non quello più economico.
    arb_nx = nx.minimum_spanning_arborescence(G, attr='weight', preserve_attrs=True)
    # 3. Stampa e Confronto
    print(f"{'Padre':<6} -> {'Figlio':<6} {'Peso':<10}")
    print("-" * 30)

    tot_weight_nx = 0.0
    edges_nx = list(arb_nx.edges(data=True))
    edges_nx.sort(key=lambda x: x[1]) # Ordiniamo per figlio per leggibilità

    for u, v, data in edges_nx:
        w = data['weight']
        tot_weight_nx += w
        print(f"{u:<6} -> {v:<6} {w:.4f}")

    print("-" * 30)
    print(f"Peso Totale NetworkX: {tot_weight_nx:.4f}")

    # Check automatico
    if abs(tot_weight_nx - 14.404) < 0.01:
        print("\n✅ SUCCESSO: Il risultato di NetworkX conferma il calcolo teorico!")
    else:
        print("\n❌ ATTENZIONE: I risultati divergono. Controlla se volevi Min o Max.")
    
except Exception as e:
    print(f"\nERRORE DURANTE L'ESECUZIONE:\n{e}")