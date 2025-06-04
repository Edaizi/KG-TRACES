import random
import networkx as nx
from collections import deque
import numpy as np
from typing import List, Tuple, Optional, Dict

from tools.logger_factory import setup_logger

logger = setup_logger("graph_utils")


def build_graph(graph: list) -> nx.Graph:
    G = nx.DiGraph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def bfs_with_rule(graph, start_node, target_rule, max_p = 10):
    result_paths = []
    queue = deque([(start_node, [])])
    while queue:
        current_node, current_path = queue.popleft()

        if len(current_path) == len(target_rule):
            result_paths.append(current_path)

        if len(current_path) < len(target_rule):
            if current_node not in graph:
                continue
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]['relation']
                if rel != target_rule[len(current_path)] or len(current_path) > len(target_rule):
                    continue
                queue.append((neighbor, current_path + [(current_node, rel, neighbor)]))
    
    return result_paths
    

def get_truth_paths(q_entity: list, a_entity: list, graph: nx.Graph) -> list:
    '''
    Get shortest paths connecting question and answer entities.
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_shortest_paths(graph, h, t):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        for i in range(len(p)-1):
            u = p[i]
            v = p[i+1]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths
    
def get_simple_paths(q_entity: list, a_entity: list, graph: nx.Graph, hop=2) -> list:
    '''
    Get all simple paths connecting question and answer entities within given hop
    '''
    # Select paths
    paths = []
    for h in q_entity:
        if h not in graph:
            continue
        for t in a_entity:
            if t not in graph:
                continue
            try:
                for p in nx.all_simple_edge_paths(graph, h, t, cutoff=hop):
                    paths.append(p)
            except:
                pass
    # Add relation to paths
    result_paths = []
    for p in paths:
        result_paths.append([(e[0], graph[e[0]][e[1]]['relation'], e[1]) for e in p])
    return result_paths


def get_negative_paths(q_entity: list, a_entity: list, graph: nx.Graph, n_neg: int, hop=2) -> list:
    '''
    Get negative paths for question witin hop
    '''
    # sample paths
    start_nodes = []
    end_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    for t in a_entity:
        if t in graph:
            end_nodes.append(node_idx.index(t))
    paths = random_walks(graph, n_walks=n_neg, walk_len=hop, start_nodes=start_nodes)
    # Add relation to paths
    result_paths = []
    for p in paths:
        tmp = []
        # remove paths that end with answer entity
        if p[-1] in end_nodes:
            continue
        for i in range(len(p)-1):
            u = node_idx[p[i]]
            v = node_idx[p[i+1]]
            tmp.append((u, graph[u][v]['relation'], v))
        result_paths.append(tmp)
    return result_paths


def random_walks(graph: nx.Graph, n_walks: int, walk_len: int, start_nodes: list[str]) -> list[list[str]]:
    if not start_nodes or n_walks <= 0 or walk_len <= 0:
        return []
    
    valid_start_nodes = [n for n in start_nodes if graph.has_node(n)]
    if not valid_start_nodes:
        print("Error: No valid start nodes in graph")
        return []
    
    unique_paths = set()
    visited_failures = set()
    max_attempts = n_walks * 5
    
    attempt = 0
    while len(unique_paths) < n_walks and attempt < max_attempts:
        attempt += 1
        
        start = random.choice(valid_start_nodes)
        if start in visited_failures:
            continue
            
        path = _single_walk(graph, start, walk_len)
        
        if len(path) < 2:
            visited_failures.add(start)
            continue
            
        path_tuple = tuple(path)
        if path_tuple not in unique_paths:
            unique_paths.add(path_tuple)
            attempt = 0
    
    result = [list(p) for p in unique_paths]
    
    if len(result) < n_walks:
        print(f"Warning: Only found {len(result)} unique paths (requested {n_walks})")
        
    return result[:n_walks]


def _single_walk(graph: nx.Graph, start: str, max_len: int) -> list[str]:
    path = [start]
    current = start
    
    for _ in range(max_len - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        next_node = random.choice([n for n in neighbors if n != path[-1]])
        path.append(next_node)
        current = next_node
        
    return path


def get_random_paths(q_entity: list, graph: nx.Graph, n: int = 3, hop:int = 2) -> tuple [list, list]:
    '''
    Get random paths for question within hop
    '''
    # sample paths
    start_nodes = []
    node_idx = list(graph.nodes())
    for h in q_entity:
        if h in graph:
            start_nodes.append(node_idx.index(h))
    paths = random_walks(graph, n_walks=n, walk_len=hop, start_nodes=start_nodes)
    # Add relation to paths
    result_paths = []
    rules = []
    for p in paths:
        tmp = []
        tmp_rule = []
        for i in range(len(p)-1):
            u = node_idx[p[i]]
            v = node_idx[p[i+1]]
            tmp.append((u, graph[u][v]['relation'], v))
            tmp_rule.append(graph[u][v]['relation'])
        result_paths.append(tmp)
        rules.append(tmp_rule)
    return result_paths, rules



def get_weight_paths(
    q_entity: list[str],
    graph: nx.DiGraph,
    weight_method: str = 'hits',
    top_k: int = 15,
    hits_iter: int = 50,
    tolerance: float = 1e-6,
    aggregate_method: str = 'mean',
    hop: int = 2,
) -> List[Tuple[List[str], float]]:
    hubs, authorities, node_weights = _compute_weights(graph, weight_method, hits_iter, tolerance)
    
    paths = []
    stop_flag = False
    for entity in q_entity:
        if stop_flag:
            break
        if entity not in graph:
            continue    
        for target in graph.nodes():
            if target != entity:
                node_paths = nx.all_simple_paths(graph, entity, target, cutoff=hop)
                for node_path in node_paths:
                    triplet_path = []
                    for u, v in zip(node_path[:-1], node_path[1:]):
                        relation = graph[u][v]['relation']
                        triplet_path.append((u, relation, v))
                    if len(triplet_path)>0:
                        paths.append(triplet_path)
                    if len(paths) >= 10*top_k:
                        stop_flag = True
                        break
                if stop_flag:
                    break
        if stop_flag:
            break


    scored_paths = []
    for path in paths:
        if any(len(triple)<3 for triple in path):
            continue
        
        score = _calculate_path_score(
            path, weight_method, hubs, authorities, node_weights, aggregate_method
        )
        scored_paths.append((path, score))
    
    sorted_paths = sorted(scored_paths, key=lambda x: x[1], reverse=True)[:top_k]
    path_list = [p for p, _ in sorted_paths]
    score_list = [s for _, s in sorted_paths]

    return path_list, score_list, None



def _compute_weights(
    graph: nx.DiGraph,
    weight_method: str,
    hits_iter: int,
    tolerance: float,
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    hubs, authorities, node_weights = None, None, None
    
    if weight_method == 'hits':
        hubs, authorities = compute_hits_scores(graph, hits_iter, tolerance)
    if weight_method == 'pagerank':
        node_weights = nx.pagerank(graph)
    elif weight_method == 'degree':
        node_weights = dict(nx.degree(graph))
    
    return hubs, authorities, node_weights



def _calculate_path_score(
    path: List[str],
    method: str,
    hubs: Optional[Dict],
    authorities: Optional[Dict],
    node_weights: Optional[Dict],
    aggregate: str,
) -> float:
    scores = []
    
    for triple in path:
        h, r, t = triple
        
        if method == "pagerank":
            score = node_weights.get(h, 0.0) + node_weights.get(t, 0.0)
        elif method == "hits":
            score = hubs.get(h, 0.0) + authorities.get(t, 0.0)
        
        scores.append(score)

    if aggregate == "sum":
        return sum(scores)
    elif aggregate == "product":
        return np.prod(scores) if scores else 0.0
    elif aggregate == "avg" or aggregate == "mean":
        return np.mean(scores) if scores else 0.0
    else:
        raise ValueError(f"Invalid aggregate method: {method}")



def compute_hits_scores(
    graph: nx.DiGraph, 
    max_iters: int = 50, 
    tol: float = 1e-6
) -> Tuple[Dict, Dict]:
    nodes = list(graph.nodes())
    A = nx.to_numpy_array(graph, nodelist=nodes)
    h, a = np.ones(len(nodes)), np.ones(len(nodes))
    
    for _ in range(max_iters):
        h_new = A @ a
        a_new = A.T @ h
        h_norm = np.linalg.norm(h_new)
        a_norm = np.linalg.norm(a_new)
        h = h_new / h_norm if h_norm > 0 else h_new
        a = a_new / a_norm if a_norm > 0 else a_new
        
        if np.sum(np.abs(h_new - h)) + np.sum(np.abs(a_new - a)) < tol:
            break
    
    return (
        {node: h[i] for i, node in enumerate(nodes)},
        {node: a[i] for i, node in enumerate(nodes)}
    )
