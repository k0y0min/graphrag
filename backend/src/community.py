import networkx as nx
from typing import List, Dict, Tuple, Any
from networkx.algorithms.community import greedy_modularity_communities
import logging

logger = logging.getLogger(__name__)

def detect_communities(node_ids: List[str], edges: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Builds a graph from node IDs and edges, runs community detection,
    and returns a mapping of node_id -> community_id.
    """
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(node_ids)
    
    # Add edges
    G.add_edges_from(edges)
        
    logger.info(f"Running community detection on graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Detect communities
    try:
        if G.number_of_edges() == 0:
             communities = [set(G.nodes())]
        else:
            communities = list(greedy_modularity_communities(G))
    except Exception as e:
        logger.warning(f"Community detection failed: {e}. Defaulting to single community.")
        communities = [set(G.nodes())]
    
    logger.info(f"Detected {len(communities)} communities.")
    
    # Map node -> community_id
    community_map = {}
    for cid, community_set in enumerate(communities):
        for node in community_set:
            community_map[node] = cid
            
    return community_map
