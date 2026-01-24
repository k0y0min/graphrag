import networkx as nx
from typing import List, Dict, Any
from src.extraction import Entity, Relation
from networkx.algorithms.community import greedy_modularity_communities
import logging

logger = logging.getLogger(__name__)

def detect_communities(entities: List[Entity], relations: List[Relation]) -> Dict[str, int]:
    """
    Builds a graph from entities and relations, runs community detection,
    and returns a mapping of entity_id -> community_id.
    """
    G = nx.Graph()
    
    # Add nodes
    for e in entities:
        G.add_node(e.id)
    
    # Add edges
    for r in relations:
        G.add_edge(r.source_id, r.target_id)
        
    logger.info(f"Running community detection on graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Detect communities (Greedy Modularity is standard in NX)
    # Returns list of sets of nodes
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
