import os
import sys
import pytest
from datetime import timedelta, timezone, datetime

# Add backend directory to sys.path so imports work seamlessly
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from src.auth import get_password_hash, verify_password, create_access_token, ALGORITHM, SECRET_KEY
import jwt
from src.community import detect_communities
from src.chunking import ContextualChunker, AncestryInjector, EnrichedSentence
from src.extraction import Entity, Relation
from src.storage import GraphStorage

def test_auth_password_hashing():
    pwd = "supersecretpassword123!"
    hashed = get_password_hash(pwd)
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("wrongpassword", hashed) is False

def test_auth_jwt_token_generation():
    username = "test_user_ai"
    token = create_access_token({"sub": username}, expires_delta=timedelta(minutes=30))
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload.get("sub") == username
    assert "exp" in payload

def test_community_detection():
    # Construct a simple two-cluster graph: (1-2-3) and (4-5-6), with one bridge (3-4)
    nodes = ["n1", "n2", "n3", "n4", "n5", "n6"]
    edges = [
        ("n1", "n2"), ("n2", "n3"), ("n1", "n3"),
        ("n4", "n5"), ("n5", "n6"), ("n4", "n6"),
        ("n3", "n4")
    ]
    community_map = detect_communities(nodes, edges)
    assert len(community_map) == len(nodes)
    assert all(nid in community_map for nid in nodes)
    # n1 and n2 should belong to the same community
    assert community_map["n1"] == community_map["n2"]
    # n5 and n6 should belong to the same community
    assert community_map["n5"] == community_map["n6"]

def test_community_detection_empty_edges():
    nodes = ["iso1", "iso2"]
    edges = []
    community_map = detect_communities(nodes, edges)
    assert len(community_map) == 2
    assert "iso1" in community_map and "iso2" in community_map

def test_contextual_chunker_splitting():
    class MockLLM:
        async def generate_async(self, prompt, **kwargs):
            return "Contextual summary for test chunk"

    chunker = ContextualChunker(llm=MockLLM(), target_chunk_tokens=50, max_chunk_tokens=100)
    
    text = (
        "Paragraph 1 discusses artificial intelligence and retrieval systems.\n\n"
        "Paragraph 2 dives deeper into graph neural networks and knowledge graphs.\n\n"
        "Paragraph 3 covers vector embeddings, vector databases, and semantic similarity search."
    )
    splits = chunker._split_into_base_chunks(text)
    assert len(splits) >= 1
    assert all(isinstance(s, str) for s in splits)

def test_graph_storage_crud(tmp_path):
    test_db_dir = str(tmp_path / "test_kuzu_db")
    storage = GraphStorage(db_path=test_db_dir, clear_existing=True)

    # Ingest entities
    e1 = Entity(id="ent_1", type="Person", description="A research scientist", metadata={"name": "Alice"})
    e2 = Entity(id="ent_2", type="Company", description="An AI research lab", metadata={"name": "DeepAI"})
    r1 = Relation(source_id="ent_1", target_id="ent_2", type="WORKS_FOR", description="Principal Investigator")

    storage.ingest([e1, e2], [r1])

    # Query node
    node = storage.get_node("ent_1")
    assert node is not None
    assert node["name"] == "Alice"
    assert node["type"] == "Person"

    # Query by name
    matched = storage.get_nodes_by_name("alice")
    assert len(matched) >= 1
    assert matched[0]["id"] == "ent_1"

    # Query local neighborhood
    local_rels = storage.query_local("ent_1")
    assert len(local_rels) >= 1

    # Full graph export
    full = storage.get_full_graph()
    assert len(full["entities"]) == 2
    assert len(full["relations"]) == 1
