import os
import sys
import asyncio
import json
from datetime import timedelta

# Ensure backend directory is in sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from src.storage import GraphStorage
from src.extraction import Entity, Relation
from src.community import detect_communities
from src.chunking import ContextualChunker, FinalChunk
from src.langgraph_rag import LangGraphRAGEngine
from starlette.testclient import TestClient
from api import app

print("=" * 60)
print("🚀 RUNNING FULL END-TO-END SYSTEM VERIFICATION")
print("=" * 60)

# 1. Test FastAPI & Static File Serving
print("\n[1/5] Testing FastAPI Endpoints & Static Files...")
client = TestClient(app)

# Health endpoint
health_res = client.get("/api/health")
assert health_res.status_code == 200, f"Health check failed: {health_res.text}"
health_data = health_res.json()
assert health_data["status"] == "healthy", f"Status not healthy: {health_data}"
print(f"  ✓ /api/health returned: {health_data}")

# Config.js endpoint
cfg_res = client.get("/config.js")
assert cfg_res.status_code == 200, f"Config.js failed: {cfg_res.text}"
assert "window.BACKEND_URL" in cfg_res.text
print(f"  ✓ /config.js served correctly: {cfg_res.text.strip()}")

# Static root endpoint
root_res = client.get("/")
assert root_res.status_code == 200, f"Root index failed: {root_res.text}"
assert "GraphRAG" in root_res.text
print("  ✓ Static index.html served correctly")

# 2. Test Authentication (Register & Login)
print("\n[2/5] Testing User Registration and Authentication...")
test_user = "verifier_admin"
test_pass = "TestSecurePassword2026!"

# Register
reg_res = client.post("/api/auth/register", json={"username": test_user, "password": test_pass})
assert reg_res.status_code in [200, 400], f"Registration error: {reg_res.text}"
print(f"  ✓ Registration endpoint response: {reg_res.json()}")

# Login
login_res = client.post("/api/auth/login", json={"username": test_user, "password": test_pass})
assert login_res.status_code == 200, f"Login failed: {login_res.text}"
token = login_res.json().get("access_token")
assert token is not None, "Access token missing"
auth_headers = {"Authorization": f"Bearer {token}"}
print(f"  ✓ Login succeeded, JWT issued: {token[:20]}...")

# 3. Test LadybugDB Embedded Storage & CRUD
print("\n[3/5] Testing LadybugDB Embedded Graph Database...")
import tempfile
with tempfile.TemporaryDirectory() as tmp_dir:
    db_path = os.path.join(tmp_dir, "test_ladybug_db")
    storage = GraphStorage(db_path=db_path, clear_existing=True)

    # Ingest entities and relations
    e1 = Entity(id="elon_musk", type="Person", description="CEO of SpaceX and Tesla", metadata={"name": "Elon Musk"})
    e2 = Entity(id="spacex", type="Company", description="Aerospace manufacturer", metadata={"name": "SpaceX"})
    e3 = Entity(id="starship", type="Vehicle", description="Super heavy-lift launch vehicle", metadata={"name": "Starship"})
    
    r1 = Relation(source_id="elon_musk", target_id="spacex", type="FOUNDED", description="Founded SpaceX in 2002")
    r2 = Relation(source_id="spacex", target_id="starship", type="MANUFACTURES", description="Builds Starship in Starbase, Texas")

    storage.ingest([e1, e2, e3], [r1, r2])
    print("  ✓ Entities and relations ingested into LadybugDB")

    # Verify nodes
    node = storage.get_node("elon_musk")
    assert node is not None and node["name"] == "Elon Musk"
    
    # Query by name
    matches = storage.get_nodes_by_name("spacex")
    assert len(matches) >= 1 and matches[0]["id"] == "spacex"

    # Query local neighborhood
    local_rels = storage.query_local("spacex")
    assert len(local_rels) >= 2, f"Expected at least 2 edges, got {len(local_rels)}"
    print(f"  ✓ Cypher local neighborhood query returned {len(local_rels)} connected relationships")

    # Export full graph
    graph = storage.get_full_graph()
    assert len(graph["entities"]) == 3
    assert len(graph["relations"]) == 2
    print(f"  ✓ Full graph export verified: {len(graph['entities'])} nodes, {len(graph['relations'])} edges")

# 4. Test Anthropic Contextual Retrieval Chunking
print("\n[4/5] Testing Anthropic Contextual Retrieval Chunker...")
class MockContextualLLM:
    async def generate_async(self, prompt, **kwargs):
        return "Context: Overview of aerospace exploration and rocket development."

chunker = ContextualChunker(llm=MockContextualLLM(), target_chunk_tokens=25, max_chunk_tokens=35)

sample_document = (
    "SpaceX was founded in 2002 by Elon Musk with the goal of reducing space transportation costs.\n\n"
    "The Starship spacecraft and Super Heavy rocket represent a fully reusable transportation system designed to carry both crew and cargo to Earth orbit, the Moon, Mars and beyond.\n\n"
    "During Flight 4 in 2024, the vehicle demonstrated controlled reentry through the atmosphere, withstanding peak heating conditions."
)

async def test_chunking_run():
    chunks = []
    async for item in chunker.chunk_async(sample_document):
        if not (isinstance(item, dict) and item.get("type") == "progress"):
            chunks = item
    return chunks

chunks = asyncio.run(test_chunking_run())
assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
for c in chunks:
    assert "Context:" in c.text, "Contextual prefix missing from chunk"
    assert c.context != "", "Chunk context empty"
print(f"  ✓ ContextualChunker generated {len(chunks)} chunks with prompt-cached contextual enrichment")

# 5. Test LangGraph Multi-Agent Adaptive Hybrid RAG
print("\n[5/5] Testing LangGraph Multi-Agent Hybrid RAG Engine...")
with tempfile.TemporaryDirectory() as tmp_dir:
    db_path = os.path.join(tmp_dir, "test_lg_db")
    storage = GraphStorage(db_path=db_path, clear_existing=True)
    
    e1 = Entity(id="saturn_v", type="Rocket", description="Historic Moon rocket", metadata={"name": "Saturn V"})
    e2 = Entity(id="apollo_11", type="Mission", description="First crewed Moon landing", metadata={"name": "Apollo 11"})
    r1 = Relation(source_id="saturn_v", target_id="apollo_11", type="LAUNCHED", description="Launched Apollo 11 in 1969")
    storage.ingest([e1, e2], [r1])

    class MockRAGLLM:
        async def generate_async(self, prompt, schema=None, **kwargs):
            if schema and hasattr(schema, "model_fields") and "entities" in schema.model_fields:
                return {"entities": ["Saturn V", "Apollo 11"], "route": "hybrid"}
            return "Saturn V was the rocket that launched Apollo 11 to the Moon in 1969."

    rag_engine = LangGraphRAGEngine(storage=storage, llm=MockRAGLLM())
    
    async def test_rag():
        return await rag_engine.run("How are Saturn V and Apollo 11 related?", query_type="hybrid")

    rag_result = asyncio.run(test_rag())
    assert "answer" in rag_result
    assert "Saturn V was the rocket" in rag_result["answer"]
    assert len(rag_result["entities"]) >= 2
    assert len(rag_result["relations"]) >= 1
    assert rag_result["is_grounded"] is True
    print(f"  ✓ LangGraph Route: {rag_result.get('route')}")
    print(f"  ✓ LangGraph Answer: {rag_result.get('answer')}")
    print(f"  ✓ Grounded Evaluation: {rag_result.get('is_grounded')}")

print("\n" + "=" * 60)
print("🎉 ALL 5 SUBSYSTEM VERIFICATIONS PASSED SUCCESSFULLY!")
print("=" * 60)
