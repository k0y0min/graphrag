import pytest
import shutil
import os
from src.storage import GraphStorage
from src.extraction import Entity, Relation

@pytest.fixture
def storage(tmp_path):
    db_path = tmp_path / "test_kuzu"
    return GraphStorage(str(db_path), clear_existing=True)

def test_ingest_and_query(storage):
    entities = [
        Entity("e1", "TypeA", "Desc A"),
        Entity("e2", "TypeB", "Desc B"),
        Entity("root", "Root", "The Root")
    ]
    
    relations = [
        Relation("e1", "e2", "LINKS_TO", "Some link"),
        Relation("root", "e1", "PARENT_OF", "")
    ]
    
    storage.ingest(entities, relations)
    
    # Query Local
    results = storage.query_local("e1")
    # e1 links to e2
    assert len(results) == 1
    # row format depends on return. b.id, b.type, b.description, r.type, r.description
    # result[0] is b.id -> "e2"
    row = results[0]
    # Kuzu python API returns list of values for the row if not using as_df
    # Assuming row is list-like or accessible by index
    assert "e2" in row # ID
    assert "LINKS_TO" in row # Rel Type
    
    # Query Structural
    struct_results = storage.query_structural("e1")
    # root is parent of e1
    assert len(struct_results) == 1
    assert "root" in struct_results[0]

def test_duplicates(storage):
    # Test safe merge
    entities = [Entity("e1", "TypeA", "Desc A")]
    storage.ingest(entities, [])
    storage.ingest(entities, []) # Should not error
    
    results = storage.conn.execute("MATCH (a:Entity) RETURN count(a)").get_next()
    assert results[0] == 1
