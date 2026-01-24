import pytest
from src.extraction import GraphExtractor, Entity, Relation
from src.chunking import FinalChunk
from src.ingestion import DocumentNode
from src.llm_service import LLMBackend

class MockLLM(LLMBackend):
    def generate(self, prompt: str, schema=None):
        return {
            "entities": [
                {"id": "Apple", "type": "Company", "description": "Tech company"},
                {"id": "iPhone", "type": "Product", "description": "Phone"}
            ],
            "relations": [
                {"source": "Apple", "target": "iPhone", "type": "MAKES", "description": "Manufactures"}
            ]
        }
    async def generate_async(self, prompt: str, schema=None):
        return self.generate(prompt, schema)
    def embed(self, text):
        return []
    def get_perplexity(self, text):
        return 1.0
    def check_causal_link(self, context, sentence):
        return 0.5

def test_graph_extractor():
    llm = MockLLM()
    extractor = GraphExtractor(llm)
    
    # 1. Setup structural nodes
    root = DocumentNode(id="root", text="ROOT", line_ids=[])
    h1 = DocumentNode(id="h1", text="Header 1", line_ids=[], metadata={"type": "header"}, parent_id="root")
    c1 = DocumentNode(id="c1", text="Content", line_ids=[], metadata={"type": "content"}, parent_id="h1")
    nodes = [root, h1, c1]
    
    # 2. Setup semantic chunks
    chunk = FinalChunk(
        id="ch1",
        text="Apple makes iPhone.",
        context="Header 1",
        start_char_idx=0,
        end_char_idx=10,
        sentences=[]
    )
    
    # Extract
    entities, relations = extractor.extract([chunk], nodes)
    
    # Verify Structural Entities
    # h1 and c1 should be entities
    ent_ids = {e.id for e in entities}
    assert "h1" in ent_ids
    assert "c1" in ent_ids
    
    # Verify Structural Relations
    # h1 -> c1 (PARENT_OF)
    struc_rels = [r for r in relations if r.type == "PARENT_OF"]
    assert len(struc_rels) >= 1
    assert struc_rels[0].source_id == "h1" or struc_rels[0].source_id == "root" # Check logic
    # Actually root -> h1 is a relation if root is treated as entity and parent is not "root" ?
    # In code: `if node.parent_id and node.parent_id != "root":`
    # So ROOT -> H1 is excluded if parent_id is "root".
    # But H1 -> C1 is included.
    
    parent_of = next(r for r in struc_rels if r.source_id == "h1" and r.target_id == "c1")
    assert parent_of is not None

    # Verify Semantic Entities
    assert "Apple" in ent_ids
    assert "iPhone" in ent_ids
    
    # Verify Semantic Relations
    sem_rels = [r for r in relations if r.type == "MAKES"]
    assert len(sem_rels) == 1
    assert sem_rels[0].source_id == "Apple"
    assert sem_rels[0].target_id == "iPhone"

@pytest.mark.asyncio
async def test_extractor_async():
    llm = MockLLM()
    extractor = GraphExtractor(llm)
    
    # 1. Setup structural nodes
    root = DocumentNode(id="root", text="ROOT", line_ids=[])
    h1 = DocumentNode(id="h1", text="Header 1", line_ids=[], metadata={"type": "header"}, parent_id="root")
    c1 = DocumentNode(id="c1", text="Content", line_ids=[], metadata={"type": "content"}, parent_id="h1")
    nodes = [root, h1, c1]
    
    # 2. Setup semantic chunks
    chunk = FinalChunk(
        id="ch1",
        text="Apple makes iPhone.",
        context="Header 1",
        start_char_idx=0,
        end_char_idx=10,
        sentences=[]
    )
    
    # Extract Async
    entities, relations = await extractor.extract_async([chunk], nodes)
    
    # Verify Structural Entities
    ent_ids = {e.id for e in entities}
    assert "h1" in ent_ids
    assert "c1" in ent_ids
    
    # Verify Semantic Entities
    assert "Apple" in ent_ids
    assert "iPhone" in ent_ids
    
    # Verify Semantic Relations
    sem_rels = [r for r in relations if r.type == "MAKES"]
    assert len(sem_rels) == 1
    assert sem_rels[0].source_id == "Apple"
