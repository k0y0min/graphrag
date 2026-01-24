import pytest
import os
from src.ingestion import LineMap, HierarchyDetector, DocumentTreeBuilder
from src.llm_service import LLMBackend
from typing import Any, Dict, List, Optional

class MockLLM(LLMBackend):
    def generate(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        # Simple mock response based on prompt or just deterministic return
        # For HierarchyDetector, we expect mappings
        return {
            "mappings": [
                {"id": 1, "role": "Header"},
                {"id": 2, "role": "Text"},
                {"id": 3, "role": "List_Item"},
                {"id": 4, "role": "Header"}, # Nested header?
                {"id": 5, "role": "Text"}
            ]
        }

    async def generate_async(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        return self.generate(prompt, schema)

    def embed(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]
    def get_perplexity(self, text: str) -> float:
        return 1.0
    def check_causal_link(self, context: str, sentence: str) -> float:
        return 0.5

@pytest.fixture
def sample_file(tmp_path):
    p = tmp_path / "sample.md"
    content = """# Header 1
Content under header 1.
- List item 1
## Header 2
Content under header 2."""
    p.write_text(content, encoding='utf-8')
    return str(p)

def test_line_map(sample_file):
    lm = LineMap(sample_file)
    assert lm.total_lines == 5
    assert lm.get_line(1) == "# Header 1"
    assert lm.get_line(5) == "Content under header 2."
    
    batch = lm.get_batch(1, 2)
    assert len(batch) == 2
    assert batch[0].content == "# Header 1"

def test_hierarchy_detector(sample_file):
    lm = LineMap(sample_file)
    llm = MockLLM()
    detector = HierarchyDetector(llm)
    roles = detector.detect(lm, batch_size=5)
    
    assert len(roles) == 5
    assert roles[1] == "Header"
    assert roles[2] == "Text"

def test_document_tree_builder(sample_file):
    lm = LineMap(sample_file)
    # Manual roles to ensure test correctness independent of LLM
    roles = {
        1: "Header",
        2: "Text",
        3: "List_Item",
        4: "Header",
        5: "Text"
    }
    
    builder = DocumentTreeBuilder()
    nodes = builder.build(lm, roles)
    
    # Structure should be:
    # Root
    #   Header 1 (Node 1)
    #     Content (Node 2, 3)
    #     Header 2 (Node 4) -> Wait, H2 is child of H1?
    #       Content (Node 5)
    
    # In Markdown, H2 is sub-section of H1.
    # Our simple logic:
    # "# Header 1" -> level 1
    # "## Header 2" -> level 2
    
    # Let's check nodes.
    # Node for Line 1 (Header 1)
    # The builder returns root children (Level 1 nodes)
    assert len(nodes) == 1
    h1 = nodes[0]
    assert h1.line_ids == [1]
    assert h1.depth == 1
    assert h1.parent_id == "_root"
    
    # H1 children should be Content(2,3) and Header2(4)
    assert len(h1.children) == 2
    
    # Check Content (lines 2, 3)
    c1 = h1.children[0]
    assert 2 in c1.line_ids
    assert c1.parent_id == h1.id
    
    # Check H2 (line 4)
    h2 = h1.children[1]
    assert h2.line_ids == [4]
    assert h2.parent_id == h1.id
    
    # H2 children should be Content(5)
    assert len(h2.children) == 1
    c2 = h2.children[0]
    assert 5 in c2.line_ids
    assert c2.parent_id == h2.id

@pytest.mark.asyncio
async def test_hierarchy_detector_async(sample_file):
    lm = LineMap(sample_file)
    llm = MockLLM()
    detector = HierarchyDetector(llm)
    
    # We rely on pre-scan for headers in async
    # # Header 1
    # ## Header 2
    # So both should be detected as headers deterministically even without LLM if logic holds,
    # but let's see if our mock LLM response is used. 
    # Logic: 
    # 1. Deterministic Overrides (Markdown) -> priority
    # 2. LLM result
    
    # In sample file:
    # 1: # Header 1 (Method: Deterministic)
    # 2: Content (Method: LLM -> "Text")
    # 3: - List item (Method: Deterministic)
    # 4: ## Header 2 (Method: Deterministic)
    # 5: Content (Method: LLM -> "Text")
    
    roles = await detector.detect_async(lm, batch_size=2)
    
    assert len(roles) == 5
    assert roles[1] == "Header"
    assert roles[2] == "Text"
    assert roles[3] == "List_Item"
    assert roles[4] == "Header"
    assert roles[5] == "Text"
