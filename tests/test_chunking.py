import pytest
from src.chunking import AncestryInjector, PerplexityChunker, EnrichedSentence
from src.ingestion import DocumentNode
from src.llm_service import LLMBackend, GeminiBackend

class MockLLM(LLMBackend):
    def generate(self, prompt: str, schema=None):
        return ""
    async def generate_async(self, prompt: str, schema=None):
        return ""
    def embed(self, text):
        return []
    def get_perplexity(self, text):
        return 1.0
    def check_causal_link(self, context, sentence):
        return 0.5

def test_ancestry_injector():
    # Setup tree
    # Root -> H1 -> Content("A. B.")
    #      -> H2 -> Content("C.")
    
    root = DocumentNode(id="r", text="ROOT", line_ids=[], metadata={"type": "header"})
    h1 = DocumentNode(id="h1", text="Section 1", line_ids=[], metadata={"type": "header"})
    c1 = DocumentNode(id="c1", text="Sentence A. Sentence B.", line_ids=[], metadata={"type": "content"})
    
    h2 = DocumentNode(id="h2", text="Section 2", line_ids=[], metadata={"type": "header"})
    c2 = DocumentNode(id="c2", text="Sentence C.", line_ids=[], metadata={"type": "content"})
    
    # Link
    root.children = [h1]
    h1.children = [c1, h2] # H2 is child of H1
    h2.children = [c2]
    
    injector = AncestryInjector()
    sentences = injector.inject([root])
    
    # Check
    # 1. Sentence A. Context: ROOT > Section 1
    # 2. Sentence B. Context: ROOT > Section 1
    # 3. Sentence C. Context: ROOT > Section 1 > Section 2
    
    assert len(sentences) == 3
    assert sentences[0].text == "Sentence A."
    assert sentences[0].context == "ROOT > Section 1"
    assert sentences[2].text == "Sentence C."
    assert sentences[2].context == "ROOT > Section 1 > Section 2"

def test_perplexity_chunker():
    llm = MockLLM()
    chunker = PerplexityChunker(llm, max_tokens=10) # Small token limit to force split
    
    sentences = [
        EnrichedSentence("Short one.", "ctx", "1"),
        EnrichedSentence("Short two.", "ctx", "1"),
        EnrichedSentence("Very long sentence that should trigger a split because it exceeds the limit.", "ctx", "1")
    ]
    
    chunks = chunker.chunk(sentences)
    
    assert len(chunks) >= 2
    assert chunks[0].sentences[0].text == "Short one."
