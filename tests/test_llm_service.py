import pytest
import os
from src.llm_service import GeminiBackend

@pytest.fixture
def backend():
    api_key = os.getenv("GOOGLE_API_KEY")
    # Basic check to skip if no key, but we expect .env to be present.
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    return GeminiBackend()

def test_generate_text(backend):
    response = backend.generate("Say 'Hello' and nothing else.")
    assert "Hello" in response

def test_generate_json(backend):
    # Verify strict JSON output
    # Note: TypedDict or Pydantic models are better for schemas, keeping it simple for now.
    import typing_extensions as typing

    class Fruit(typing.TypedDict):
        name: str
        color: str

    response = backend.generate(
        "Return a JSON object for a red apple.", 
        schema=Fruit
    )
    assert isinstance(response, dict)
    assert "apple" in response['name'].lower()
    assert response['color'].lower() == "red"

def test_embed(backend):
    vector = backend.embed("Hello world")
    assert isinstance(vector, list)
    assert len(vector) > 0
