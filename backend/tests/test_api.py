import pytest
import os
from httpx import AsyncClient, ASGITransport
from api import app

# Ensure we are testing with the real app, but we might want to be careful 
# about "no side effects". The user said "ensure there are no side effects".
# The /health endpoint is safe. 
# The /graph endpoint is read-only.
# The /ingest endpoint HAS side effects (writes to DB). We should probably avoid testing that 
# in the build unless we clear it after, but we can't easily guarantee cleanup in a container run unless we control it perfectly.
# For now, we stick to safe endpoints.

@pytest.mark.asyncio
async def test_health_check():
    # Use ASGITransport for newer httpx versions
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_graph_endpoint_structure():
    """
    Verifies that the graph endpoint returns the correct structure.
    This effectively tests that the pipeline storage and basic logic are working.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/graph")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "results" in data
    results = data["results"]
    assert "entities" in results
    assert "relations" in results
    assert isinstance(results["entities"], list)
    assert isinstance(results["relations"], list)

@pytest.mark.asyncio
async def test_gemini_api_key_loaded():
    """
    Verifies that the Google API Key is actually loaded in the environment.
    This is critical because Cloud Build often fails to pass secrets.
    """
    # This test will fail if the GOOGLE_API_KEY env var is missing in the container
    api_key = os.getenv("GOOGLE_API_KEY")
    assert api_key is not None, "GOOGLE_API_KEY environment variable is missing!"
    assert len(api_key) > 0, "GOOGLE_API_KEY is empty!"

