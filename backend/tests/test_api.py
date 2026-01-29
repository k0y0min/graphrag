import pytest
import os
from httpx import AsyncClient
from api import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_graph_endpoint():
    # This will now use the real API key if provided in the environment
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/graph")
    assert response.status_code == 200
    assert "status" in response.json()
