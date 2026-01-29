import pytest
import os
from httpx import AsyncClient, ASGITransport
from api import app

@pytest.mark.asyncio
async def test_health_check():
    # Use ASGITransport for newer httpx versions
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_graph_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/graph")
    assert response.status_code == 200
    assert "status" in response.json()
