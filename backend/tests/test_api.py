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
async def test_get_graph_empty():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/graph")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
