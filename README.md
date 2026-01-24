# GraphRAG DevOps Deployment Guide

This project transforms the initial GraphRAG experiment into a production-grade service using Docker, FastAPI, and GitHub Actions.

## Features
- **API Access**: Interact with GraphRAG via RESTful endpoints.
- **Dockerized**: Easy deployment anywhere with Docker support.
- **CI/CD**: Automated linting, testing, and container builds on every push.
- **Persistent Storage**: KuzuDB data persists across container restarts via volumes.

## Getting Started

### Local Setup (Non-Docker)
```bash
cd custom_graphrag
pip install -r requirements.txt
uvicorn api:app --reload
```

### Docker Setup
```bash
docker-compose up --build
```

## API Endpoints

### 1. Ingest Text
**POST** `/ingest`
```json
{
  "text": "Your long document content here...",
  "clear_db": false
}
```

### 2. Query Entity
**POST** `/query`
```json
{
  "entity_id": "Apple",
  "query_type": "local"
}
```

### 3. Health Check
**GET** `/health`

## CI/CD
The project includes GitHub Actions:
- **CI**: Runs `ruff` for linting and `pytest` for unit tests.
- **CD**: Builds a Docker image and pushes it to GHCR (GitHub Container Registry).

## Deployment to Render
1. Connect your GitHub repo to Render.
2. Select **Web Service**.
3. Choose **Docker** as the runtime.
4. Add your `GEMINI_API_KEY` to environment variables.
5. Deploy!
