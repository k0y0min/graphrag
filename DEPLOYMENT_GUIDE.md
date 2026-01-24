# GraphRAG Deployment Guide

This guide details how to deploy the GraphRAG demo to **Google Cloud Run** using GitHub Actions.

## Prerequisites

1.  **Google Cloud Platform Project**: Create a project with billing enabled.
2.  **APIs Enabled**: Enable "Cloud Run Admin API" and "Artifact Registry API".
3.  **Service Account**: Create a Service Account with `Cloud Run Admin` and `Service Account User` roles.

## 1. Local Setup

Your project is configured with `custom_graphrag` as the root.

### Environment
The `.env` file and `models/` directory are **ignored** by git for security and size.
- `venv` is for local development only.
- The Docker image will **bake in** the Qwen model (~1.5GB) to ensure fast startup times (averting >60s download delays).

### Testing Locally
Run strict tests before pushing:
```bash
pytest tests/
```

Run Docker locally to simulate Cloud Run:
```bash
docker build -t graphrag-demo .
docker run -p 8080:8080 -e PORT=8080 -e GEMINI_API_KEY=your_key graphrag-demo
```

## 2. GitHub Secrets

Go to your GitHub Repository -> Settings -> Secrets and variables -> Actions.
Add the following secrets:

- `GEMINI_API_KEY`: Your Google Gemini API Key.
- `GCP_SA_KEY`: The JSON key of your Google Cloud Service Account.

## 3. CI/CD Pipeline

 The workflows are located in `.github/workflows/`:
 
- **CI (`ci.yml`)**: Runs on every push. Installs dependencies and runs `pytest`.
- **CD (`cd.yml`)**: Runs on push to `main`.
    1.  Builds the Docker image (including the model).
    2.  Pushes to GitHub Container Registry (GHCR).
    3.  Deploys to Google Cloud Run `us-central1` as `graphrag-demo`.

### Deployment Configuration
- **Memory**: The Qwen model + app requires memory. Cloud Run default is 512MB which is too low.
    - **Recommended**: 4GB+ RAM. 
    - *Note*: Use the `gcloud run services update graphrag-demo --memory 4Gi` command if the initial deploy fails on OOM, or update the CD workflow args.

- **Persistence**:
    - By default, the database (`demo_db`) **clears on startup** (`CLEAR_DB_ON_STARTUP=true`) to ensure a fresh session for each demo user.
    - To persist data, set `CLEAR_DB_ON_STARTUP=false` in Cloud Run environment variables.

## 4. Manual Deployment (Optional)

If you prefer to deploy manually from your terminal:

```bash
gcloud run deploy graphrag-demo \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --set-env-vars "GEMINI_API_KEY=$GEMINI_API_KEY,CLEAR_DB_ON_STARTUP=true"
```
