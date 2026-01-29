import json
import os
import logging
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from src.pipeline import GraphRAGPipeline

load_dotenv()

# Setup API logging
logger = logging.getLogger("API")

app = FastAPI(title="GraphRAG API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline with startup clear
# Ensuring a fresh state for Cloud Run environments
pipeline = GraphRAGPipeline(
    db_path=os.getenv("DB_PATH", "demo_db"),
    clear_db=os.getenv("CLEAR_DB_ON_STARTUP", "true").lower() == "true"
)

class IngestRequest(BaseModel):
    text: str
    clear_db: Optional[bool] = False

class QueryRequest(BaseModel):
    query: str
    query_type: Optional[str] = "local" # or "structural"

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/ingest")
async def ingest(request: IngestRequest):
    async def event_generator():
        try:
            async for update in pipeline.ingest_async(request.text, clear_db=request.clear_db):
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            logger.exception("Ingestion failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/clear_db")
def clear_db():
    try:
        pipeline.storage.clear()
        return {"status": "success", "message": "Database cleared successfully."}
    except Exception as e:
        logger.exception("Failed to clear database")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(request: QueryRequest):
    try:
        results = pipeline.query(request.query, query_type=request.query_type)
        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
def get_graph():
    try:
        # Use existing singleton storage from pipeline
        data = pipeline.storage.get_full_graph()
        return {"status": "success", "results": data}
    except Exception as e:
        logger.exception("Failed to get graph data")
        raise HTTPException(status_code=500, detail=str(e))


