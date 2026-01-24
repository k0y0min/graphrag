from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Any
from src.pipeline import GraphRAGPipeline
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GraphRAG API")

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
def ingest(request: IngestRequest):
    try:
        results = pipeline.ingest_text(request.text, clear_db=request.clear_db)
        return {"status": "success", "results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_db")
def clear_db():
    try:
        pipeline.storage.clear()
        return {"status": "success", "message": "Database cleared successfully."}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(request: QueryRequest):
    try:
        results = pipeline.query(request.query, query_type=request.query_type)
        return {"status": "success", "results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
def get_graph():
    try:
        # Use existing singleton storage from pipeline
        data = pipeline.storage.get_full_graph()
        return {"status": "success", "results": data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
