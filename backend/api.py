import json
import os
import sqlite3
import logging
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from src.pipeline import GraphRAGPipeline
from datetime import timedelta

# Import auth functions
from src.auth import get_db, get_password_hash, verify_password, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

load_dotenv()

# Setup API logging
logger = logging.getLogger("API")

app = FastAPI(title="GraphRAG API")

# Configure CORS (Relaxed for portfolio project)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication Endpoints ---

class UserAuth(BaseModel):
    username: str
    password: str

@app.post("/api/auth/register")
def register(user: UserAuth):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                       (user.username, get_password_hash(user.password)))
        conn.commit()
        return {"status": "success", "message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    finally:
        conn.close()

@app.post("/api/auth/login")
def login(user: UserAuth):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (user.username,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None or not verify_password(user.password, row[0]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- Graph Pipeline Dependency ---

def get_pipeline(username: str = Depends(get_current_user)):
    user_db_path = f"db/user_{username}_db"
    # Ensure DB directory exists
    os.makedirs(os.path.dirname(user_db_path), exist_ok=True)
    
    pipeline = GraphRAGPipeline(
        db_path=user_db_path,
        clear_db=False 
    )
    return pipeline

class IngestRequest(BaseModel):
    text: str
    clear_db: Optional[bool] = False

class QueryRequest(BaseModel):
    query: str
    query_type: Optional[str] = "local" # or "structural"

# --- API Endpoints ---

@app.get("/api/health")
async def health_check():
    vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{vllm_url}/models", timeout=2.0)
            if response.status_code == 200:
                return {"status": "healthy", "llm": "connected"}
    except Exception:
        pass
    
    return {
        "status": "warming_up", 
        "llm": "loading", 
        "message": "LLM server is starting up (Model loading typically takes 2-5 mins)..."
    }

@app.post("/api/ingest")
async def ingest(request: IngestRequest, pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    async def event_generator():
        try:
            async for update in pipeline.ingest_async(request.text, clear_db=request.clear_db):
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            logger.exception("Ingestion failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/clear_db")
def clear_db(pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    try:
        pipeline.storage.clear()
        return {"status": "success", "message": "Database cleared successfully."}
    except Exception as e:
        logger.exception("Failed to clear database")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query(request: QueryRequest, pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    try:
        results = await pipeline.query(request.query, query_type=request.query_type)
        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph")
def get_graph(pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    try:
        data = pipeline.storage.get_full_graph()
        return {"status": "success", "results": data}
    except Exception as e:
        logger.exception("Failed to get graph data")
        raise HTTPException(status_code=500, detail=str(e))

# --- Static Frontend Serving ---
frontend_path = os.getenv("FRONTEND_PATH", os.path.join(os.path.dirname(__file__), "../frontend/static"))

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    
    @app.get("/")
    def read_index():
        return FileResponse(os.path.join(frontend_path, "index.html"))
    
    @app.get("/config.js")
    def read_config():
        return Response(content="window.BACKEND_URL = '/api';", media_type="application/javascript")
else:
    logger.warning(f"Frontend path {frontend_path} not found. Static files will not be served.")
