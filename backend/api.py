import json
import os
import sqlite3
import logging
import base64
import io
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://.*",
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

_user_pipelines: dict[str, GraphRAGPipeline] = {}

def get_pipeline(username: str = Depends(get_current_user)) -> GraphRAGPipeline:
    user_db_path = f"db/user_{username}_db"
    # Ensure DB directory exists
    os.makedirs(os.path.dirname(user_db_path), exist_ok=True)
    
    if username not in _user_pipelines:
        _user_pipelines[username] = GraphRAGPipeline(
            db_path=user_db_path,
            clear_db=False 
        )
    return _user_pipelines[username]

class IngestRequest(BaseModel):
    text: str
    clear_db: Optional[bool] = False
    max_chunk_tokens: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    query_type: Optional[str] = "local" # or "structural"

# --- API Endpoints ---

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "engine": "gemini-3.8-flash",
        "database": "ladybug"
    }

@app.post("/api/ingest")
async def ingest(request: IngestRequest, pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    async def event_generator():
        try:
            async for update in pipeline.ingest_async(request.text, clear_db=request.clear_db, max_chunk_tokens=request.max_chunk_tokens):
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            logger.exception("Ingestion failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

class CypherQueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 100

@app.post("/api/ingest_file")
async def ingest_file(
    request: Request,
    file: Optional[UploadFile] = File(None),
    clear_db: bool = Form(False),
    max_chunk_tokens: Optional[int] = Form(None),
    pipeline: GraphRAGPipeline = Depends(get_pipeline)
):
    """
    Robust single-file multimodal ingestion endpoint.
    Accepts any supported modality: PDF, Image, Audio, or Text/Markdown/Code/JSON/CSV.
    """
    uploaded_file = file
    if uploaded_file is None:
        try:
            form = await request.form()
            for key, val in form.items():
                if hasattr(val, "filename") and getattr(val, "filename"):
                    uploaded_file = val
                    break
            if not clear_db and form.get("clear_db"):
                clear_db = str(form.get("clear_db")).lower() in ["true", "1", "yes"]
            if max_chunk_tokens is None and form.get("max_chunk_tokens"):
                tok_str = str(form.get("max_chunk_tokens"))
                if tok_str.isdigit():
                    max_chunk_tokens = int(tok_str)
        except Exception as e:
            logger.warning(f"Form parsing error: {e}")

    if uploaded_file is None or not getattr(uploaded_file, "filename", None):
        raise HTTPException(status_code=400, detail="No file was provided for ingestion.")

    fname = uploaded_file.filename or "uploaded_document"
    fbytes = await uploaded_file.read()
    ext = os.path.splitext(fname)[1].lower()

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'Parsing', 'current': 0, 'total': 1, 'status': f'Parsing {fname}...' })}\n\n"

            file_text = ""
            # 1. PDF Parsing
            if ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(fbytes))
                    page_texts = []
                    for p_idx, page in enumerate(reader.pages):
                        pt = page.extract_text() or ""
                        if pt.strip():
                            page_texts.append(f"--- Page {p_idx + 1} ---\n{pt}")
                    file_text = "\n\n".join(page_texts)
                    logger.info(f"pypdf extracted {len(file_text)} characters from {fname}")
                except Exception as e:
                    logger.warning(f"pypdf extraction error on {fname}: {e}")

                # If pypdf got no text or minimal text (e.g. scanned image-based PDF or resume)
                if not file_text or len(file_text.strip()) < 30:
                    yield f"data: {json.dumps({'type': 'progress', 'stage': 'Parsing', 'current': 0, 'total': 1, 'status': f'Invoking Gemini Document Vision for visual/scanned PDF: {fname}...' })}\n\n"
                    try:
                        b64_pdf = base64.b64encode(fbytes).decode("utf-8")
                        pdf_messages = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "You are an expert document extractor. Extract all text, factual assertions, "
                                            "timelines, personal details, skills, roles, projects, and relationships from this PDF "
                                            "document into a rich, structured markdown document suitable for Knowledge Graph ingestion."
                                        )
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:application/pdf;base64,{b64_pdf}"}
                                    }
                                ]
                            }
                        ]
                        file_text = await pipeline.llm_service.extractor.generate_async(pdf_messages)
                    except Exception as e2:
                        logger.warning(f"Multimodal PDF fallback failed on {fname}: {e2}")

            # 2. Image Multimodal Vision Extraction via Gemini
            elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'Parsing', 'current': 0, 'total': 1, 'status': f'Analyzing image with Gemini Vision: {fname}...' })}\n\n"
                try:
                    b64_img = base64.b64encode(fbytes).decode("utf-8")
                    img_mime = "image/png" if ext == ".png" else "image/jpeg"
                    vision_messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "You are a multimodal knowledge extractor. Analyze this image thoroughly.\n"
                                        "Extract all factual details, structural components, diagrams, relationships, "
                                        "workflows, timeline steps, or embedded text into a rich, detailed markdown narrative "
                                        "specifically structured for Knowledge Graph indexing."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_mime};base64,{b64_img}"}
                                }
                            ]
                        }
                    ]
                    file_text = await pipeline.llm_service.extractor.generate_async(vision_messages)
                except Exception as e:
                    logger.warning(f"Vision extraction error on {fname}: {e}")

            # 3. Audio Ingestion via Gemini Multimodal Audio
            elif ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'Parsing', 'current': 0, 'total': 1, 'status': f'Transcribing audio with Gemini: {fname}...' })}\n\n"
                try:
                    b64_audio = base64.b64encode(fbytes).decode("utf-8")
                    audio_mime = "audio/wav" if ext == ".wav" else "audio/mp3"
                    audio_messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "You are an expert audio transcription and knowledge extractor. "
                                        "Transcribe this audio recording thoroughly and synthesize all discussed facts, "
                                        "speaker interactions, entity relationships, timeline milestones, and decisions "
                                        "into a rich, structured markdown document specifically structured for Knowledge Graph indexing."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{audio_mime};base64,{b64_audio}"}
                                }
                            ]
                        }
                    ]
                    file_text = await pipeline.llm_service.extractor.generate_async(audio_messages)
                except Exception as e:
                    logger.warning(f"Audio extraction error on {fname}: {e}")

            # 4. Standard Text / Markdown / Code / JSON / CSV
            else:
                try:
                    file_text = fbytes.decode("utf-8", errors="replace")
                except Exception as e:
                    logger.warning(f"Text decode error on {fname}: {e}")

            if not file_text or not file_text.strip():
                raise ValueError(f"Could not extract readable text or content from '{fname}'.")

            final_payload = f"# Document Source: {fname}\n\n{file_text.strip()}"
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'Parsing', 'current': 1, 'total': 1, 'status': f'Successfully parsed {fname} ({len(final_payload)} chars). Starting GraphRAG pipeline...'})}\n\n"

            # Execute graph pipeline
            async for update in pipeline.ingest_async(final_payload, input_filename=fname, clear_db=clear_db, max_chunk_tokens=max_chunk_tokens):
                yield f"data: {json.dumps(update)}\n\n"

        except Exception as e:
            logger.exception("Ingestion of files failed")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/chunks")
def get_chunks(pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    """Returns stored contextual chunks."""
    try:
        db_chunks = pipeline.storage.get_all_chunks()
        if not db_chunks and pipeline.latest_chunks:
            db_chunks = pipeline.latest_chunks
        return {"status": "success", "chunks": db_chunks}
    except Exception as e:
        logger.exception("Failed to get chunks")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/aliases")
def get_aliases(pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    """Returns latest entity resolution and alias disambiguation records."""
    if pipeline.latest_alias_records:
        return {"status": "success", "aliases": pipeline.latest_alias_records}
    
    # Fallback to querying current entities from storage so table is always populated
    full_graph = pipeline.storage.get_full_graph()
    fallback_records = []
    for node in full_graph.get("entities", []):
        name = node.get("name") or node.get("id")
        fallback_records.append({
            "canonical_name": name,
            "assigned_id": node.get("id"),
            "entity_type": node.get("type", "Concept"),
            "aliases": [name],
            "resolution_type": "indexed",
            "description": node.get("description", "")
        })
    return {"status": "success", "aliases": fallback_records}

@app.post("/api/cypher")
def run_cypher(req: CypherQueryRequest, pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    """Executes arbitrary Cypher query on LadybugDB for developer inspection."""
    return pipeline.storage.execute_cypher(req.query, limit=req.limit or 100)

@app.get("/api/stats")
def get_stats(pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    """Returns high-level statistics for the dashboard header."""
    try:
        data = pipeline.storage.get_full_graph()
        chunks = pipeline.storage.get_all_chunks()
        nodes = data.get("entities", [])
        rels = data.get("relations", [])
        communities = {n.get("community_id") for n in nodes if n.get("community_id") is not None and n.get("community_id") != -1}
        return {
            "status": "success",
            "stats": {
                "nodes": len(nodes),
                "relations": len(rels),
                "chunks": len(chunks) if chunks else len(pipeline.latest_chunks),
                "communities": len(communities)
            }
        }
    except Exception as e:
        return {
            "status": "success",
            "stats": {"nodes": 0, "relations": 0, "chunks": 0, "communities": 0}
        }

@app.post("/api/clear_db")
def clear_db(pipeline: GraphRAGPipeline = Depends(get_pipeline)):
    try:
        pipeline.storage.clear()
        pipeline.latest_chunks = []
        pipeline.latest_alias_records = []
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
candidate_paths = [
    os.getenv("FRONTEND_PATH", ""),
    os.path.join(os.path.dirname(__file__), "frontend/static"),
    os.path.join(os.path.dirname(__file__), "../frontend/static"),
    "/app/frontend/static"
]
frontend_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

if frontend_path:
    logger.info(f"Serving static frontend from: {frontend_path}")
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    
    @app.get("/")
    def read_index():
        return FileResponse(os.path.join(frontend_path, "index.html"))
    
    @app.get("/config.js")
    def read_config():
        return Response(content="window.BACKEND_URL = '/api';", media_type="application/javascript")
else:
    logger.warning("Frontend path not found. Running in API-only mode.")
