
import pytest
import shutil
import os
from dotenv import load_dotenv
from src.pipeline import GraphRAGPipeline

load_dotenv()

@pytest.mark.asyncio
async def test_pipeline_ingest_async(tmp_path):
    # Setup
    db_path = tmp_path / "test_async_db"
    pipeline = GraphRAGPipeline(db_path=str(db_path), log_enabled=False)
    
    text = "# Header\nSome content about Apple."
    
    # Run Async Ingest
    stats = await pipeline.ingest_async(text, "test_doc.md", clear_db=True)
    
    # Verify Stats
    assert stats["nodes_processed"] > 0
    assert stats["chunks_created"] > 0
    
    # Verify DB content (basic check)
    # We can query it or check if files exist
    assert os.path.exists(str(db_path))
    
    # Cleanup done by tmp_path fixture usually, but pipeline might produce residuals
