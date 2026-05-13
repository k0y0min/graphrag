from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import asyncio
import logging
from src.llm_service import LLMBackend
from src.chunking import FinalChunk

logger = logging.getLogger(__name__)
from src.ingestion import DocumentNode
from pydantic import BaseModel, Field

# Define the exact structure we want the LLM to output
class ExtractedEntity(BaseModel):
    id: str = Field(description="The canonical NAME of the entity (e.g. 'Apple', 'Elon Musk', 'User')")
    type: str = Field(description="The category of the entity (e.g., 'Person', 'Organization')")
    description: str = Field(description="Brief summary of what this entity is in this context")

class ExtractedRelation(BaseModel):
    source: str = Field(description="The ID of the source entity")
    target: str = Field(description="The ID of the target entity")
    type: str = Field(description="The type of relationship (e.g., 'FOUNDED', 'WORKS_FOR')")
    description: str = Field(description="Brief explanation of their connection")

class GraphExtractionSchema(BaseModel):
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]

@dataclass
class Entity:
    id: str
    type: str
    description: str = ""
    source_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)

@dataclass
class Relation:
    source_id: str
    target_id: str
    type: str
    description: str = ""
    source_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class GraphExtractor:
    def __init__(self, llm: LLMBackend):
        self.llm = llm

    async def extract_async(self, chunks: List[FinalChunk], nodes: List[DocumentNode]):
        """
        Async generator version of extract with parallel chunk processing.
        """
        entities: Dict[str, Entity] = {}
        relations: List[Relation] = []

        # 1. Semantic Extraction (Parallel with Semaphore)
        total = len(chunks)
        completed = 0
        results = [None] * total
        
        # Limit concurrent vLLM requests to avoid overwhelming the server
        semaphore = asyncio.Semaphore(5)

        async def _run_extraction(idx, chunk):
            async with semaphore:
                results[idx] = await self._extract_from_chunk_async(chunk)
            return idx

        # Start all tasks
        tasks = [asyncio.ensure_future(_run_extraction(i, c)) for i, c in enumerate(chunks)]
        
        # Monitor completion
        for future in asyncio.as_completed(tasks):
            await future
            completed += 1
            yield {
                "type": "progress",
                "stage": "Extraction",
                "current": completed,
                "total": total,
                "status": f"Extracting entities: {completed}/{total} chunks"
            }

        # Merge results
        for chunk, extracted in zip(chunks, results):
            for ent_dict in extracted.get("entities", []):
                eid = ent_dict.get("id")
                etype = ent_dict.get("type")
                edescr = ent_dict.get("description", "")
                if not eid: continue
                
                if eid in entities:
                    existing = entities[eid]
                    if len(existing.description) < len(edescr):
                         existing.description = edescr
                    if chunk.id not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(chunk.id)
                else:
                    entities[eid] = Entity(
                        id=eid,
                        type=etype,
                        description=edescr,
                        source_chunk_ids=[chunk.id]
                    )
            
            for rel_dict in extracted.get("relations", []):
                src = rel_dict.get("source")
                tgt = rel_dict.get("target")
                rtype = rel_dict.get("type")
                rdescr = rel_dict.get("description", "")
                if src and tgt:
                    relations.append(Relation(
                        source_id=src,
                        target_id=tgt,
                        type=rtype,
                        description=rdescr,
                        source_chunk_ids=[chunk.id]
                    ))
                    
        yield {"type": "result", "entities": list(entities.values()), "relations": relations}



    async def _extract_from_chunk_async(self, chunk: FinalChunk) -> Dict[str, Any]:
        # Pass 1: Initial Extraction
        prompt1 = (
            f"Context: {chunk.context}\n"
            f"Text: {chunk.text}\n\n"
            f"Rules:\n"
            f"1. For any self-references (I, me, my), use the canonical entity ID 'User' unless the speaker's actual name is explicitly provided.\n"
            f"2. Identical real-world entities MUST have the exact same ID.\n"
            f"Extract all entities and relationships."
        )
        
        try:
            # We pass our Pydantic schema to your LLM backend
            # Note: Your vLLM backend wrapper will need to pass this into `guided_json` or `response_format`
            res1_text = await self.llm.generate_async(prompt1, schema=GraphExtractionSchema)
            
            # NO MORE STRING STRIPPING. We know this is valid JSON.
            parsed1 = json.loads(res1_text) if isinstance(res1_text, str) else res1_text
            
            # Pass 2: The "Glance Back" Refinement
            prompt2 = (
                f"Text: {chunk.text}\n\n"
                f"Already Extracted: {json.dumps(parsed1, indent=2)}\n\n"
                f"Are there any specialized relationships, secondary entities, or deep connections in the text that were missed?\n"
                f"Extract any EXTRA entities and relations. If nothing was missed, return empty lists."
            )
            
            res2_text = await self.llm.generate_async(prompt2, schema=GraphExtractionSchema)
            parsed2 = json.loads(res2_text) if isinstance(res2_text, str) else res2_text
            
            # Combine results
            combined = {
                "entities": parsed1.get("entities", []) + parsed2.get("entities", []),
                "relations": parsed1.get("relations", []) + parsed2.get("relations", [])
            }
            return combined
            
        except Exception as e:
            # The only exceptions caught here now will be actual network timeouts, 
            # not JSON parsing errors.
            print(f"Extraction failed for chunk {chunk.id}: {e}")
            return {"entities": [], "relations": []}

    def _parse_llm_json(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            try:
                clean = response.replace("```json", "").replace("```", "").strip()
                return json.loads(clean)
            except Exception as e:
                logger.error(f"Failed to parse LLM JSON: {e}")
        return {"entities": [], "relations": []}


