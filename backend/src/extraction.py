from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import asyncio
from src.llm_service import LLMBackend
from src.chunking import FinalChunk
from src.ingestion import DocumentNode

@dataclass
class Entity:
    id: str
    type: str
    description: str = ""
    source_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

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

        # 1. Semantic Extraction (Parallel)
        total = len(chunks)
        completed = 0


        # Start all tasks
        futures = [asyncio.ensure_future(self._extract_from_chunk_async(chunk)) for chunk in chunks]
        
        # Monitor completion
        for future in asyncio.as_completed(futures):
            await future
            completed += 1
            yield {
                "type": "progress",
                "stage": "Extraction",
                "current": completed,
                "total": total,
                "status": f"Extracting entities: {completed}/{total} chunks"
            }

        # Gather results (already finished)
        results = await asyncio.gather(*futures)

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
            f"Extract entities and relationships from the following text.\n"
            f"Context: {chunk.context}\n"
            f"Text: {chunk.text}\n\n"
            f"Rules:\n"
            f"1. For any self-references (I, me, my, mine, Narrator, Author, 'the user'), use the canonical entity ID 'User' unless the speaker's actual name is explicitly provided and clear.\n"
            f"2. Identical real-world entities MUST have the same ID (e.g., 'Apple' for the company).\n"
            f"3. Return a JSON object with 'entities' (list of {{id, type, description}}) "
            f"and 'relations' (list of {{source, target, type, description}}).\n"
            f"IMPORTANT: The 'id' field should be the NAME of the entity (e.g. 'Apple', 'Elon Musk', 'User'). "
        )
        
        try:
            res1 = await self.llm.generate_async(prompt1)
            parsed1 = self._parse_llm_json(res1)
            
            # Pass 2: The "Glance Back" Refinement
            # We provide the LLM with what it already found and ask what it missed.
            prompt2 = (
                f"Review the following text and the entities/relationships already extracted from it.\n"
                f"Text: {chunk.text}\n\n"
                f"Already Extracted: {json.dumps(parsed1, indent=2)}\n\n"
                f"QUESTION: Are there any specialized relationships, secondary entities, or deep connections in the text that were missed or could be refined?\n"
                f"Return a JSON object with any EXTRA 'entities' and 'relations' found. If nothing missed, return empty lists."
            )
            
            res2 = await self.llm.generate_async(prompt2)
            parsed2 = self._parse_llm_json(res2)
            
            # Combine
            combined = {
                "entities": parsed1.get("entities", []) + parsed2.get("entities", []),
                "relations": parsed1.get("relations", []) + parsed2.get("relations", [])
            }
            return combined
            
        except Exception as e:
            return {"entities": [], "relations": []}

    def _parse_llm_json(self, response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            try:
                clean = response.replace("```json", "").replace("```", "").strip()
                return json.loads(clean)
            except:
                pass
        return {"entities": [], "relations": []}


