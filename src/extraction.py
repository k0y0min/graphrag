from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
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

    async def extract_async(self, chunks: List[FinalChunk], nodes: List[DocumentNode]) -> Tuple[List[Entity], List[Relation]]:
        """
        Async version of extract with parallel chunk processing.
        """
        entities: Dict[str, Entity] = {}
        relations: List[Relation] = []

        # 1. Structural Extraction (Fast, kept synchronous)
        for node in nodes:
            # Create entity for the node itself (structural entity)
            # Use node.id as entity id
            
            # Determine type
            node_type = node.metadata.get("type", "content").capitalize()
            # E.g. "Header", "Content"
            
            entity = Entity(
                id=node.id,
                type=f"Structure_{node_type}",
                description=node.text[:100], # Preview
                source_chunk_ids=[]
            )
            entities[node.id] = entity
            
            # Parent-Child Relation
            if node.parent_id and node.parent_id != "root":
                relations.append(Relation(
                    source_id=node.parent_id,
                    target_id=node.id,
                    type="PARENT_OF",
                    description="Structural hierarchy"
                ))

        # 2. Semantic Extraction (Parallel)
        import asyncio
        tasks = [self._extract_from_chunk_async(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        # Merge results
        for chunk, extracted in zip(chunks, results):
            # Merge extracted entities
            for ent_dict in extracted.get("entities", []):
                eid = ent_dict.get("id")
                etype = ent_dict.get("type")
                edescr = ent_dict.get("description", "")
                
                if not eid: continue
                
                if eid in entities:
                    # Merge? For now, just append description or ignored
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
            
            # Merge extracted relations
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
                    
        return list(entities.values()), relations

    async def _extract_from_chunk_async(self, chunk: FinalChunk) -> Dict[str, Any]:
        prompt = (
            f"Extract entities and relationships from the following text.\n"
            f"Context: {chunk.context}\n"
            f"Text: {chunk.text}\n\n"
            f"Return a JSON object with 'entities' (list of {{id, type, description}}) "
            f"and 'relations' (list of {{source, target, type, description}}).\n"
            f"IMPORTANT: The 'id' field should be the NAME of the entity (e.g. 'Apple', 'Elon Musk', 'iPhone 17'). "
            f"Do NOT use generic IDs like 'e1', 'e2'. Use the actual name so identical entities merge."
        )
        
        try:
            response = await self.llm.generate_async(prompt)
            if isinstance(response, str):
                # Try parsing if string
                # Clean markdown blocks
                clean = response.replace("```json", "").replace("```", "")
                return json.loads(clean)
            return response
        except Exception as e:
            # Fallback
            return {"entities": [], "relations": []}

    def extract(self, chunks: List[FinalChunk], nodes: List[DocumentNode]) -> Tuple[List[Entity], List[Relation]]:
        entities: Dict[str, Entity] = {}
        relations: List[Relation] = []

        # 1. Structural Extraction (from Tree)
        # Create Entity for each DocumentNode (Header/Content)
        # Create PARENT_OF edges
        
        for node in nodes:
            # Create entity for the node itself (structural entity)
            # Use node.id as entity id
            
            # Determine type
            node_type = node.metadata.get("type", "content").capitalize()
            # E.g. "Header", "Content"
            
            entity = Entity(
                id=node.id,
                type=f"Structure_{node_type}",
                description=node.text[:100], # Preview
                source_chunk_ids=[]
            )
            entities[node.id] = entity
            
            # Parent-Child Relation
            if node.parent_id and node.parent_id != "root":
                relations.append(Relation(
                    source_id=node.parent_id,
                    target_id=node.id,
                    type="PARENT_OF",
                    description="Structural hierarchy"
                ))

        # 2. Semantic Extraction (from Chunks)
        # Pass each chunk to LLM
        for chunk in chunks:
            extracted = self._extract_from_chunk(chunk)
            
            # Merge extracted entities
            for ent_dict in extracted.get("entities", []):
                eid = ent_dict.get("id")
                etype = ent_dict.get("type")
                edescr = ent_dict.get("description", "")
                
                if not eid: continue
                
                if eid in entities:
                    # Merge? For now, just append description or ignored
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
            
            # Merge extracted relations
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
                    
        return list(entities.values()), relations

    def _extract_from_chunk(self, chunk: FinalChunk) -> Dict[str, Any]:
        prompt = (
            f"Extract entities and relationships from the following text.\n"
            f"Context: {chunk.context}\n"
            f"Text: {chunk.text}\n\n"
            f"Return a JSON object with 'entities' (list of {{id, type, description}}) "
            f"and 'relations' (list of {{source, target, type, description}}).\n"
            f"IMPORTANT: The 'id' field should be the NAME of the entity (e.g. 'Apple', 'Elon Musk', 'iPhone 17'). "
            f"Do NOT use generic IDs like 'e1', 'e2'. Use the actual name so identical entities merge."
        )
        
        # Schema definition for more robust extraction using Gemini
        # (Simplified to plain dict return for this implementation)
        
        try:
            # We assume LLM returns dict or we parse it.
            # In Phase 1 we handled this.
            response = self.llm.generate(prompt)
            if isinstance(response, str):
                # Try parsing if string
                # Clean markdown blocks
                clean = response.replace("```json", "").replace("```", "")
                return json.loads(clean)
            return response
        except Exception as e:
            # Fallback
            return {"entities": [], "relations": []}
