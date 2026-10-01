import os
import shutil
import uuid
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.logger import PipelineLogger
from src.chunking import ContextualChunker, FinalChunk
from src.extraction import GraphExtractor, Entity, Relation
from src.community import detect_communities
from src.storage import GraphStorage
from src.llm_service import LLMService, LiteLLMBackend
from src.langgraph_rag import LangGraphRAGEngine

class QueryEntities(BaseModel):
    """Schema for extracting entity names from a user query."""
    entities: List[str] = Field(description="List of entity names found in the query")

class GraphRAGPipeline:
    def __init__(self, db_path="db/demo_db", log_enabled=True, clear_db=False):
        self.db_path = db_path
        self.logger = PipelineLogger(enabled=log_enabled)
        
        # Initialize LadybugDB Storage (Singleton)
        self.storage = GraphStorage(self.db_path, clear_existing=clear_db)
        if clear_db:
             self.logger.info("Database cleared on pipeline initialization.")

        # Default model: Gemini 3.8 Flash via Vertex AI / ADC
        gemini_model = os.getenv("GEMINI_MODEL", "vertex_ai/gemini-3.8-flash")
        gemini_backend = LiteLLMBackend(model_name=gemini_model)

        self.llm_service = LLMService(
            chunking_model=gemini_backend, 
            extraction_model=gemini_backend
        )
        self.logger.info(f"Initialized monolithic pipeline with Gemini 3.8 Flash ({gemini_model}) via ADC.")

        # Initialize LangGraph Adaptive Multi-Agent Hybrid RAG engine
        self.langgraph_engine = LangGraphRAGEngine(self.storage, self.llm_service.extractor)
        self.latest_chunks: List[Dict[str, Any]] = []
        self.latest_alias_records: List[Dict[str, Any]] = []

    async def resolve_entities_async(self, entities: List[Entity], relations: Optional[List[Relation]] = None) -> tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Resolves extracted entities to existing nodes OR new unique nodes.
        Uses exact name grouping combined with an LLM discriminator for homonyms.
        Guarantees relation endpoints map to valid entity IDs.
        Returns a tuple: (id_map, alias_records).
        """
        id_map = {}
        alias_records = []
        unique_extracted = {}
        for ent in entities:
            norm_name = ent.id.strip().upper()
            ent_aliases = getattr(ent, "aliases", []) or []
            if norm_name not in unique_extracted:
                unique_extracted[norm_name] = {
                    "original_ids": [ent.id], 
                    "entity": ent, 
                    "descriptions": [ent.description],
                    "aliases": list(ent_aliases)
                }
            else:
                if ent.id not in unique_extracted[norm_name]["original_ids"]:
                    unique_extracted[norm_name]["original_ids"].append(ent.id)
                unique_extracted[norm_name]["descriptions"].append(ent.description)
                for a in ent_aliases:
                    if a and a not in unique_extracted[norm_name]["aliases"]:
                        unique_extracted[norm_name]["aliases"].append(a)

        for norm_name, data in unique_extracted.items():
            ent = data["entity"]
            original_ids = data["original_ids"]
            combined_desc = " ".join([d for d in data["descriptions"] if d.strip()])
            all_aliases = list(dict.fromkeys(original_ids + data.get("aliases", [])))

            existing = self.storage.get_nodes_by_name(norm_name)
            db_candidates = [node for node in existing if node.get('name', '').strip().upper() == norm_name]

            resolution_type = "new"
            if not db_candidates:
                assigned_id = str(uuid.uuid4())
                self.logger.info(f"Resolution: '{norm_name}' is entirely new. Assigned UUID {assigned_id}")
                final_desc = combined_desc
                display_name = original_ids[0]
                resolution_type = "new"
            else:
                db_desc_text = ""
                for i, db_node in enumerate(db_candidates):
                    db_desc_text += f"{i}: \"{db_node.get('description', '')}\"\n"

                prompt = (
                    f"You are an AI performing Entity Disambiguation.\n"
                    f"We extracted a new entity named '{norm_name}'.\n"
                    f"Description of extracted entity:\n\"{combined_desc}\"\n\n"
                    f"We found existing entities in the database with the exact same name. Here are their descriptions:\n"
                    f"{db_desc_text}\n"
                    f"Does the newly extracted entity refer to the EXACT SAME real-world concept as any of the database entities?\n"
                    f"If yes, return the integer index (e.g. 0 or 1) of the matching database entity.\n"
                    f"If no (it is a homonym or completely different concept), return -1.\n"
                    f"Return ONLY the integer. No other text."
                )

                try:
                    response = await self.llm_service.extractor.generate_async(prompt)
                    if isinstance(response, str):
                        clean_resp = "".join([c for c in response.strip() if c.isdigit() or c == '-'])
                        match_idx = int(clean_resp) if clean_resp else -1
                    else:
                        match_idx = int(response)

                    if 0 <= match_idx < len(db_candidates):
                        best_match = db_candidates[match_idx]
                        assigned_id = best_match["id"]
                        self.logger.info(f"Resolution: '{norm_name}' matched DB entity {assigned_id}.")

                        merge_prompt = f"Combine these two descriptions of the entity '{norm_name}' into one coherent summary. Return ONLY the final combined summary.\n\nDescription 1: {best_match.get('description', '')}\nDescription 2: {combined_desc}"
                        merged_desc = await self.llm_service.extractor.generate_async(merge_prompt)
                        final_desc = str(merged_desc).strip()
                        display_name = best_match.get("name", original_ids[0])
                        resolution_type = "merged"
                    else:
                        assigned_id = str(uuid.uuid4())
                        self.logger.info(f"Resolution: '{norm_name}' is a new homonym. Assigned UUID {assigned_id}")
                        final_desc = combined_desc
                        display_name = original_ids[0]
                        resolution_type = "homonym"
                except Exception as e:
                    self.logger.error(f"Discriminator failed for '{norm_name}': {e}. Safely creating new entity.")
                    assigned_id = str(uuid.uuid4())
                    final_desc = combined_desc
                    display_name = original_ids[0]
                    resolution_type = "new"

            ent.description = final_desc
            ent.metadata["id"] = assigned_id
            ent.metadata["name"] = display_name

            for a in all_aliases:
                clean_a = a.strip()
                id_map[clean_a] = assigned_id
                id_map[clean_a.upper()] = assigned_id
                id_map[clean_a.lower()] = assigned_id

            alias_records.append({
                "canonical_name": display_name,
                "assigned_id": assigned_id,
                "entity_type": ent.type,
                "aliases": all_aliases,
                "resolution_type": resolution_type,
                "description": final_desc
            })

        # Ensure all relation endpoints exist in id_map and entities
        if relations:
            for r in relations:
                for ep in [r.source_id, r.target_id]:
                    clean_ep = ep.strip()
                    if clean_ep not in id_map and clean_ep.upper() not in id_map:
                        new_id = str(uuid.uuid4())
                        entities.append(Entity(
                            id=new_id,
                            type="CONCEPT",
                            description=f"Entity '{clean_ep}' referenced in knowledge graph relations.",
                            metadata={"id": new_id, "name": clean_ep}
                        ))
                        id_map[clean_ep] = new_id
                        id_map[clean_ep.upper()] = new_id
                        id_map[clean_ep.lower()] = new_id

        self.latest_alias_records = alias_records
        return id_map, alias_records

    def _report_progress(self, stage: str, current: int, total: int, status: str, extra: Optional[Dict[str, Any]] = None):
        """Standardized progress update helper for the frontend."""
        data = {
            "type": "progress",
            "stage": stage,
            "current": current,
            "total": total,
            "status": status,
            "progress": (current / total * 100) if total > 0 else 100
        }
        if extra:
            data.update(extra)
        return data

    async def ingest_async(self, text: str, input_filename=None, clear_db=False, max_chunk_tokens: Optional[int] = None):
        """
        Async streaming ingestion pipeline leveraging Macro-Context Horizon extraction.
        Partitions chunks into extraction targets with preceding/forward horizons and builds LadybugDB graph with communities.
        """
        try:
            # 1. Macro-Context Horizon Chunking
            chunk_kwargs = {}
            if max_chunk_tokens and max_chunk_tokens >= 200:
                chunk_kwargs["max_chunk_tokens"] = int(max_chunk_tokens)
                chunk_kwargs["target_chunk_tokens"] = int(max_chunk_tokens * 0.8)
            chunker = ContextualChunker(self.llm_service.chunker, **chunk_kwargs)
            chunks: List[FinalChunk] = []

            async for update in chunker.chunk_async(text):
                if isinstance(update, dict) and update.get("type") == "progress":
                    yield update
                else:
                    chunks = update

            formatted_chunks = [
                {
                    "id": c.id,
                    "text": c.text,
                    "context": c.context,
                    "start_char_idx": c.start_char_idx,
                    "end_char_idx": c.end_char_idx
                }
                for c in chunks
            ]
            self.latest_chunks = formatted_chunks
            yield self._report_progress("Horizon Chunking", len(chunks), len(chunks), f"Partitioned {len(chunks)} target chunks with 35k preceding & 1.5k forward macro-context", {"chunks_preview": formatted_chunks[:3]})

            # 2. Graph Extraction (Pydantic Schema Enforced)
            extractor = GraphExtractor(self.llm_service.extractor)
            entities, relations = [], []

            async for update in extractor.extract_async(chunks, []):
                if update["type"] == "progress":
                    yield update
                else:
                    entities = update["entities"]
                    relations = update["relations"]

            # 3. Entity Resolution & Disambiguation
            yield self._report_progress("Resolution", 0, 1, "Resolving entities and aliases...")
            id_map, alias_records = await self.resolve_entities_async(entities, relations)

            for e in entities:
                e.id = id_map.get(e.id, id_map.get(e.id.strip().upper(), e.id))
            for r in relations:
                r.source_id = id_map.get(r.source_id, id_map.get(r.source_id.strip().upper(), r.source_id))
                r.target_id = id_map.get(r.target_id, id_map.get(r.target_id.strip().upper(), r.target_id))
            yield self._report_progress("Resolution", 1, 1, f"Resolved {len(alias_records)} entity concepts & aliases", {"alias_records": alias_records})

            # 4. Storage in LadybugDB
            yield self._report_progress("Storage", 0, 1, "Saving to LadybugDB graph...")
            if clear_db:
                self.storage.clear()
            self.storage.ingest(entities, relations)
            self.storage.ingest_chunks(chunks)
            yield self._report_progress("Storage", 1, 1, "LadybugDB graph & chunk storage complete")

            # 5. Global Community Detection (NetworkX Modularity)
            yield self._report_progress("Communities", 0, 1, "Detecting global knowledge communities...")
            full_graph = self.storage.get_full_graph()
            all_node_ids = {n["id"] for n in full_graph["entities"]}
            all_edge_tuples = [(r["source_id"], r["target_id"]) for r in full_graph["relations"]]

            community_map = detect_communities(list(all_node_ids), all_edge_tuples)
            self.storage.update_communities(community_map)
            yield self._report_progress("Communities", 1, 1, f"Identified {len(set(community_map.values()))} knowledge communities")

            yield {
                "progress": 100,
                "status": "Ingestion complete!",
                "type": "result",
                "results": {
                    "nodes_processed": len(chunks),
                    "chunks_created": len(chunks),
                    "entities_extracted": len(entities),
                    "relations_extracted": len(relations),
                    "entities": [vars(e) for e in entities],
                    "relations": [vars(r) for r in relations],
                    "chunks": formatted_chunks,
                    "alias_resolutions": alias_records,
                    "id_map": id_map,
                    "communities": community_map
                }
            }
        except Exception as e:
            self.logger.error(f"Ingestion failed: {e}")
            yield {"type": "error", "detail": str(e)}

    async def query(self, query: str, query_type: str = "local"):
        """
        Executes query through the LangGraph Adaptive Multi-Agent Hybrid RAG engine.
        """
        return await self.langgraph_engine.run(query, query_type=query_type)
