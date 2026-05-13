import os
import shutil
import uuid
import hashlib
import math
import json
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.logger import PipelineLogger
from src.ingestion import LineMap, HierarchyDetector, DocumentTreeBuilder
from src.chunking import AncestryInjector, PerplexityChunker
from src.extraction import GraphExtractor, Entity, Relation
from src.community import detect_communities
from src.storage import GraphStorage
from src.llm_service import LLMService, VLLMBackend

class QueryEntities(BaseModel):
    """Schema for extracting entity names from a user query."""
    entities: List[str] = Field(description="List of entity names found in the query")

class EntityMatchSchema(BaseModel):
    is_same_entity: bool = Field(description="True if the new entity is the exact same real-world entity as the existing one, False otherwise")

class GraphRAGPipeline:
    def __init__(self, db_path="db/demo_db", log_enabled=True, clear_db=False):
        self.db_path = db_path
        self.logger = PipelineLogger(enabled=log_enabled)
        
        # Initialize Storage (Singleton)
        self.storage = GraphStorage(self.db_path, clear_existing=clear_db)
        if clear_db:
             self.logger.info("Database cleared on pipeline initialization.")
        

        # In pipeline.py __init__
        vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        model_name = os.getenv("VLLM_MODEL_NAME", "nvidia/Gemma-4-31B-IT-NVFP4")
        unified_vllm = VLLMBackend(model_name=model_name, base_url=vllm_url)


        use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
        
        if use_gemini:
            from src.llm_service import LiteLLMBackend
            gemini_model = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
            gemini_backend = LiteLLMBackend(model_name=gemini_model)
            
            self.llm_service = LLMService(
                chunking_model=unified_vllm, 
                extraction_model=gemini_backend
            )
            self.logger.info(f"Initialized with Gemini ({gemini_model}) for extraction/querying.")
        else:
            self.llm_service = LLMService(
                chunking_model=unified_vllm, 
                extraction_model=unified_vllm
            )
            self.logger.info("Initialized with vLLM for all tasks.")

        # Cleanup stale ingestion files on boot
        temp_dir = "temp_ingestion"
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                self.logger.info(f"Cleaned up stale temp directory: {temp_dir}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup stale temp directory: {e}")



    async def resolve_entities_async(self, entities: List[Entity]) -> Dict[str, str]:
        """
        Resolves extracted entities to existing nodes OR new unique nodes.
        Returns a mapping of {extraction_id: assigned_unique_id}.
        """
        id_map = {} # Extraction ID -> New/Existing UUID
        
        # 1. Deduplicate by name first (LLM usually merges these within a run)
        unique_extracted = {}
        for ent in entities:
            if ent.id not in unique_extracted:
                unique_extracted[ent.id] = ent
        
        # 2. Process each unique entity
        for name, ent in unique_extracted.items():
            # Get existing nodes with same name
            existing = self.storage.get_nodes_by_name(name)
            
            assigned_id = None
            if not existing:
                # New entity name entirely
                assigned_id = str(uuid.uuid4())
                self.logger.info(f"Resolution: '{name}' is new. Assigned UUID {assigned_id}")
            else:
                if len(ent.description.strip()) < 30:
                    assigned_id = existing[0]["id"]
                    self.logger.info(f"Resolution: '{name}' merged by heuristic (short desc).")
                    best_match = existing[0]
                else:
                    best_match = None
                    import difflib
                    
                    for node in existing:
                        # Pre-check name similarity to avoid false positives on generic names
                        name1_clean = name.lower().replace(" ", "")
                        name2_clean = node['name'].lower().replace(" ", "")
                        similarity = difflib.SequenceMatcher(None, name1_clean, name2_clean).ratio()
                        
                        if similarity < 0.8:
                            self.logger.info(f"Resolution: Skipping LLM check for '{name}' vs '{node['name']}' (similarity {similarity:.2f} < 0.8)")
                            continue
                            
                        prompt = f"Entity 1:\nName: {name}\nDescription: {ent.description}\n\nEntity 2:\nName: {node['name']}\nDescription: {node['description']}\n\nAre these exactly the same real-world entity? Consider their descriptions carefully."
                        try:
                            result = await self.llm_service.extractor.generate_async(prompt, schema=EntityMatchSchema, temperature=0.0)
                            if isinstance(result, dict):
                                is_same = result.get('is_same_entity', False)
                            elif isinstance(result, str):
                                parsed = json.loads(result.replace("```json", "").replace("```", "").strip())
                                is_same = parsed.get('is_same_entity', False)
                            else:
                                is_same = False
                        except Exception as e:
                            self.logger.error(f"LLM match failed: {e}")
                            is_same = False
                            
                        if is_same:
                            best_match = node
                            break
                    
                    if best_match:
                        self.logger.info(f"Resolution: Merging '{name}' with {best_match['id']} via LLM logic")
                        assigned_id = best_match["id"]
                    else:
                        assigned_id = str(uuid.uuid4())
                        self.logger.info(f"Resolution: Disambiguating '{name}' via LLM logic. New UUID {assigned_id}")
                
                # Option B: Merge descriptions if we found a match
                if best_match and assigned_id == best_match["id"] and best_match.get('description'):
                    merge_prompt = f"Combine these two descriptions of the same entity '{name}' into one coherent summary. Return ONLY the final combined summary. Do not include introductory text, explanations, or multiple options. Just the summary.\n\nDesc 1: {best_match['description']}\n\nDesc 2: {ent.description}"
                    try:
                        self.logger.info(f"Resolution: Merging descriptions for '{name}' via LLM.")
                        merged_desc = await self.llm_service.extractor.generate_async(merge_prompt, temperature=0.0)
                        if isinstance(merged_desc, str) and merged_desc.strip():
                            ent.description = merged_desc.strip()
                    except Exception as e:
                        self.logger.error(f"Failed to merge descriptions: {e}")
            
            id_map[name] = assigned_id
            ent.metadata["id"] = assigned_id
            ent.metadata["name"] = name

        return id_map


    def _report_progress(self, stage: str, current: int, total: int, status: str):
        """Standardized progress update helper for the frontend."""
        return {
            "type": "progress",
            "stage": stage,
            "current": current,
            "total": total,
            "status": status,
            "progress": (current / total * 100) if total > 0 else 100
        }

    async def ingest_async(self, text: str, input_filename=None, clear_db=False):
        doc_id = str(uuid.uuid4())[:8]

        temp_dir = "temp_ingestion"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        if not input_filename:
            input_filename = os.path.join(temp_dir, f"temp_{doc_id}.md")
        else:
            if not os.path.isabs(input_filename):
                input_filename = os.path.join(temp_dir, input_filename)
            
        with open(input_filename, "w") as f:
            f.write(text)
        
        try:
            # 1. Ingestion (Async Generator) - Becomes SILENT
            detector = HierarchyDetector(self.llm_service.extractor)

            line_map = LineMap(input_filename)

            roles = {}
            async for update in detector.detect_async(line_map, batch_size=20):
                if update["type"] == "progress":
                    continue # Silent hierarchy
                else:
                    roles = update["data"]
            
            tree_builder = DocumentTreeBuilder()
            nodes = tree_builder.build(line_map, roles, doc_id=doc_id)

            # 2. Chunking (Now Granular)
            injector = AncestryInjector()
            sentences = injector.inject(nodes)
            
            chunker = PerplexityChunker(
                self.llm_service.chunker, 
                max_tokens=int(os.getenv("CHUNKING_MAX_TOKENS", 100)),
                ppl_threshold=float(os.getenv("CHUNKING_PPL_THRESHOLD", 100.0))
            )

            chunks = []
            async for update in chunker.chunk_async(sentences):
                if isinstance(update, dict) and update.get("type") == "progress":
                    yield update
                else:
                    chunks = update

            # 3. Extraction (Async Generator)

            extractor = GraphExtractor(self.llm_service.extractor)
            entities, relations = [], []
            async for update in extractor.extract_async(chunks, nodes):
                if update["type"] == "progress":
                    yield update
                else:
                    entities = update["entities"]
                    relations = update["relations"]

            # 4. Entity Resolution & ID Mapping
            yield self._report_progress("Resolution", 0, 1, "Resolving entities and aliases...")
            id_map = await self.resolve_entities_async(entities)
            
            # Update entity IDs to the resolved unique IDs
            for e in entities:
                e.id = id_map.get(e.id, e.id)
            
            # Update relation IDs to point to the new unique IDs
            for r in relations:
                r.source_id = id_map.get(r.source_id, r.source_id)
                r.target_id = id_map.get(r.target_id, r.target_id)
            yield self._report_progress("Resolution", 1, 1, "Resolution complete")

            # 5. Storage
            yield self._report_progress("Storage", 0, 1, "Saving to knowledge graph...")
            if clear_db:
                self.storage.clear()
            self.storage.ingest(entities, relations)
            yield self._report_progress("Storage", 1, 1, "Storage complete")

            # 6. Global Community Detection (after storage so all data is persisted)
            yield self._report_progress("Communities", 0, 1, "Detecting global knowledge communities...")
            
            # Fetch full graph (now includes newly ingested data)
            full_graph = self.storage.get_full_graph()
            all_node_ids = set()
            all_edge_tuples = []
            
            # Entities from storage (includes current run)
            for n in full_graph["entities"]:
                all_node_ids.add(n["id"])
                
            # Relations from storage (includes current run)
            for r in full_graph["relations"]:
                all_edge_tuples.append((r["source_id"], r["target_id"]))
            
            # Run detection
            community_map = detect_communities(list(all_node_ids), all_edge_tuples)
            
            # Update storage (batch update for all nodes)
            self.storage.update_communities(community_map)
            
            yield self._report_progress("Communities", 1, 1, "Community detection complete")
            
            yield {
                "progress": 100,
                "status": "Ingestion complete!",
                "type": "result",
                "results": {
                    "nodes_processed": len(nodes),
                    "chunks_created": len(chunks),
                    "entities_extracted": len(entities),
                    "relations_extracted": len(relations),
                    "entities": [vars(e) for e in entities],
                    "relations": [vars(r) for r in relations]
                }
            }
        finally:
            if os.path.exists(input_filename):
                try:
                    os.remove(input_filename)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp file {input_filename}: {e}")
            
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")




    async def query(self, query: str, query_type="local"):
        # Use persistent storage
        storage = self.storage
        
        # 1. Extract potential entities from query using LLM
        prompt_extract = (
            f"Identify the key named entities in the following query that would match nodes in a knowledge graph.\n"
            f"Query: {query}\n"
            f"Return the entities found."
        )
        try:
            extracted_response = await self.llm_service.extractor.generate_async(prompt_extract, schema=QueryEntities)
            if isinstance(extracted_response, dict):
                extracted = extracted_response.get("entities", [query])
            elif isinstance(extracted_response, str):
                parsed = json.loads(extracted_response.replace("```json", "").replace("```", "").strip())
                extracted = parsed.get("entities", [query]) if isinstance(parsed, dict) else parsed
            else:
                extracted = [query]
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            extracted = [query] # Fallback to query itself

        self.logger.info(f"Extracted entities for query: {extracted}")

        # 2. Retrieve Context (Local Neighbors)
        all_nodes = {} # dedupe by ID
        all_edges = []
        seen_edges = set()
        
        context_str_parts = []

        for entity_id in extracted:
            # We assume exact match for now. In prod, use vector search.
            if query_type == "local":
                neighbors = storage.query_local(entity_id)
                
                # Fetch center node details
                center_node = storage.get_node(entity_id)
                if not center_node:
                    nodes_by_name = storage.get_nodes_by_name(entity_id)
                    if nodes_by_name:
                        center_node = nodes_by_name[0]
                
                if center_node:
                    all_nodes[center_node["id"]] = center_node
                    center_label = center_node["name"]
                else:
                    all_nodes[entity_id] = {"id": entity_id, "name": entity_id, "type": "QueryEntity"}
                    center_label = entity_id

                if neighbors:
                    for row in neighbors:
                        tgt_id, tgt_type, tgt_desc, rel_type, rel_desc, is_outgoing, tgt_name = row
                        
                        all_nodes[tgt_id] = {"id": tgt_id, "name": tgt_name, "type": tgt_type, "description": tgt_desc}
                        
                        if is_outgoing:
                            src, tgt = center_label, tgt_name
                            ctx_str = f"{center_label} --[{rel_type}: {rel_desc}]--> {tgt_name} ({tgt_desc})"
                        else:
                            src, tgt = tgt_name, center_label
                            ctx_str = f"{tgt_name} ({tgt_desc}) --[{rel_type}: {rel_desc}]--> {center_label}"
                            
                        edge_norm_key = f"{src.lower()}-{rel_type}-{tgt.lower()}"
                        
                        if edge_norm_key not in seen_edges:
                            seen_edges.add(edge_norm_key)
                            
                            all_edges.append({
                                "source_id": center_node["id"] if is_outgoing else tgt_id,
                                "target_id": tgt_id if is_outgoing else center_node["id"],
                                "source_name": center_label if is_outgoing else tgt_name,
                                "target_name": tgt_name if is_outgoing else center_label,
                                "type": rel_type,
                                "description": rel_desc
                            })
                            
                            if ctx_str not in context_str_parts:
                                context_str_parts.append(ctx_str)

        # 3. Generate Answer
        context_text = "\n".join(context_str_parts)
        
        if not context_text:
            answer = "I couldn't find any relevant information in the knowledge graph."
        else:
            prompt_answer = (
                f"You are a helpful assistant. Use the following Knowledge Graph context to answer the user's question.\n"
                f"If the answer is not in the context, say 'I assume X but it's not in the graph'.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            answer = await self.llm_service.extractor.generate_async(prompt_answer)
            if isinstance(answer, dict): 
                answer = str(answer)

        return {
            "answer": answer,
            "entities": [v for k,v in all_nodes.items()],
            "relations": all_edges
        }
