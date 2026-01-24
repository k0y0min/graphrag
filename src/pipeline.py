import os
import shutil
from src.llm_service import LLMService, GeminiBackend, OllamaBackend, LocalHuggingFaceBackend
from src.logger import PipelineLogger
from src.ingestion import LineMap, HierarchyDetector, DocumentTreeBuilder
from src.chunking import AncestryInjector, PerplexityChunker
from src.extraction import GraphExtractor
from src.community import detect_communities
from src.storage import GraphStorage

class GraphRAGPipeline:
    def __init__(self, db_path="demo_db", log_enabled=True, clear_db=False):
        self.db_path = db_path
        self.logger = PipelineLogger(enabled=log_enabled)
        
        # Initialize Storage (Singleton)
        self.storage = GraphStorage(self.db_path, clear_existing=clear_db)
        if clear_db:
             self.logger.info("Database cleared on pipeline initialization.")
        
        # Initialize Backend
        # Extraction Model (API - High Intelligence)
        extraction_backend = GeminiBackend()
        
        # Chunking Model (Local HF - Perplexity Based)
        # Using Qwen/Qwen2.5-0.5B-Instruct or similar small model for PPL
        chunking_backend = LocalHuggingFaceBackend(
            model_id="Qwen/Qwen3-0.6B",
            local_dir="./models"
        )
        
        self.llm_service = LLMService(
            chunking_model=chunking_backend,
            extraction_model=extraction_backend
        )

    async def ingest_async(self, text: str, input_filename=None, clear_db=False):
        import uuid
        import asyncio
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
            # 1. Ingestion (Async)
            line_map = LineMap(input_filename)
            detector = HierarchyDetector(self.llm_service.extractor)
            
            # Async detection
            roles = await detector.detect_async(line_map, batch_size=20)
            
            tree_builder = DocumentTreeBuilder()
            nodes = tree_builder.build(line_map, roles, doc_id=doc_id)

            # 2. Chunking (Still Sync for now as it's sequential/CPU bound per document)
            # We could offload to thread if needed, but it's local model inference which releases GIL.
            # Ideally we run this in a thread executor to not block loop during heavy PPL calc.
            
            injector = AncestryInjector()
            sentences = injector.inject(nodes)
            
            chunker = PerplexityChunker(self.llm_service.chunker, max_tokens=100, logger=self.logger)
            
            # Offload chunking to thread to keep loop responsive
            chunks = await asyncio.to_thread(chunker.chunk, sentences)

            # 3. Extraction (Async)
            extractor = GraphExtractor(self.llm_service.extractor)
            entities, relations = await extractor.extract_async(chunks, nodes)

            # 4. Community Detection (Sync - fast enough or offload)
            community_map = detect_communities(entities, relations)
            for e in entities:
                if e.id in community_map:
                    e.metadata["community_id"] = community_map[e.id]

            # 5. Storage (Sync - likely IO bound but fast local DB)
            # Use persistent storage instance
            if clear_db:
                self.storage.clear()
            self.storage.ingest(entities, relations)
            
            return {
                "nodes_processed": len(nodes),
                "chunks_created": len(chunks),
                "entities_extracted": len(entities),
                "relations_extracted": len(relations),
                "entities": [vars(e) for e in entities],
                "relations": [vars(r) for r in relations]
            }
        finally:
            if os.path.exists(input_filename):
                try:
                    os.remove(input_filename)
                    self.logger.info(f"Cleaned up temporary file: {input_filename}")
                except Exception as e:
                    self.logger.error(f"Failed to cleanup {input_filename}: {e}")
            
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except: pass

    def ingest_text(self, text: str, input_filename=None, clear_db=False):
        import uuid
        doc_id = str(uuid.uuid4())[:8]
        
        temp_dir = "temp_ingestion"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        if not input_filename:
            input_filename = os.path.join(temp_dir, f"temp_{doc_id}.md")
        else:
            # If a filename is provided, ensure it's in the temp dir if it's relative
            if not os.path.isabs(input_filename):
                input_filename = os.path.join(temp_dir, input_filename)
            
        # Save text to temporary file for LineMap
        with open(input_filename, "w") as f:
            f.write(text)
        
        try:
            # 1. Ingestion
            line_map = LineMap(input_filename)
            detector = HierarchyDetector(self.llm_service.extractor)
            roles = detector.detect(line_map, batch_size=20)
            
            tree_builder = DocumentTreeBuilder()
            nodes = tree_builder.build(line_map, roles, doc_id=doc_id)

            # 2. Chunking
            injector = AncestryInjector()
            sentences = injector.inject(nodes)
            
            chunker = PerplexityChunker(self.llm_service.chunker, max_tokens=100, logger=self.logger)
            chunks = chunker.chunk(sentences)

            # 3. Extraction
            extractor = GraphExtractor(self.llm_service.extractor)
            entities, relations = extractor.extract(chunks, nodes)

            # 4. Community Detection
            community_map = detect_communities(entities, relations)
            for e in entities:
                if e.id in community_map:
                    e.metadata["community_id"] = community_map[e.id]

            # 5. Storage
            # Use persistent storage instance
            if clear_db:
                self.storage.clear()
            self.storage.ingest(entities, relations)
            
            return {
                "nodes_processed": len(nodes),
                "chunks_created": len(chunks),
                "entities_extracted": len(entities),
                "relations_extracted": len(relations),
                "entities": [vars(e) for e in entities],
                "relations": [vars(r) for r in relations]
            }
        finally:
            # Cleanup: Delete the temp file
            if os.path.exists(input_filename):
                try:
                    os.remove(input_filename)
                    self.logger.info(f"Cleaned up temporary file: {input_filename}")
                except Exception as e:
                    self.logger.error(f"Failed to cleanup {input_filename}: {e}")
            
            # Optional: Clean up the directory if empty
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except: pass

    def query(self, query: str, query_type="local"):
        # Use persistent storage
        storage = self.storage
        
        # 1. Extract potential entities from query using LLM
        prompt_extract = (
            f"Identify the key named entities in the following query that would match nodes in a knowledge graph.\n"
            f"Query: {query}\n"
            f"Return ONLY a JSON list of strings, e.g. [\"Elon Musk\", \"SpaceX\"]."
        )
        try:
            import json
            extracted = self.llm_service.extractor.generate(prompt_extract, schema=list)
            if isinstance(extracted, str):
                # Fallback parse if backend didn't handle schema
                extracted = json.loads(extracted.replace("```json", "").replace("```", "").strip())
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
            # Local query returns list of neighbor tuples/dicts
            if query_type == "local":
                neighbors = storage.query_local(entity_id)
                # Format: [[target_id, target_type, target_desc, rel_type, rel_desc, is_outgoing]]
                
                # Fetch center node details to avoid "QueryEntity" artifact
                center_node = storage.get_node(entity_id)
                if center_node:
                    all_nodes[center_node["id"]] = center_node
                else:
                    # Fallback if not found (shouldn't happen if neighbors exist, unless phantom edge)
                    all_nodes[entity_id] = {"id": entity_id, "type": "QueryEntity"}

                if neighbors:
                    for row in neighbors:
                        # row = [b.id, b.type, b.description, r.rel_type, r.description, is_outgoing]
                        tgt_id, tgt_type, tgt_desc, rel_type, rel_desc, is_outgoing = row
                        
                        all_nodes[tgt_id] = {"id": tgt_id, "type": tgt_type, "description": tgt_desc}
                        
                        # Edge direction for graph vis
                        if is_outgoing:
                            src, tgt = entity_id, tgt_id
                            arrow = "-->"
                            ctx_str = f"{entity_id} --[{rel_type}: {rel_desc}]--> {tgt_id} ({tgt_desc})"
                        else:
                            src, tgt = tgt_id, entity_id
                            arrow = "-->" # Standard vis direction
                            ctx_str = f"{tgt_id} ({tgt_desc}) --[{rel_type}: {rel_desc}]--> {entity_id}"
                            
                        # Robust Deduplication
                        # Normalize keys to avoid "Gukesh" vs "gukesh" duplicates
                        edge_norm_key = f"{src.lower()}-{rel_type}-{tgt.lower()}"
                        
                        # We also want to avoid showing exact same info multiple times even if IDs match
                        # Track seen normalized edges for this query
                        if edge_norm_key not in seen_edges:
                            seen_edges.add(edge_norm_key)
                            
                            all_edges.append({
                                "source_id": src,
                                "target_id": tgt,
                                "type": rel_type,
                                "description": rel_desc
                            })
                            
                            # Add to context (check for string dup just in case desc differs slightly, 
                            # but usually we want one edge per relation type)
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
            answer = self.llm_service.extractor.generate(prompt_answer)
            if isinstance(answer, dict): 
                answer = str(answer) # handle edge case if schema lingers

        return {
            "answer": answer,
            "entities": [v for k,v in all_nodes.items()],
            "relations": all_edges
        }
