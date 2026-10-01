import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.llm_service import LLMBackend
from src.storage import GraphStorage

logger = logging.getLogger(__name__)

class QueryEntities(BaseModel):
    """Schema for extracting entity names from a user query and determining the optimal retrieval strategy."""
    entities: List[str] = Field(description="List of key named entity concepts found in the query")
    route: str = Field(
        default="hybrid", 
        description=(
            "Retrieval route: "
            "'vector' for direct textual quotes, passage excerpts, specific textual descriptions, summaries of text sections, or when asked to summarize text; "
            "'graph' for relationship questions, multi-hop traversals, connections between entities, or structural hierarchy; "
            "'hybrid' for complex inquiries needing both entity relationships and textual nuances."
        )
    )

class RouteDecision(BaseModel):
    route: str = Field(description="'graph', 'vector', or 'hybrid'")
    reasoning: str = Field(description="Brief explanation of the chosen retrieval route")

class GradeEvaluation(BaseModel):
    is_grounded: bool = Field(description="True if context has enough info to answer, False otherwise")
    feedback: str = Field(description="Brief assessment of the retrieved context")

class GraphRAGState(TypedDict):
    query: str
    query_type: str
    route: str
    extracted_entities: List[str]
    graph_context: str
    vector_context: str
    combined_context: str
    graph_nodes: List[Dict[str, Any]]
    graph_edges: List[Dict[str, Any]]
    answer: str
    retry_count: int
    is_grounded: bool

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


class LangGraphRAGEngine:
    """
    Adaptive Multi-Agent Hybrid RAG engine orchestrated via LangGraph.
    Dynamically routes queries between LadybugDB Cypher graph traversals
    and text retrieval, synthesizing grounded answers with self-correction.
    """
    def __init__(self, storage: GraphStorage, llm: LLMBackend):
        self.storage = storage
        self.llm = llm
        self.graph_app = self._build_graph() if HAS_LANGGRAPH else None

    def _build_graph(self):
        workflow = StateGraph(GraphRAGState)

        # Register nodes for pure deterministic Hybrid RAG
        workflow.add_node("concept_extractor", self._router_node)
        workflow.add_node("graph_retrieval", self._graph_retrieval_node)
        workflow.add_node("vector_retrieval", self._vector_retrieval_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("grader", self._grader_node)

        # Pure Hybrid Retrieval Pipeline:
        # Every query extracts concepts -> traverses Cypher graph in LadybugDB -> retrieves contextual text chunks -> synthesizes -> grades
        workflow.set_entry_point("concept_extractor")
        workflow.add_edge("concept_extractor", "graph_retrieval")
        workflow.add_edge("graph_retrieval", "vector_retrieval")
        workflow.add_edge("vector_retrieval", "synthesizer")
        workflow.add_edge("synthesizer", "grader")

        # Grader evaluation: groundness verification and 2-hop self-correction retry
        def grade_decision(state: GraphRAGState):
            if state.get("is_grounded", True) or state.get("retry_count", 0) >= 1:
                return END
            return "graph_retrieval"

        workflow.add_conditional_edges("grader", grade_decision, {
            END: END,
            "graph_retrieval": "graph_retrieval"
        })

        return workflow.compile()

    async def _router_node(self, state: GraphRAGState) -> Dict[str, Any]:
        query = state["query"]
        extracted_entities = []
        route = "hybrid"

        query_lower = query.lower().strip()
        global_keywords = [
            "summarize", "summary", "theme", "themes", "overview", "what is this", 
            "tell me about everything", "knowledge base", "all entities", "outline", 
            "main topics", "key points", "what do you know", "what's in the graph", 
            "explain the graph", "what is in the graph", "show me everything"
        ]
        is_global = any(k in query_lower for k in global_keywords)

        prompt = (
            f"Analyze the following user query for knowledge graph and contextual document retrieval.\n"
            f"Query: {query}\n"
            f"Extract all key named entities, concepts, skills, technologies, people, or subjects mentioned."
        )
        try:
            res = await self.llm.generate_async(prompt, schema=QueryEntities)
            if isinstance(res, dict):
                extracted_entities = res.get("entities", [])
            elif hasattr(res, "entities"):
                extracted_entities = res.entities
        except Exception as e:
            logger.debug(f"Router node LLM extraction: {e}")

        # In addition, search LadybugDB for any existing entities mentioned in the query text or semantically relevant
        db_matched = self.storage.find_entities_relevant_to_query(query, limit=12)
        db_entity_names = [m["name"] for m in db_matched if m.get("name")]

        # Combine unique entities
        all_entities = []
        for e in extracted_entities + db_entity_names:
            clean = e.strip()
            if clean and clean not in all_entities and len(clean) >= 2:
                all_entities.append(clean)

        # Fallback to key capitalized words or tokens if nothing extracted
        if not all_entities:
            import re
            caps = re.findall(r'\b[A-Z][a-zA-Z0-9_\-\']+\b', query)
            stopwords = {"What", "Who", "Where", "When", "Why", "How", "Which", "Is", "Are", "Did", "Can", "Could", "The"}
            clean_caps = [c for c in caps if c not in stopwords]
            all_entities = clean_caps if clean_caps else [query]

        # Hybrid retrieval is always the default to eliminate brittle routing failures
        route = "global" if is_global else "hybrid"

        return {
            "extracted_entities": all_entities,
            "route": route
        }

    async def _graph_retrieval_node(self, state: GraphRAGState) -> Dict[str, Any]:
        entities = state.get("extracted_entities", [])
        route = state.get("route", "hybrid")
        all_nodes = {n["id"]: n for n in state.get("graph_nodes", [])}
        all_edges = state.get("graph_edges", []).copy()
        seen_edges = {f"{e['source_id']}-{e['type']}-{e['target_id']}" for e in all_edges}
        context_parts = []

        for entity_id in entities:
            # Query local connections in LadybugDB
            neighbors = self.storage.query_local(entity_id)
            center_node = self.storage.get_node(entity_id)
            if not center_node:
                matched = self.storage.get_nodes_by_name(entity_id)
                if matched:
                    center_node = matched[0]

            if center_node:
                all_nodes[center_node["id"]] = center_node
                center_label = center_node.get("name") or center_node["id"]
            else:
                all_nodes[entity_id] = {"id": entity_id, "name": entity_id, "type": "QueryEntity"}
                center_label = entity_id

            if neighbors:
                for row in neighbors:
                    tgt_id, tgt_type, tgt_desc, rel_type, rel_desc, is_outgoing, tgt_name = row
                    all_nodes[tgt_id] = {"id": tgt_id, "name": tgt_name, "type": tgt_type, "description": tgt_desc}

                    src = center_label if is_outgoing else tgt_name
                    tgt = tgt_name if is_outgoing else center_label
                    ctx_str = f"{src} --[{rel_type}: {rel_desc}]--> {tgt} ({tgt_desc})"

                    edge_key = f"{src.lower()}-{rel_type}-{tgt.lower()}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        all_edges.append({
                            "source_id": (center_node["id"] if center_node else entity_id) if is_outgoing else tgt_id,
                            "target_id": tgt_id if is_outgoing else (center_node["id"] if center_node else entity_id),
                            "source_name": src,
                            "target_name": tgt,
                            "type": rel_type,
                            "description": rel_desc
                        })
                        if ctx_str not in context_parts:
                            context_parts.append(ctx_str)

        # Global or Broad Fallback: If no edges found or global route requested, include graph backbone
        if route == "global" or not all_edges:
            full_g = self.storage.get_full_graph()
            graph_entities = full_g.get("entities", [])
            graph_relations = full_g.get("relations", [])

            if graph_entities:
                for ent in graph_entities[:15]:
                    if ent["id"] not in all_nodes:
                        all_nodes[ent["id"]] = ent

            if graph_relations:
                for r in graph_relations[:20]:
                    s_name = r.get("source_name") or r["source_id"]
                    t_name = r.get("target_name") or r["target_id"]
                    r_type = r.get("type", "RELATED")
                    r_desc = r.get("description", "")
                    edge_key = f"{s_name.lower()}-{r_type}-{t_name.lower()}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        all_edges.append(r)
                        ctx_str = f"{s_name} --[{r_type}: {r_desc}]--> {t_name}"
                        if ctx_str not in context_parts:
                            context_parts.append(ctx_str)

        graph_context = "\n".join(context_parts)
        return {
            "graph_nodes": list(all_nodes.values()),
            "graph_edges": all_edges,
            "graph_context": graph_context
        }

    async def _vector_retrieval_node(self, state: GraphRAGState) -> Dict[str, Any]:
        query = state["query"]
        nodes = state.get("graph_nodes", [])
        combined = []

        # 1. Search actual stored contextual chunks in LadybugDB
        matched_chunks = self.storage.search_chunks(query, limit=4)
        chunk_snippets = []
        for i, c in enumerate(matched_chunks):
            ctext = c.get("text", "").strip()
            cctx = c.get("context", "").strip()
            if ctext:
                if cctx:
                    chunk_snippets.append(f"[Excerpt {i+1}]: {ctext}\n[Document Context]: {cctx}")
                else:
                    chunk_snippets.append(f"[Excerpt {i+1}]: {ctext}")

        # 2. Enrich with matching node descriptions as text context
        text_snippets = []
        for n in nodes:
            desc = n.get("description", "")
            name = n.get("name", n.get("id", ""))
            if desc and desc != "No description":
                text_snippets.append(f"[{name}]: {desc}")

        vector_parts = []
        if chunk_snippets:
            vector_parts.append("=== Document Passages & Contextual Excerpts ===\n" + "\n\n".join(chunk_snippets))
        if text_snippets:
            vector_parts.append("=== Entity Descriptions ===\n" + "\n".join(text_snippets))

        vector_context = "\n\n".join(vector_parts)

        if state.get("graph_context"):
            combined.append(f"=== Knowledge Graph Relationships ===\n{state['graph_context']}")
        if vector_context:
            combined.append(vector_context)

        return {
            "vector_context": vector_context,
            "combined_context": "\n\n".join(combined)
        }

    async def _synthesizer_node(self, state: GraphRAGState) -> Dict[str, Any]:
        query = state["query"]
        context = state.get("combined_context", "")

        if not context.strip():
            full_g = self.storage.get_full_graph()
            existing_entities = full_g.get("entities", [])
            existing_relations = full_g.get("relations", [])
            existing_chunks = self.storage.get_all_chunks()

            if not existing_entities and not existing_chunks:
                answer = "The knowledge graph is currently empty. Ingest text, files, or select a sample dataset above to start exploring."
            else:
                # Construct fallback context from full graph and chunks so the LLM answers factually
                overview_lines = []
                if existing_relations:
                    overview_lines.append("=== Knowledge Graph Facts ===")
                    for r in existing_relations[:15]:
                        s_name = r.get("source_name") or r["source_id"]
                        t_name = r.get("target_name") or r["target_id"]
                        overview_lines.append(f"- {s_name} --[{r.get('type')}]--> {t_name}: {r.get('description', '')}")

                if existing_entities:
                    overview_lines.append("\n=== Key Entities ===")
                    for e in existing_entities[:10]:
                        overview_lines.append(f"- **{e.get('name')}** ({e.get('type')}): {e.get('description', '')}")

                if existing_chunks:
                    overview_lines.append("\n=== Document Excerpts ===")
                    for c in existing_chunks[:3]:
                        overview_lines.append(f"- {c.get('text', '')}")

                fallback_context = "\n".join(overview_lines)
                prompt = (
                    f"You are a knowledge graph AI assistant.\n"
                    f"Answer the user's question clearly, thoroughly, and factually using the following knowledge graph context.\n"
                    f"If the exact answer is not mentioned, explain what information is present in the knowledge base.\n\n"
                    f"Context:\n{fallback_context}\n\n"
                    f"Question: {query}\n"
                    f"Answer:"
                )
                try:
                    res = await self.llm.generate_async(prompt)
                    answer = str(res) if isinstance(res, (dict, list)) else res
                except Exception as e:
                    logger.warning(f"Fallback synthesis error: {e}")
                    sample_names = [e["name"] for e in existing_entities[:6] if e.get("name")]
                    answer = f"I couldn't find specific relations for '{query}' in the current graph. Try asking about existing indexed entities such as: **" + ", ".join(sample_names) + "**."
        else:
            prompt = (
                f"You are a knowledge graph AI assistant.\n"
                f"Answer the user's question clearly, thoroughly, and factually using the following verified knowledge graph context and document passages.\n"
                f"Be informative, accurate, and direct.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            try:
                res = await self.llm.generate_async(prompt)
                answer = str(res) if isinstance(res, (dict, list)) else res
                if not answer or "Error generating" in answer:
                    raise ValueError("Empty LLM synthesis response")
            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}. Generating structured response directly from retrieved triples.")
                triples = [line.strip() for line in state.get("graph_context", "").split("\n") if line.strip()]
                nodes = state.get("graph_nodes", [])
                node_bullets = [f"**{n.get('name', n.get('id'))}** ({n.get('type', 'Concept')}): {n.get('description', '')}" for n in nodes if n.get('description')]
                
                parts = [f"### Knowledge Graph Findings for \"{query}\":\n"]
                if triples:
                    parts.append("#### Key Relationships:\n" + "\n".join([f"- {t}" for t in triples[:10]]))
                if node_bullets:
                    parts.append("\n#### Entity Details:\n" + "\n".join([f"- {nb}" for nb in node_bullets[:6]]))
                answer = "\n".join(parts)

        return {"answer": answer}

    async def _grader_node(self, state: GraphRAGState) -> Dict[str, Any]:
        context = state.get("combined_context", "")
        answer = state.get("answer", "")
        # If context was present and answer produced, mark grounded
        is_grounded = bool(context.strip() and "couldn't find specific relations" not in answer and "couldn't find any relevant" not in answer)
        return {
            "is_grounded": is_grounded,
            "retry_count": state.get("retry_count", 0) + 1
        }

    async def run(self, query: str, query_type: str = "local") -> Dict[str, Any]:
        initial_state: GraphRAGState = {
            "query": query,
            "query_type": query_type,
            "route": "hybrid",
            "extracted_entities": [],
            "graph_context": "",
            "vector_context": "",
            "combined_context": "",
            "graph_nodes": [],
            "graph_edges": [],
            "answer": "",
            "retry_count": 0,
            "is_grounded": False
        }

        if self.graph_app:
            final_state = await self.graph_app.ainvoke(initial_state)
        else:
            # Fallback direct execution if LangGraph is not installed in current host
            s1 = await self._router_node(initial_state)
            initial_state.update(s1)
            s2 = await self._graph_retrieval_node(initial_state)
            initial_state.update(s2)
            s3 = await self._vector_retrieval_node(initial_state)
            initial_state.update(s3)
            s4 = await self._synthesizer_node(initial_state)
            initial_state.update(s4)
            s5 = await self._grader_node(initial_state)
            initial_state.update(s5)
            final_state = initial_state

        return {
            "answer": final_state.get("answer", ""),
            "entities": final_state.get("graph_nodes", []),
            "relations": final_state.get("graph_edges", []),
            "route": final_state.get("route", "hybrid"),
            "extracted_entities": final_state.get("extracted_entities", []),
            "graph_context": final_state.get("graph_context", ""),
            "vector_context": final_state.get("vector_context", ""),
            "is_grounded": final_state.get("is_grounded", True),
            "retry_count": final_state.get("retry_count", 0)
        }
