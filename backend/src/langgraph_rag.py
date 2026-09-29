import os
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.llm_service import LLMBackend
from src.storage import GraphStorage

logger = logging.getLogger(__name__)

class QueryEntities(BaseModel):
    """Schema for extracting entity names from a user query."""
    entities: List[str] = Field(description="List of key named entity concepts found in the query")
    route: Optional[str] = Field(default="hybrid", description="'graph' for multi-hop/relational queries, 'vector' for specific factual text, 'hybrid' for complex questions")

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

        # Register nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("graph_retrieval", self._graph_retrieval_node)
        workflow.add_node("vector_retrieval", self._vector_retrieval_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        workflow.add_node("grader", self._grader_node)

        # Conditional routing from router
        def route_decision(state: GraphRAGState):
            r = state.get("route", "hybrid")
            if r == "graph":
                return "graph_retrieval"
            elif r == "vector":
                return "vector_retrieval"
            return "graph_retrieval"  # hybrid executes graph first then vector

        workflow.set_entry_point("router")
        workflow.add_conditional_edges("router", route_decision, {
            "graph_retrieval": "graph_retrieval",
            "vector_retrieval": "vector_retrieval"
        })

        # Connect retrieval to synthesis
        workflow.add_edge("graph_retrieval", "vector_retrieval")
        workflow.add_edge("vector_retrieval", "synthesizer")
        workflow.add_edge("synthesizer", "grader")

        # Grader evaluation
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
        prompt = (
            f"Analyze the following user query for Knowledge Graph and document retrieval.\n"
            f"Query: {query}\n"
            f"Extract all key named entities and determine the optimal route ('graph' for relationship/multi-hop questions, 'vector' for specific single facts, 'hybrid' for broad inquiries)."
        )
        try:
            res = await self.llm.generate_async(prompt, schema=QueryEntities)
            if isinstance(res, dict):
                entities = res.get("entities", [query])
                route = res.get("route", "hybrid")
            else:
                entities = [query]
                route = "hybrid"
        except Exception as e:
            logger.error(f"Router node extraction failed: {e}")
            entities = [query]
            route = "hybrid"

        # Override route if user explicitly requested local or structural
        if state.get("query_type") in ["local", "structural"]:
            route = "graph"

        return {
            "extracted_entities": entities,
            "route": route
        }

    async def _graph_retrieval_node(self, state: GraphRAGState) -> Dict[str, Any]:
        entities = state.get("extracted_entities", [])
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
                            "source_id": center_node["id"] if is_outgoing else tgt_id,
                            "target_id": tgt_id if is_outgoing else center_node["id"],
                            "source_name": src,
                            "target_name": tgt,
                            "type": rel_type,
                            "description": rel_desc
                        })
                        if ctx_str not in context_parts:
                            context_parts.append(ctx_str)

        graph_context = "\n".join(context_parts)
        return {
            "graph_nodes": list(all_nodes.values()),
            "graph_edges": all_edges,
            "graph_context": graph_context
        }

    async def _vector_retrieval_node(self, state: GraphRAGState) -> Dict[str, Any]:
        # If vector route or hybrid, enrich with matching node descriptions as text context
        nodes = state.get("graph_nodes", [])
        text_snippets = []
        for n in nodes:
            desc = n.get("description", "")
            name = n.get("name", n.get("id", ""))
            if desc:
                text_snippets.append(f"[{name}]: {desc}")

        vector_context = "\n".join(text_snippets)
        combined = []
        if state.get("graph_context"):
            combined.append(f"=== Knowledge Graph Relationships ===\n{state['graph_context']}")
        if vector_context:
            combined.append(f"=== Entity Details & Text Context ===\n{vector_context}")

        return {
            "vector_context": vector_context,
            "combined_context": "\n\n".join(combined)
        }

    async def _synthesizer_node(self, state: GraphRAGState) -> Dict[str, Any]:
        query = state["query"]
        context = state.get("combined_context", "")

        if not context.strip():
            answer = "I couldn't find any relevant information in the knowledge graph."
        else:
            prompt = (
                f"You are a helpful assistant answering using a structured Knowledge Graph.\n"
                f"Use the following verified context to answer the question thoroughly.\n"
                f"If the answer cannot be deduced from the context, state what is known and what is missing.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                f"Answer:"
            )
            try:
                res = await self.llm.generate_async(prompt)
                answer = str(res) if isinstance(res, (dict, list)) else res
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                answer = "Error generating answer from context."

        return {"answer": answer}

    async def _grader_node(self, state: GraphRAGState) -> Dict[str, Any]:
        context = state.get("combined_context", "")
        answer = state.get("answer", "")
        # If context was present and answer produced, mark grounded
        is_grounded = bool(context.strip() and "couldn't find any relevant" not in answer)
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
            "is_grounded": final_state.get("is_grounded", True)
        }
