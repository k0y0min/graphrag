from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import asyncio
import logging
from src.llm_service import LLMBackend
from src.chunking import FinalChunk

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

# Define the exact structure we want the LLM to output
class ExtractedEntity(BaseModel):
    id: str = Field(description="The canonical NAME of the entity (e.g. 'Apple', 'Elon Musk', 'SpaceX')")
    type: str = Field(description="The category of the entity (e.g., 'Person', 'Organization', 'Technology', 'Project')")
    description: str = Field(description="Brief summary of what this entity is in this context")
    aliases: List[str] = Field(default_factory=list, description="Other names, abbreviations, acronyms, or nicknames used for this entity in the text")

class ExtractedRelation(BaseModel):
    source: str = Field(description="The ID of the source entity")
    target: str = Field(description="The ID of the target entity")
    type: str = Field(description="The type of relationship (e.g., 'DEVELOPED', 'PART_OF', 'WORKS_FOR')")
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
    aliases: List[str] = field(default_factory=list)

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

    async def extract_async(self, chunks: List[FinalChunk], nodes: Optional[List[Any]] = None):
        """
        Async generator for Macro-Context Horizon extraction.
        Maintains a Cumulative Entity Registry across chunks so canonical entity
        names and aliases propagate across the document without isolation gaps.
        """
        entities: Dict[str, Entity] = {}
        relations: List[Relation] = []
        known_registry: Dict[str, Dict[str, Any]] = {}

        total = len(chunks)
        completed = 0
        results = [None] * total

        # High concurrency extraction across chunks
        semaphore = asyncio.Semaphore(6)

        def _format_known_entities() -> str:
            if not known_registry:
                return "None yet identified."
            lines = []
            for eid, data in list(known_registry.items())[-25:]: # Most recent 25 entities
                aliases_str = f" | Aliases: {data.get('aliases', [])}" if data.get('aliases') else ""
                lines.append(f"- {eid} ({data.get('type', 'Concept')}): {data.get('description', '')[:100]}{aliases_str}")
            return "\n".join(lines)

        async def _run_extraction(idx: int, chunk: FinalChunk):
            async with semaphore:
                known_str = _format_known_entities()
                extracted = await self._extract_from_chunk_async(chunk, known_str)
                # Register new entities into the cumulative registry for subsequent chunks
                for ent in extracted.get("entities", []):
                    eid = ent.get("id")
                    if eid and eid not in known_registry:
                        known_registry[eid] = {
                            "type": ent.get("type", "Concept"),
                            "description": ent.get("description", ""),
                            "aliases": ent.get("aliases", [])
                        }
                results[idx] = extracted
            return idx

        tasks = [asyncio.ensure_future(_run_extraction(i, c)) for i, c in enumerate(chunks)]

        for future in asyncio.as_completed(tasks):
            await future
            completed += 1
            yield {
                "type": "progress",
                "stage": "Knowledge Extraction",
                "current": completed,
                "total": total,
                "status": f"Extracting triples with macro-context: {completed}/{total} chunks"
            }

        # Merge extracted results
        for chunk, extracted in zip(chunks, results):
            if not extracted:
                continue
            for ent_dict in extracted.get("entities", []):
                eid = ent_dict.get("id")
                etype = ent_dict.get("type")
                edescr = ent_dict.get("description", "")
                ealiases = ent_dict.get("aliases", [])
                if not eid:
                    continue

                if eid in entities:
                    existing = entities[eid]
                    if len(existing.description) < len(edescr):
                        existing.description = edescr
                    if chunk.id not in existing.source_chunk_ids:
                        existing.source_chunk_ids.append(chunk.id)
                    for a in ealiases:
                        if a and a not in existing.aliases:
                            existing.aliases.append(a)
                else:
                    entities[eid] = Entity(
                        id=eid,
                        type=etype,
                        description=edescr,
                        source_chunk_ids=[chunk.id],
                        aliases=list(ealiases) if ealiases else []
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

    async def _extract_from_chunk_async(self, chunk: FinalChunk, known_entities_str: str = "") -> Dict[str, Any]:
        preceding = (getattr(chunk, 'preceding_context', '') or '').strip()
        forward = (getattr(chunk, 'forward_lookahead', '') or '').strip()

        preceding_block = f"<preceding_context>\n{preceding}\n</preceding_context>\n" if preceding else "<preceding_context>\n(Start of document - no preceding text)\n</preceding_context>\n"
        forward_block = f"<forward_lookahead>\n{forward}\n</forward_lookahead>\n" if forward else "<forward_lookahead>\n(End of document - no forward text)\n</forward_lookahead>\n"

        prompt1 = (
            f"<system_guidelines>\n"
            f"You are an expert Knowledge Graph Extractor.\n"
            f"Your task is to extract all valid entities and relationships strictly asserted or active within <target_chunk>.\n"
            f"The surrounding text in <preceding_context> and <forward_lookahead> is provided solely for co-reference resolution, anaphora resolution, entity disambiguation, and background situational awareness.\n\n"
            f"Rules:\n"
            f"1. Target Chunk Scope: Only extract entities and relations that are mentioned, active, or asserted within <target_chunk>. Do NOT extract entities that appear exclusively in <preceding_context> or <forward_lookahead>.\n"
            f"2. Speaker & Subject Resolution: Identify the primary author, candidate, speaker, organization, or central subject from the surrounding document context. Resolve all first-person pronouns ('I', 'me', 'my', 'we', 'our team') and unattributed accomplishments/actions directly to the explicit canonical entity name (e.g. 'Dr. Jane Smith', 'Ayush Dongre', 'NASA'), NEVER to generic placeholders like 'User', 'Author', 'Narrator', or 'Speaker'.\n"
            f"3. Entity Consistency & Aliases: If an entity corresponds to an entry in <known_entities>, reuse its exact canonical ID and record any observed alternative mentions or abbreviations in the 'aliases' field.\n"
            f"4. Exact Identity Mapping: Distinct mentions of the exact same real-world concept within this chunk MUST share the identical canonical ID string.\n"
            f"5. Exhaustive Coverage & Linkages: Extract all specialized technical relationships, implicit causal connections, component architectures, and secondary entities mentioned in <target_chunk> in this single pass.\n"
            f"</system_guidelines>\n\n"
            f"<known_entities>\n{known_entities_str or 'None yet identified.'}\n</known_entities>\n\n"
            f"<document_stream>\n"
            f"{preceding_block}\n"
            f"<target_chunk>\n{chunk.text}\n</target_chunk>\n\n"
            f"{forward_block}"
            f"</document_stream>\n\n"
            f"<extraction_task>\n"
            f"Extract all entities and relationships strictly for the content inside <target_chunk>.\n"
            f"Output valid JSON conforming to the GraphExtractionSchema.\n"
            f"</extraction_task>"
        )

        try:
            res1_text = await self.llm.generate_async(prompt1, schema=GraphExtractionSchema)
            parsed1 = json.loads(res1_text) if isinstance(res1_text, str) else res1_text

            combined = {
                "entities": parsed1.get("entities", []) if isinstance(parsed1, dict) else [],
                "relations": parsed1.get("relations", []) if isinstance(parsed1, dict) else []
            }
            if not combined.get("entities"):
                logger.info(f"LLM returned no entities for chunk {chunk.id}. Activating heuristic fallback extractor.")
                return self._heuristic_extract(chunk)
            return combined

        except Exception as e:
            logger.warning(f"Extraction failed for chunk {chunk.id}: {e}. Activating heuristic fallback extractor.")
            return self._heuristic_extract(chunk)

    def _heuristic_extract(self, chunk: FinalChunk) -> Dict[str, Any]:
        """
        Graceful heuristic entity/relationship extractor fallback.
        Ensures the UI graph visualizer renders nodes and connections even when
        the cloud LLM backend is unauthenticated or restricted by VM scopes.
        """
        import re
        words = re.findall(r'\b[A-Z][a-zA-Z0-9_\-\']*(?:\s+[A-Z][a-zA-Z0-9_\-\']*)*\b', chunk.text)
        stopwords = {
            "The", "This", "That", "These", "Those", "A", "An", "In", "On", "At", "By",
            "For", "With", "About", "Against", "Between", "Into", "Through", "During",
            "Before", "After", "Above", "Below", "To", "From", "Up", "Down", "If",
            "Because", "As", "Until", "While", "Of", "Although", "However", "Therefore",
            "Moreover", "Furthermore", "Rules", "Context", "Text", "Note", "User", "Here",
            "Please", "Answer", "It", "They", "We", "You", "He", "She"
        }
        found_entities = {}
        ordered_ids = []
        for w in words:
            clean_name = w.strip()
            if len(clean_name) < 3 or clean_name in stopwords:
                continue
            if clean_name not in found_entities:
                found_entities[clean_name] = {
                    "id": clean_name,
                    "type": "Concept",
                    "description": f"Entity observed in {chunk.id}",
                    "aliases": []
                }
                ordered_ids.append(clean_name)

        relations = []
        for i in range(len(ordered_ids) - 1):
            src = ordered_ids[i]
            tgt = ordered_ids[i+1]
            relations.append({
                "source": src,
                "target": tgt,
                "type": "ASSOCIATED_WITH",
                "description": f"Co-occurrence linkage in {chunk.id}"
            })

        return {
            "entities": list(found_entities.values()),
            "relations": relations
        }
