from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.ingestion import DocumentNode
from src.llm_service import LLMBackend
import numpy as np

@dataclass
class EnrichedSentence:
    text: str
    context: str # e.g. "Header 1 > Header 2"
    original_id: str

@dataclass
class FinalChunk:
    id: str
    text: str
    context: str
    start_char_idx: int
    end_char_idx: int
    sentences: List[EnrichedSentence]

class AncestryInjector:
    def inject(self, nodes: List[DocumentNode], parent_context: str = "") -> List[EnrichedSentence]:
        """
        Flattens the tree into a list of sentences with ancestry context.
        """
        sentences = []
        for node in nodes:
            # Build context
            current_context = parent_context
            if node.metadata.get("type") == "header":
                sep = " > " if current_context else ""
                current_context = f"{current_context}{sep}{node.text}"
            
            # If node has children, recurse
            if node.children:
                sentences.extend(self.inject(node.children, current_context))
            
            # If node is content, split into sentences (simple split for now) and add
            if node.metadata.get("type") == "content":
                # Simple sentence splitting by '.'
                # In prod, use nltk or spacy
                raw_sentences = [s.strip() for s in node.text.split('.') if s.strip()]
                for s in raw_sentences:
                    sentences.append(EnrichedSentence(
                        text=s + ".",
                        context=current_context,
                        original_id=node.id
                    ))
        return sentences

class PerplexityChunker:
    def __init__(self, llm: LLMBackend, max_tokens: int = 500, logger=None, ppl_threshold: float = 100.0):
        self.llm = llm
        self.max_tokens = max_tokens
        self.logger = logger
        self.ppl_threshold = ppl_threshold

    def chunk(self, sentences: List[EnrichedSentence]) -> List[FinalChunk]:
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        chunk_counter = 1
        
        for i, sentence in enumerate(sentences):
            # Estimate tokens (approx 4 chars/token)
            sent_tokens = len(sentence.text) // 4
            
            # Decision Logic
            # 1. Hard Constraint: Max tokens
            if current_chunk_sentences and (current_token_count + sent_tokens > self.max_tokens):
                chunks.append(self._create_chunk(current_chunk_sentences, chunk_counter))
                chunk_counter += 1
                current_chunk_sentences = []
                current_token_count = 0
            
            # 2. Soft Constraint: Perplexity / Semantic Distance
            # If we have a current chunk, check if next sentence is surprising
            if current_chunk_sentences:
                ppl = self._calculate_perplexity(current_chunk_sentences, sentence)
                if ppl > self.ppl_threshold:
                     chunks.append(self._create_chunk(current_chunk_sentences, chunk_counter))
                     chunk_counter += 1
                     current_chunk_sentences = []
                     current_token_count = 0
            
            current_chunk_sentences.append(sentence)
            current_token_count += sent_tokens
            
        if current_chunk_sentences:
            chunks.append(self._create_chunk(current_chunk_sentences, chunk_counter))
            
        return chunks

    def _create_chunk(self, sentences: List[EnrichedSentence], limit_id: int) -> FinalChunk:
        full_text = " ".join([s.text for s in sentences])
        # Use context of the first sentence as dominant context? 
        # Or mixed? Usually context is stable within a chunk if we split on headers (which we don't explicitly here, but Injector handles order)
        # Actually, AncestryInjector traverse order implies we stay within context until we change.
        context = sentences[0].context if sentences else ""
        return FinalChunk(
            id=f"chunk_{limit_id}",
            text=full_text,
            context=context,
            start_char_idx=0, # placeholders
            end_char_idx=0,
            sentences=sentences
        )

    def _calculate_perplexity(self, current_batch: List[EnrichedSentence], next_sent: EnrichedSentence) -> float:
        # If the LLM backend supports causal check (like our LocalHuggingFaceBackend), use it.
        if hasattr(self.llm, "check_causal_link"):
            # Build context from last few sentences
            context_text = " ".join([s.text for s in current_batch[-3:]])
            score = self.llm.check_causal_link(context_text, next_sent.text)
            
            # Score is Similarity (0.0 to 1.0). Higher = Better fit.
            # Convert to "Perplexity" (Lower = Better fit).
            if score < 0.0001: score = 0.0001
            ppl = 1.0 / score
            
            # Use logger if available, else print
            if self.logger:
                self.logger.log_perplexity(context_text, next_sent.text, score, ppl)
            else:
                print(f"  [Chunker] Similarity: {score:.4f} -> PPL: {ppl:.2f}")

            return ppl
        
        return 0.0
