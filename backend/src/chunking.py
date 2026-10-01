import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from src.llm_service import LLMBackend

@dataclass
class EnrichedSentence:
    text: str
    context: str = ""
    original_id: str = ""

@dataclass
class FinalChunk:
    id: str
    text: str                     # Target extraction chunk (400-600 tokens)
    context: str = ""             # Short label/excerpt
    preceding_context: str = ""   # Up to 35,000 tokens of preceding document text
    forward_lookahead: str = ""   # Up to 1,500 tokens of forward document text
    start_char_idx: int = 0
    end_char_idx: int = 0
    sentences: List[EnrichedSentence] = field(default_factory=list)


class ContextualChunker:
    """
    Macro-Context Horizon Chunker.
    Partitions documents into coherent 400-600 token target chunks,
    preserving continuous chronological preceding context (up to 35,000 tokens)
    and forward lookahead horizon (up to 1,500 tokens) for zero-loss extraction.
    """
    def __init__(
        self,
        llm: Optional[LLMBackend] = None,
        target_chunk_tokens: int = 500,
        max_chunk_tokens: int = 650,
        lookback_tokens: int = 35000,
        lookahead_tokens: int = 1500
    ):
        self.llm = llm
        self.target_chunk_tokens = int(os.getenv("TARGET_CHUNK_TOKENS", str(target_chunk_tokens)))
        self.max_chunk_tokens = int(os.getenv("MAX_CHUNK_TOKENS", str(max_chunk_tokens)))
        self.lookback_tokens = int(os.getenv("HORIZON_LOOKBACK_TOKENS", str(lookback_tokens)))
        self.lookahead_tokens = int(os.getenv("HORIZON_LOOKAHEAD_TOKENS", str(lookahead_tokens)))

        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())

    def _slice_tokens_from_end(self, text: str, max_tokens: int) -> str:
        """Returns at most max_tokens from the end of the text string."""
        if not text:
            return ""
        if self._count_tokens(text) <= max_tokens:
            return text
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            return self.tokenizer.decode(tokens[-max_tokens:])
        words = text.split()
        return " ".join(words[-max_tokens:])

    def _slice_tokens_from_start(self, text: str, max_tokens: int) -> str:
        """Returns at most max_tokens from the start of the text string."""
        if not text:
            return ""
        if self._count_tokens(text) <= max_tokens:
            return text
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            return self.tokenizer.decode(tokens[:max_tokens])
        words = text.split()
        return " ".join(words[:max_tokens])

    def _split_into_base_chunks(self, text: str) -> List[str]:
        """Splits text into chunks respecting paragraph and sentence boundaries."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []

        raw_chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            if current_chunk and (current_tokens + para_tokens > self.max_chunk_tokens):
                raw_chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            raw_chunks.append("\n\n".join(current_chunk))

        # If a single chunk is still too large, break by sentence
        final_splits = []
        for chunk in raw_chunks:
            if self._count_tokens(chunk) > self.max_chunk_tokens:
                import nltk
                try:
                    sents = nltk.sent_tokenize(chunk)
                except LookupError:
                    nltk.download('punkt', quiet=True)
                    nltk.download('punkt_tab', quiet=True)
                    try:
                        sents = nltk.sent_tokenize(chunk)
                    except Exception:
                        sents = chunk.split(". ")
                
                sub_chunk = []
                sub_tokens = 0
                for s in sents:
                    st = self._count_tokens(s)
                    if sub_chunk and (sub_tokens + st > self.target_chunk_tokens):
                        final_splits.append(" ".join(sub_chunk))
                        sub_chunk = [s]
                        sub_tokens = st
                    else:
                        sub_chunk.append(s)
                        sub_tokens += st
                if sub_chunk:
                    final_splits.append(" ".join(sub_chunk))
            else:
                final_splits.append(chunk)

        return final_splits if final_splits else [text]

    async def chunk_async(self, full_text: str):
        """
        Async generator that splits the document into target chunks (400-600 tokens)
        and attaches the surrounding preceding context (up to 35,000 tokens)
        and forward lookahead horizon (up to 1,500 tokens) without costly separate LLM calls.
        """
        base_chunk_texts = self._split_into_base_chunks(full_text)
        total = len(base_chunk_texts)

        yield {
            "type": "progress",
            "stage": "Horizon Chunking",
            "current": 0,
            "total": total,
            "status": f"Partitioned document into {total} target chunks. Slicing macro-context horizons..."
        }

        final_chunks: List[FinalChunk] = []
        search_cursor = 0

        for i, chunk_text in enumerate(base_chunk_texts):
            # Locate chunk character offsets in full text
            start_idx = full_text.find(chunk_text, search_cursor)
            if start_idx == -1:
                start_idx = full_text.find(chunk_text)
            if start_idx == -1:
                start_idx = search_cursor
            end_idx = start_idx + len(chunk_text)
            search_cursor = end_idx

            # Extract raw preceding and forward string slices
            raw_preceding = full_text[:start_idx].strip()
            raw_forward = full_text[end_idx:].strip()

            # Bound them strictly by token limits
            preceding_context = self._slice_tokens_from_end(raw_preceding, self.lookback_tokens)
            forward_lookahead = self._slice_tokens_from_start(raw_forward, self.lookahead_tokens)

            final_chunks.append(FinalChunk(
                id=f"chunk_{i+1}",
                text=chunk_text,
                context=f"Chunk {i+1}/{total}",
                preceding_context=preceding_context,
                forward_lookahead=forward_lookahead,
                start_char_idx=start_idx,
                end_char_idx=end_idx,
                sentences=[EnrichedSentence(text=chunk_text, context="", original_id=f"chunk_{i+1}")]
            ))

        yield {
            "type": "progress",
            "stage": "Horizon Chunking",
            "current": total,
            "total": total,
            "status": f"Generated {total} chunks with 35k preceding & 1.5k forward macro-context horizons"
        }

        yield final_chunks

# Backwards compatibility alias
HorizonChunker = ContextualChunker
