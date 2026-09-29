from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.llm_service import LLMBackend

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


class ContextualChunker:
    """
    Implements Anthropic's Contextual Retrieval (Sept 2024 SOTA).
    Splits text into coherent chunks and enriches each chunk with a
    succinct document-level situational context using prompt cache control.
    """
    def __init__(self, llm: LLMBackend, target_chunk_tokens: int = 350, max_chunk_tokens: int = 500):
        self.llm = llm
        self.target_chunk_tokens = target_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        import tiktoken
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())

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
        Async generator that splits the document and enriches chunks in parallel
        using Anthropic's Contextual Retrieval prompt with explicit cache_control.
        """
        base_chunks = self._split_into_base_chunks(full_text)
        total = len(base_chunks)

        yield {
            "type": "progress",
            "stage": "Contextual Retrieval",
            "current": 0,
            "total": total,
            "status": f"Created {total} base chunks. Initiating Anthropic contextual enrichment..."
        }

        import asyncio
        semaphore = asyncio.Semaphore(8)

        async def _enrich_chunk(idx: int, chunk_content: str):
            async with semaphore:
                prompt_messages = [
                    {
                        "role": "user",
                        "content": f"<document>\n{full_text}\n</document>",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Here is the chunk we want to situate within the whole document:\n"
                            f"<chunk>\n{chunk_content}\n</chunk>\n"
                            f"Please give a short succinct context (1-2 sentences, max 50 words) to situate this chunk within the overall document to improve search retrieval of the chunk. Answer only with the succinct context and nothing else."
                        )
                    }
                ]
                try:
                    context = await self.llm.generate_async(prompt_messages)
                    if isinstance(context, dict):
                        context = str(context)
                    context = context.strip()
                except Exception as e:
                    context = ""

                return idx, context, chunk_content

        tasks = [asyncio.ensure_future(_enrich_chunk(i, c)) for i, c in enumerate(base_chunks)]
        enriched_results = [None] * total
        completed = 0

        for future in asyncio.as_completed(tasks):
            idx, context, content = await future
            enriched_results[idx] = (context, content)
            completed += 1
            yield {
                "type": "progress",
                "stage": "Contextual Retrieval",
                "current": completed,
                "total": total,
                "status": f"Enriched {completed}/{total} chunks with document context"
            }

        final_chunks = []
        for i, res in enumerate(enriched_results):
            if res:
                ctx, raw_text = res
                enriched_text = f"Context: {ctx}\n\n{raw_text}" if ctx else raw_text
                final_chunks.append(FinalChunk(
                    id=f"chunk_{i+1}",
                    text=enriched_text,
                    context=ctx,
                    start_char_idx=0,
                    end_char_idx=0,
                    sentences=[EnrichedSentence(text=raw_text, context=ctx, original_id=f"chunk_{i+1}")]
                ))

        yield final_chunks


