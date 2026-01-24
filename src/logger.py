import os
import logging

class PipelineLogger:
    def __init__(self, log_dir: str = "logs", enabled: bool = False):
        self.log_dir = log_dir
        self.enabled = enabled
        if enabled and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # File paths
        self.chunk_log_path = os.path.join(log_dir, "enriched_input_chunkwise.md")
        self.cont_log_path = os.path.join(log_dir, "enriched_input_continuous.md")
        self.ppl_log_path = os.path.join(log_dir, "perplexity_scores.log")
        
        # Clear existing logs if enabled
        if enabled:
            open(self.chunk_log_path, 'w').close()
            open(self.cont_log_path, 'w').close()
            open(self.ppl_log_path, 'w').close()

    def log_chunk(self, chunk_id: int, text: str, context: str):
        if not self.enabled: return
        with open(self.chunk_log_path, 'a') as f:
            f.write(f"## Chunk {chunk_id}\n")
            f.write(f"**Context**: `{context}`\n\n")
            f.write(f"{text}\n\n---\n\n")

    def log_continuous(self, text: str):
        if not self.enabled: return
        # Just append text largely unmodified to read like a doc
        with open(self.cont_log_path, 'a') as f:
            f.write(f"{text}\n")

    def log_perplexity(self, context: str, sentence: str, score: float, ppl: float):
        if not self.enabled: return
        with open(self.ppl_log_path, 'a') as f:
            f.write(f"Context: {context[-50:]}...\n")
            f.write(f"Sentence: {sentence[:30]}...\n")
            f.write(f"Sim Score: {score:.4f} | PPL: {ppl:.2f}\n")
            f.write("-" * 40 + "\n")

    def info(self, msg: str):
        if self.enabled:
            print(f"[INFO] {msg}")

    def error(self, msg: str):
        if self.enabled:
            print(f"[ERROR] {msg}")
