from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import json
import time
import requests
import torch
import logging
import asyncio

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging
    transformers_logging.set_verbosity_error()
except ImportError:
    pass


try:
    from aiolimiter import AsyncLimiter
except ImportError:
    AsyncLimiter = None


# --- Logging Setup ---
logger = logging.getLogger(__name__)

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        pass

    async def generate_async(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        """Default async implementation delegates to sync generation in a thread."""
        return await asyncio.to_thread(self.generate, prompt, schema)

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    def get_perplexity(self, text: str) -> float:
        return 0.0

    @abstractmethod
    def check_causal_link(self, context: str, sentence: str) -> float:
        return 0.5

# --- Implementations ---

class GeminiBackend(LLMBackend):
    def __init__(self, model_name: str = "gemini-3-flash-preview", api_key: Optional[str] = None):
        import google.generativeai as genai
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.embedding_model = "models/gemini-embedding-001"
        self.model_name = model_name
        
        # Rate Limiting setup
        rpm_limit = int(os.getenv("GEMINI_RPM_LIMIT", "15"))
        self.limiter = AsyncLimiter(rpm_limit, 60) if AsyncLimiter else None
        
    def generate(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        # Synchronous version (minimal fallback)
        generation_config = {}
        is_gemma = "gemma" in self.model_name.lower()
        
        if schema and not is_gemma:
            generation_config["response_mime_type"] = "application/json"
            if isinstance(schema, type) or hasattr(schema, "__annotations__"):
                 generation_config["response_schema"] = schema

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return self._parse_response(response, schema, is_gemma)
        except Exception as e:
            logger.error(f"Gemini generate error: {e}")
            return ""

    async def generate_async(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        # Async version with Rate Limiting
        if self.limiter:
            async with self.limiter:
                return await self._generate_internal_async(prompt, schema)
        return await self._generate_internal_async(prompt, schema)

    async def _generate_internal_async(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        
        generation_config = {}
        is_gemma = "gemma" in self.model_name.lower()
        
        if schema and not is_gemma:
            generation_config["response_mime_type"] = "application/json"
            if isinstance(schema, type) or hasattr(schema, "__annotations__"):
                 generation_config["response_schema"] = schema

        try:
            # Check if async method exists (it should in recent versions)
            if hasattr(self.model, "generate_content_async"):
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=generation_config
                )
                return self._parse_response(response, schema, is_gemma)
            else:
                # Fallback to thread pool if async generate is not available
                return await asyncio.to_thread(self.generate, prompt, schema)
        except Exception as e:
            logger.error(f"Gemini async generate error: {e}")
            return ""


    def _parse_response(self, response, schema, is_gemma):
        text = response.text
        if schema and is_gemma:
            # Manual JSON cleanup
            text = text.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from {self.model_name}")
                return {} 
        elif schema:
            return json.loads(text)
            
        return text

    def embed(self, text: str) -> List[float]:
        import google.generativeai as genai
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Gemini embed error: {e}")
            return []

    async def embed_async(self, text: str) -> List[float]:
        if self.limiter:
            async with self.limiter:
                return await asyncio.to_thread(self.embed, text)
        return await asyncio.to_thread(self.embed, text)


    def get_perplexity(self, text: str) -> float:
        return 0.0

    def check_causal_link(self, context: str, sentence: str) -> float:
        return 0.5

class LocalHuggingFaceBackend(LLMBackend):
    def __init__(self, model_id: str = "gpt2", device: str = None, local_dir: str = "models"):
        self.model_id = model_id
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        if torch.cuda.is_available(): self.device = "cuda"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=local_dir).to(self.device)
        self.model.eval()

    def generate(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        return "" # Mostly used for PPL/Chunking in this setup

    # generate_async uses default implementation which runs generate in a thread
    # Since generate is empty/unused here, it's fine. 
    # If we implement generation, PyTorch releases GIL, so threading is okay.

    def embed(self, text: str) -> List[float]:
        return []

    def get_perplexity(self, text: str) -> float:
        if not text: return 0.0
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()

    def check_causal_link(self, context: str, sentence: str) -> float:
        # Cross-Entropy Loss logic
        full_text = f"{context} {sentence}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        context_ids = self.tokenizer(context, return_tensors="pt")["input_ids"]
        len_context = context_ids.shape[1]
        
        labels = input_ids.clone()
        if len_context < labels.shape[1]:
             labels[:, :len_context] = -100
        else:
             return 0.0
             
        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
        
        loss = outputs.loss.item()
        similarity = torch.exp(torch.tensor(-loss)).item()
        return similarity

class OllamaBackend(LLMBackend):
    def __init__(self, model_name: str, embedding_model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.base_url = base_url

    def generate(self, prompt: str, schema: Optional[Any] = None) -> str | Dict[str, Any]:
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            if schema:
                payload["format"] = "json"

            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            text = data.get("response", "")

            if schema and isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from {self.model_name}")
                    return {}
            return text
        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            return ""

    # generate_async uses default (threads) because requests is blocking.
    # We could implement aiohttp here if we wanted to add that dependency.

    def embed(self, text: str) -> List[float]:
        try:
            url = f"{self.base_url}/api/embeddings"
            payload = {
                "model": self.embedding_model_name,
                "prompt": text
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])
        except Exception as e:
            logger.error(f"Ollama embed error: {e}")
            return []
        
    def get_perplexity(self, text: str) -> float:
        return 0.0

    def check_causal_link(self, context: str, sentence: str) -> float:
        # Placeholder for Ollama-based causal link check
        return 0.5

# --- Service Manager ---

class LLMService:
    def __init__(self, chunking_model: LLMBackend, extraction_model: LLMBackend):
        self.chunking_model = chunking_model
        self.extraction_model = extraction_model
        
    @property
    def chunker(self) -> LLMBackend:
        return self.chunking_model
        
    @property
    def extractor(self) -> LLMBackend:
        return self.extraction_model

