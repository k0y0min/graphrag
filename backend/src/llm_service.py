from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import json
import requests
import logging
import asyncio

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI, AsyncOpenAI = None, None

logger = logging.getLogger(__name__)



class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        pass

    async def generate_async(self, prompt: str, schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        return await asyncio.to_thread(self.generate, prompt, schema, temperature)

    @abstractmethod
    def get_sentence_perplexities(self, sentences: List[str]) -> List[float]:
        pass

    async def get_sentence_perplexities_async(self, sentences: List[str]) -> List[float]:
        return await asyncio.to_thread(self.get_sentence_perplexities, sentences)

class VLLMBackend(LLMBackend):
    """
    Backend for connecting to a local vLLM server.
    Optimized for Gemma-4 / Qwen using guided_json structured outputs.
    """
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1", api_key: str = "dummy-key"):
        if not OpenAI:
            raise ImportError("Please install openai: pip install openai")
            
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def get_sentence_perplexities_async(self, sentences: List[str]) -> List[float]:
        """
        Calculates real perplexity using vLLM's logprobs API.
        """
        if not sentences: return []
        
        full_text = " ".join(sentences)
        
        try:
            response = await self.async_client.completions.create(
                model=self.model_name,
                prompt=full_text,
                max_tokens=0,
                echo=True,
                logprobs=1,
                timeout=3.0
            )
            
            token_logprobs = response.choices[0].logprobs.token_logprobs
            token_logprobs = [lp if lp is not None else 0.0 for lp in token_logprobs]
            text_offsets = response.choices[0].logprobs.text_offset
            
            sentence_ppls = []
            current_token_idx = 0
            text_offset = 0
            
            for sent in sentences:
                sent_logprobs = []
                start_off = full_text.find(sent, text_offset)
                
                if start_off == -1: 
                    sentence_ppls.append(0.0)
                    continue
                    
                end_off = start_off + len(sent)
                text_offset = end_off
                
                for i, offset in enumerate(text_offsets[current_token_idx:], start=current_token_idx):
                    if offset >= end_off:
                        break
                    if offset >= start_off:
                        sent_logprobs.append(token_logprobs[i])
                        current_token_idx = i + 1
                
                if sent_logprobs:
                    avg_logprob = sum(sent_logprobs) / len(sent_logprobs)
                    import math
                    ppl = math.exp(-avg_logprob)
                    sentence_ppls.append(ppl)
                else:
                    sentence_ppls.append(0.0)
                    
            return sentence_ppls
        except Exception as e:
            logger.error(f"Failed to get perplexities from vLLM: {e}")
            return [0.0] * len(sentences)

    def get_sentence_perplexities(self, sentences: List[str]) -> List[float]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We shouldn't hit this if using async properly
                return [0.0] * len(sentences)
            return loop.run_until_complete(self.get_sentence_perplexities_async(sentences))
        except RuntimeError:
            return [0.0] * len(sentences)

    def generate(self, prompt: str, schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
            
        if schema and hasattr(schema, "model_json_schema"):
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema()
                }
            }
            
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        content = response.choices[0].message.content
        if schema:
            try:
                clean_content = content.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_content)
            except json.JSONDecodeError:
                logger.error("vLLM failed to return valid JSON despite response_format.")
        return content

    async def generate_async(self, prompt: str, schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
            
        if schema and hasattr(schema, "model_json_schema"):
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema()
                }
            }
            
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            content = response.choices[0].message.content
            if schema:
                try:
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_content)
                except json.JSONDecodeError:
                    logger.error("vLLM failed to return valid JSON despite response_format.")
            return content
        except Exception as e:
            logger.error(f"vLLM generate_async failed: {e}")
            if schema:
                return {}
            return ""

class LiteLLMBackend(LLMBackend):
    """
    Backend for connecting to external APIs via LiteLLM (e.g. Gemini).
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        import litellm
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
            
        if schema and hasattr(schema, "model_json_schema"):
            kwargs["response_format"] = schema
            
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            content = response.choices[0].message.content
            if schema:
                try:
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_content)
                except json.JSONDecodeError:
                    logger.error("LiteLLM failed to return valid JSON.")
            return content
        except Exception as e:
            logger.error(f"LiteLLM generate failed: {e}")
            if schema: return {}
            return ""

    async def generate_async(self, prompt: str, schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        import litellm
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
            
        if schema and hasattr(schema, "model_json_schema"):
            kwargs["response_format"] = schema
            
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            content = response.choices[0].message.content
            if schema:
                try:
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_content)
                except json.JSONDecodeError:
                    logger.error("LiteLLM failed to return valid JSON.")
            return content
        except Exception as e:
            logger.error(f"LiteLLM generate_async failed: {e}")
            if schema: return {}
            return ""

    def get_sentence_perplexities(self, sentences: List[str]) -> List[float]:
        return [0.0] * len(sentences)

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