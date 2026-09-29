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

class VLLMState:
    def __init__(self):
        use_vllm = os.getenv("USE_VLLM", "false").lower() == "true"
        self.is_ready = not use_vllm

vllm_state = VLLMState()

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
        if not vllm_state.is_ready:
            return [0.0] * len(sentences)
        
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
    Backend for connecting to external APIs via LiteLLM (e.g. Gemini 3.8/2.5-flash via ADC or GEMINI_API_KEY).
    Supports Anthropic-style cache_control for Gemini context caching.
    """
    def __init__(self, model_name: Optional[str] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        self.api_key = api_key
        
        default_model = "gemini/gemini-3.8-flash" if api_key else "vertex_ai/gemini-3.8-flash"
        raw_model = model_name or os.getenv("GEMINI_MODEL", default_model)
        
        # If user provided GEMINI_API_KEY and model has vertex_ai/ prefix, switch to Google AI Studio
        if api_key and raw_model.startswith("vertex_ai/"):
            clean_name = raw_model.replace("vertex_ai/", "")
            self.model_name = f"gemini/{clean_name}"
        else:
            self.model_name = raw_model
        
        # Configure Vertex project & location if available for ADC
        vertex_project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID")
        if not vertex_project:
            try:
                import urllib.request
                req = urllib.request.Request(
                    "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                    headers={"Metadata-Flavor": "Google"}
                )
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    vertex_project = resp.read().decode().strip()
            except Exception:
                pass
        if vertex_project:
            os.environ.setdefault("VERTEX_PROJECT", vertex_project)
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", vertex_project)
        vertex_location = os.getenv("VERTEX_LOCATION", "europe-west4")
        if vertex_location:
            os.environ.setdefault("VERTEX_LOCATION", vertex_location)

    def generate(self, prompt: str | List[Dict[str, Any]], schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        import litellm
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if temperature is not None:
            kwargs["temperature"] = temperature
            
        if schema and hasattr(schema, "model_json_schema"):
            kwargs["response_format"] = schema
            
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
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
            err_str = str(e)
            if ("NOT_FOUND" in err_str or "404" in err_str) and "3.8" in self.model_name:
                fallback_model = self.model_name.replace("3.8", "2.5")
                logger.warning(f"{self.model_name} not found on Vertex AI in this region. Falling back to {fallback_model}.")
                try:
                    response = litellm.completion(
                        model=fallback_model,
                        messages=messages,
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
                except Exception as fallback_err:
                    logger.error(f"Fallback {fallback_model} failed: {fallback_err}")
            elif "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in err_str or "403" in err_str:
                logger.error(
                    "GCE VM service account has restricted OAuth scopes. "
                    "To enable live Gemini extraction, please add GEMINI_API_KEY=your_key to backend/.env "
                    "or authenticate via 'gcloud auth application-default login --no-browser'."
                )
            else:
                logger.error(f"LiteLLM generate failed: {e}")
            if schema: return {}
            return ""

    async def generate_async(self, prompt: str | List[Dict[str, Any]], schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        import litellm
        kwargs = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if temperature is not None:
            kwargs["temperature"] = temperature
            
        if schema and hasattr(schema, "model_json_schema"):
            kwargs["response_format"] = schema
            
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            
        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
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
            err_str = str(e)
            if ("NOT_FOUND" in err_str or "404" in err_str) and "3.8" in self.model_name:
                fallback_model = self.model_name.replace("3.8", "2.5")
                logger.warning(f"{self.model_name} not found on Vertex AI in this region. Falling back to {fallback_model}.")
                try:
                    response = await litellm.acompletion(
                        model=fallback_model,
                        messages=messages,
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
                except Exception as fallback_err:
                    logger.error(f"Fallback {fallback_model} failed: {fallback_err}")
            elif "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in err_str or "403" in err_str:
                logger.error(
                    "GCE VM service account has restricted OAuth scopes. "
                    "To enable live Gemini extraction, please add GEMINI_API_KEY=your_key to backend/.env "
                    "or authenticate via 'gcloud auth application-default login --no-browser'."
                )
            else:
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