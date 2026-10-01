from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str | List[Dict[str, Any]], schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        pass

    async def generate_async(self, prompt: str | List[Dict[str, Any]], schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        return await asyncio.to_thread(self.generate, prompt, schema, temperature)


class LiteLLMBackend(LLMBackend):
    """
    Backend for connecting to Gemini via Vertex AI ADC or GEMINI_API_KEY.
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
        self.vertex_project = vertex_project

        vertex_location = os.getenv("VERTEX_LOCATION", "europe-west4")
        if vertex_location:
            os.environ.setdefault("VERTEX_LOCATION", vertex_location)
        self.vertex_location = vertex_location
        self.active_model: Optional[str] = None

    def _prepare_call(self, prompt: str | List[Dict[str, Any]], schema: Optional[Any] = None, temperature: Optional[float] = None):
        kwargs: Dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        else:
            if self.vertex_location:
                kwargs["vertex_location"] = self.vertex_location
            if self.vertex_project:
                kwargs["vertex_project"] = self.vertex_project

        if temperature is not None:
            kwargs["temperature"] = temperature
            
        if schema and hasattr(schema, "model_json_schema"):
            kwargs["response_format"] = schema
            
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        return messages, kwargs

    def generate(self, prompt: str | List[Dict[str, Any]], schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        import litellm
        messages, kwargs = self._prepare_call(prompt, schema, temperature)
            
        if self.active_model:
            candidate_models = [self.active_model]
        else:
            candidate_models = [self.model_name]
            if "3.8-flash" in self.model_name:
                provider = "vertex_ai/" if self.model_name.startswith("vertex_ai/") else "gemini/"
                candidate_models.append(f"{provider}gemini-2.5-flash")
                candidate_models.append(f"{provider}gemini-2.0-flash")

        last_error = None
        for model in candidate_models:
            try:
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                self.active_model = model
                content = response.choices[0].message.content
                if schema:
                    try:
                        clean_content = content.replace("```json", "").replace("```", "").strip()
                        return json.loads(clean_content)
                    except json.JSONDecodeError:
                        logger.error("LiteLLM failed to return valid JSON.")
                return content
            except Exception as e:
                last_error = e
                err_str = str(e)
                if ("404" in err_str or "NOT_FOUND" in err_str) and len(candidate_models) > 1:
                    logger.debug(f"Model {model} returned 404. Falling back to alternative model...")
                    continue
                break

        err_str = str(last_error) if last_error else ""
        if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in err_str or "403" in err_str:
            logger.error(
                "GCE VM service account has restricted OAuth scopes. "
                "To enable live Gemini extraction, please add GEMINI_API_KEY=your_key to backend/.env "
                "or authenticate via 'gcloud auth application-default login --no-browser'."
            )
        else:
            logger.error(f"LiteLLM generate failed: {last_error}")
        if schema: return {}
        return ""

    async def generate_async(self, prompt: str | List[Dict[str, Any]], schema: Optional[Any] = None, temperature: Optional[float] = None) -> str | Dict[str, Any]:
        import litellm
        messages, kwargs = self._prepare_call(prompt, schema, temperature)
            
        if self.active_model:
            candidate_models = [self.active_model]
        else:
            candidate_models = [self.model_name]
            if "3.8-flash" in self.model_name:
                provider = "vertex_ai/" if self.model_name.startswith("vertex_ai/") else "gemini/"
                candidate_models.append(f"{provider}gemini-2.5-flash")
                candidate_models.append(f"{provider}gemini-2.0-flash")

        last_error = None
        for model in candidate_models:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                self.active_model = model
                content = response.choices[0].message.content
                if schema:
                    try:
                        clean_content = content.replace("```json", "").replace("```", "").strip()
                        return json.loads(clean_content)
                    except json.JSONDecodeError:
                        logger.error("LiteLLM failed to return valid JSON.")
                return content
            except Exception as e:
                last_error = e
                err_str = str(e)
                if ("404" in err_str or "NOT_FOUND" in err_str) and len(candidate_models) > 1:
                    logger.debug(f"Model {model} returned 404. Falling back to alternative model...")
                    continue
                break

        err_str = str(last_error) if last_error else ""
        if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in err_str or "403" in err_str:
            logger.error(
                "GCE VM service account has restricted OAuth scopes. "
                "To enable live Gemini extraction, please add GEMINI_API_KEY=your_key to backend/.env "
                "or authenticate via 'gcloud auth application-default login --no-browser'."
            )
        else:
            logger.error(f"LiteLLM generate_async failed: {last_error}")
        if schema: return {}
        return ""


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