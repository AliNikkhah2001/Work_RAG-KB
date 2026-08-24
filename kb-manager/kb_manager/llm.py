"""LLM client abstraction supporting multiple backends (OpenAI, Ollama, vLLM)."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    text: str
    metadata: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[LLMResponse]:
        """Generate text for multiple prompts."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client (supports OpenAI-compatible endpoints).

    Uses the synchronous ``openai.OpenAI`` client so it can be called from
    both sync and async contexts (e.g. inside a running event loop).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._timeout = timeout
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("openai package required for OpenAIClient") from e

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
        )

    def _chat(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
        self._ensure_client()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=response.choices[0].message.content or "",
            metadata={
                "model": self._model,
                "usage": usage.model_dump() if usage is not None else None,
            },
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        **kwargs,
    ) -> LLMResponse:
        return self._chat(prompt, max_tokens, temperature)

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[LLMResponse]:
        return [self._chat(p, max_tokens, temperature) for p in prompts]


class OllamaClient(LLMClient):
    """Ollama local LLM client."""

    def __init__(
        self,
        model: str = "gemma2:27b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        **kwargs,
    ) -> LLMResponse:
        import requests
        
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            text=data.get("response", "").strip(),
            metadata={"model": self._model, "done": data.get("done")},
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[LLMResponse]:
        # Sequential for simplicity; could be parallelized
        return [self.generate(p, max_tokens, temperature) for p in prompts]


class VLLMClient(LLMClient):
    """vLLM client for high-throughput local inference."""

    def __init__(
        self,
        model: str = "google/gemma-2-27b-it",
        base_url: str = "http://localhost:8000/v1",
        tensor_parallel_size: int = 2,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError("openai package required for VLLMClient") from e
        
        self._client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM doesn't require auth
            base_url=self._base_url,
            timeout=120,
        )

    async def _generate_async(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        self._ensure_client()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResponse(
            text=response.choices[0].message.content or "",
            metadata={"model": self._model},
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        **kwargs,
    ) -> LLMResponse:
        import asyncio
        return asyncio.run(self._generate_async(prompt, max_tokens, temperature))

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[LLMResponse]:
        import asyncio
        
        async def _batch():
            self._ensure_client()
            tasks = [
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": p}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                ) for p in prompts
            ]
            responses = await asyncio.gather(*tasks)
            return [
                LLMResponse(
                    text=r.choices[0].message.content or "",
                    metadata={"model": self._model},
                ) for r in responses
            ]
        
        import asyncio
        return asyncio.run(_batch())


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        **kwargs,
    ) -> LLMResponse:
        # Return mock response based on prompt type
        if "HyDE" in prompt or "پاسخ فرضی" in prompt:
            return LLMResponse(text="این یک پاسخ فرضی برای پرسش است که حاوی اصطلاحات کلیدی است.")
        if "JSON" in prompt or "صیغ مختلف" in prompt:
            return LLMResponse(text='[{"query": "تست", "type": "verbatim"}]')
        return LLMResponse(text="پاسخ پیش‌فرض")

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> List[LLMResponse]:
        return [self.generate(p, max_tokens, temperature) for p in prompts]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(
    backend: str = "mock",
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Factory function to create LLM client based on backend."""
    
    backends = {
        "openai": OpenAIClient,
        "ollama": OllamaClient,
        "vllm": VLLMClient,
        "mock": MockLLMClient,
    }
    
    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Choose from {list(backends.keys())}")
    
    client_class = backends[backend]
    
    # Set default model per backend
    defaults = {
        "openai": "gpt-4o-mini",
        "ollama": "gemma2:27b",
        "vllm": "google/gemma-2-27b-it",
    }
    
    model = model or defaults.get(backend, "gpt-4o-mini")
    return client_class(model=model, **kwargs)


def create_llm_client_from_config(config: Any) -> LLMClient:
    """Create LLM client from AppConfig."""
    backend = os.getenv("KB_LLM_BACKEND", "mock")
    model = os.getenv("KB_LLM_MODEL")
    
    # Use RAGAS config as fallback
    if hasattr(config, "ragas"):
        model = model or config.ragas.llm_model
    
    base_url = os.getenv("KB_LLM_BASE_URL")
    if backend == "ollama":
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif backend in ("openai", "vllm"):
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
    
    return create_llm_client(
        backend=backend,
        model=model,
        base_url=base_url,
    )