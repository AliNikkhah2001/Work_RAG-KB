"""OpenRouter embeddings backend (laptop / no local GPU models).

Calls ``POST {base}/embeddings`` with the OpenAI-compatible schema.
Uses the shared ``OPENROUTER_API_KEY`` (or ``KB_OPENROUTER_API_KEY``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import numpy as np

from kb_manager.embedder.base import BaseEmbedder

logger = logging.getLogger(__name__)

_DEFAULT_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/text-embedding-3-small"
_DEFAULT_DIM = 1536


def _api_key() -> str:
    key = (
        os.getenv("KB_OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "OpenRouter embedder requires KB_OPENROUTER_API_KEY or OPENROUTER_API_KEY"
        )
    return key


def _base_url() -> str:
    return (
        os.getenv("KB_OPENROUTER_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or _DEFAULT_BASE
    ).rstrip("/")


class OpenRouterEmbedder(BaseEmbedder):
    """HTTP embedder targeting OpenRouter ``/embeddings``."""

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        timeout: float = 60.0,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model_name = (
            model_name
            or os.getenv("KB_EMBED_MODEL")
            or _DEFAULT_MODEL
        )
        self._dimensions = int(
            dimensions
            or os.getenv("KB_EMBED_DIM")
            or _DEFAULT_DIM
        )
        self._batch_size = max(1, int(batch_size))
        self._normalize = normalize
        self._timeout = timeout
        self._api_key = api_key
        self._base_url = (base_url or _base_url()).rstrip("/")
        self._dim_seen: int | None = None

    @property
    def dimensions(self) -> int:
        return self._dim_seen or self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            out.extend(self._embed_batch(batch))
        return out

    def as_numpy_fn(self):
        """Return a ``list[str] -> np.ndarray`` suitable for DenseSemanticIndex.embed_fn."""

        def _fn(texts: list[str]) -> np.ndarray:
            rows = self.embed_texts(list(texts))
            arr = np.asarray(rows, dtype=np.float32)
            if self._normalize and arr.size:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms
            return arr

        return _fn

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        key = self._api_key or _api_key()
        payload: dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }
        # Only send dimensions when the caller pinned one (OpenAI-style models).
        if self._dimensions and "openai/" in self._model_name:
            payload["dimensions"] = self._dimensions

        url = f"{self._base_url}/embeddings"
        try:
            with httpx.Client(timeout=self._timeout, trust_env=False) as client:
                resp = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            raise RuntimeError(
                f"OpenRouter embeddings failed model={self._model_name!r}: {e}"
            ) from e

        items = data.get("data") or []
        if len(items) != len(texts):
            raise RuntimeError(
                f"OpenRouter embeddings count mismatch: got {len(items)} for {len(texts)} inputs"
            )
        # OpenAI schema returns objects with index; sort to be safe.
        items = sorted(items, key=lambda x: int(x.get("index", 0)))
        vectors: list[list[float]] = []
        for item in items:
            vec = item.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise RuntimeError("OpenRouter embeddings response missing embedding vector")
            vectors.append([float(x) for x in vec])
            self._dim_seen = len(vec)
        return vectors
