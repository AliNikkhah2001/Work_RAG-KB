"""HyDE — Hypothetical Document Embeddings for improved retrieval.

Generates a hypothetical answer via LLM, embeds it with the same dense model
used for corpus indexing, and returns a query vector that better matches
answer-like content in the knowledge base.

Key benefit: keyword_only queries (e.g. "بازپرداخت وام") produce poor dense
similarity because the query is too short and lacks answer-like structure.
HyDE bridges this gap by generating a full answer paragraph first.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (ACL 2022).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default prompt template for Persian KB HyDE
_DEFAULT_HYDE_PROMPT = """شما یک متخصص بانکداری و امور مالی ایران هستید.
بر اساس سوال زیر، یک پاسخ کوتاه و مختصر (۲ تا ۳ جمله) بنویسید که احتمالاً
در پایگاه دانش بانکی وجود داشته باشد. فقط پاسخ را بنویسید، توضیح اضافه ندهید.

سوال: {query}

پاسخ:"""


class HyDEGenerator:
    """Generate hypothetical document embeddings for query expansion.

    Uses an LLM to generate a hypothetical answer, then encodes it with the
    same sentence-transformer model used for corpus embeddings.

    Args:
        llm_model: OpenAI-compatible model identifier.
        llm_api_key: API key for the LLM provider.
        llm_base_url: Base URL for OpenAI-compatible API.
        embedding_model: Sentence-transformer model name (must match corpus).
        prompt_template: Template with {query} placeholder.
        num_hypotheses: Number of hypothetical answers to generate (1-5).
            Multiple hypotheses improve recall at the cost of latency.
        cache_hypotheses: Whether to cache generated hypotheses per query.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: str = "",
        llm_base_url: str = "",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        prompt_template: str = _DEFAULT_HYDE_PROMPT,
        num_hypotheses: int = 1,
        cache_hypotheses: bool = True,
    ) -> None:
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY", "")
        self._llm_base_url = llm_base_url or os.getenv("OPENAI_BASE_URL", "")
        self._embedding_model = embedding_model
        self._prompt_template = prompt_template
        self._num_hypotheses = max(1, min(5, num_hypotheses))
        self._cache_hypotheses = cache_hypotheses

        self._embedder: Any = None
        self._cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_embedder(self):
        """Lazily load the sentence-transformer model for embedding."""
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for HyDE. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def _call_llm(self, prompt: str) -> str:
        """Call OpenAI-compatible chat completion API."""
        import httpx

        if not self._llm_api_key:
            logger.warning("No LLM API key configured for HyDE; skipping")
            return ""

        base_url = self._llm_base_url.rstrip("/")
        if not base_url:
            base_url = "https://api.openai.com/v1"

        headers = {
            "Authorization": f"Bearer {self._llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.3,
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("HyDE LLM call failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_hypothesis(self, query: str) -> str:
        """Generate a hypothetical answer for the query via LLM."""
        prompt = self._prompt_template.format(query=query)
        return self._call_llm(prompt)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string using the dense model."""
        model = self._ensure_embedder()
        vec = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)

    def get_hyde_vector(
        self,
        query: str,
        use_cache: bool = True,
    ) -> Optional[np.ndarray]:
        """Generate a HyDE embedding vector for the query.

        Returns None if LLM is unavailable or generation fails.
        """
        if use_cache and self._cache_hypotheses and query in self._cache:
            return self._cache[query]

        hypothesis = self.generate_hypothesis(query)
        if not hypothesis:
            return None

        logger.debug("HyDE hypothesis for %r: %s", query[:50], hypothesis[:100])
        vec = self.embed(hypothesis)

        if self._cache_hypotheses:
            self._cache[query] = vec

        return vec

    def get_hyde_vector_multi(
        self,
        query: str,
    ) -> list[np.ndarray]:
        """Generate multiple HyDE vectors for the query (beam search).

        Returns a list of vectors (up to num_hypotheses). Empty list on failure.
        """
        vectors: list[np.ndarray] = []
        for i in range(self._num_hypotheses):
            v = self.get_hyde_vector(query, use_cache=(i == 0))
            if v is not None:
                vectors.append(v)
        return vectors

    def clear_cache(self) -> None:
        """Clear the hypothesis cache."""
        self._cache.clear()

    @property
    def is_available(self) -> bool:
        """Check if HyDE can be used (API key configured)."""
        return bool(self._llm_api_key)


def get_hyde_generator(
    llm_model: str = "gpt-4o-mini",
    llm_api_key: str = "",
    llm_base_url: str = "",
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    num_hypotheses: int = 1,
) -> HyDEGenerator:
    """Factory function to create a HyDE generator."""
    return HyDEGenerator(
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        embedding_model=embedding_model,
        num_hypotheses=num_hypotheses,
    )
