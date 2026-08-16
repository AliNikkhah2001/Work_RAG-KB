"""Sentence-Transformers based embedder with caching support."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from kb_manager.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_DIMENSIONS = 384


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding backend backed by ``sentence-transformers``.

    The model is lazily loaded on the first call to :meth:`embed_texts` or
    :meth:`embed_query`.  A SHA-256 content hash is maintained alongside each
    embedding so that unchanged texts can be skipped during incremental updates.

    Args:
        model_name: HuggingFace model identifier or local path.
        dimensions: Expected embedding dimensionality.  When ``None`` the
            value is read from the loaded model after first load.
        device: PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"`` …).
            ``None`` lets sentence-transformers choose automatically.
        batch_size: Number of texts to encode in a single forward pass.
        normalize: Whether to L2-normalise the output vectors.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        dimensions: int | None = None,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions or _DEFAULT_DIMENSIONS
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize

        self._model: SentenceTransformer | None = None
        self._hash_cache: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> SentenceTransformer:
        """Load the sentence-transformer model if it hasn't been loaded yet."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required.  "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading sentence-transformer model %s …", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)

        # Read actual dimensions from the model if not overridden.
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim is not None:
            self._dimensions = actual_dim

        logger.info(
            "Model loaded (dim=%d, device=%s)",
            self._dimensions,
            self._device or "auto",
        )
        return self._model

    # ------------------------------------------------------------------
    # Content hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _content_hash(text: str) -> str:
        """Return the hex SHA-256 digest of *text*."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, skipping unchanged content.

        Texts whose SHA-256 hash is already present in the internal cache are
        returned directly without invoking the model.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors aligned with the input order.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        if not texts:
            return []

        model = self._ensure_model()

        hashes: list[str] = [self._content_hash(t) for t in texts]
        results: list[list[float] | None] = [None] * len(texts)

        to_embed_indices: list[int] = []
        to_embed_texts: list[str] = []

        for idx, h in enumerate(hashes):
            if h in self._hash_cache:
                results[idx] = self._hash_cache[h]
            else:
                to_embed_indices.append(idx)
                to_embed_texts.append(texts[idx])

        if to_embed_texts:
            logger.debug(
                "Embedding %d new texts (out of %d total)",
                len(to_embed_texts),
                len(texts),
            )

            try:
                import numpy  # noqa: F401
            except ImportError:
                raw = model.encode(
                    to_embed_texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
                vectors: list[list[float]] = [
                    v.tolist() if hasattr(v, "tolist") else list(v) for v in raw
                ]
            else:
                raw = model.encode(
                    to_embed_texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                vectors = [row.tolist() for row in raw]

            for local_idx, vec in zip(to_embed_indices, vectors, strict=True):
                results[local_idx] = vec
                self._hash_cache[hashes[local_idx]] = vec

        return [r for r in results if r is not None]  # type: ignore[misc]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        For batch workloads prefer :meth:`embed_texts` which supports
        caching and batching.

        Args:
            query: The query text.

        Returns:
            Embedding vector.
        """
        results = self.embed_texts([query])
        return results[0]

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Name of the underlying sentence-transformer model."""
        return self._model_name

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Drop all cached content hashes."""
        self._hash_cache.clear()

    def __repr__(self) -> str:
        return (
            f"SentenceTransformerEmbedder(model={self._model_name!r}, "
            f"dim={self._dimensions}, device={self._device!r}, "
            f"batch_size={self._batch_size}, normalize={self._normalize})"
        )
