"""Dense semantic index built on sentence-transformers with disk caching.

Provides a lightweight numpy-backed cosine sim index over precomputed chunk
embeddings so the search pipeline can add a true semantic leg on top of the
lexical BM25 leg. Embeddings are L2-normalised at build time, and the whole
matrix is persisted to disk as an ``.npz`` file keyed by a content fingerprint
so rebuilds are skipped when the KB is unchanged.

Supports contextual retrieval (Anthropic-style): prepend chunk context
(title + heading_path) to content before embedding for better semantic matching.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional

import numpy as np

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Context template for contextual retrieval (Anthropic-style)
_CONTEXT_TEMPLATE = "Title: {title}\nHeading: {heading}\nContent: {content}"


class DenseSemanticIndex:
    """Cosine-similarity index over precomputed chunk embeddings.

    Args:
        model_name: HuggingFace identifier used to embed the query at search
            time.  The corpus itself is embedded lazily via
            :meth:`build`.  An external (mock) embedder can be supplied
            through :attr:`embed_fn` for testing.
        use_context: If True, prepend title + heading to content for
            contextual retrieval (Anthropic-style).
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        batch_size: int = 64,
        embed_fn: Any | None = None,
        use_context: bool = True,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._embed_fn = embed_fn
        self._use_context = use_context
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # (n, dim), L2-normalised rows
        self._model = None

    # ------------------------------------------------------------------
    # Contextual text building
    # ------------------------------------------------------------------

    @staticmethod
    def build_context_text(
        content: str,
        title: str = "",
        heading: str = "",
        chunk_type: str = "",
    ) -> str:
        """Build contextual text for embedding (Anthropic-style).

        Prepends title and heading path to content for better semantic matching.
        """
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if heading:
            parts.append(f"Heading: {heading}")
        if chunk_type == "qa_pair":
            parts.append(f"Type: Q&A")
        parts.append(f"Content: {content}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Embedding backend
    # ------------------------------------------------------------------

    def _ensure_model(self):
        """Lazily load the sentence-transformer model."""
        if self._model is not None:
            return self._model
        if self._embed_fn is not None:
            return self._embed_fn
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - env specific
            raise ImportError(
                "sentence-transformers is required for dense search.  "
                "Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return np.asarray(self._embed_fn(texts), dtype=np.float32)
        model = self._ensure_model()
        vecs = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Index construction / persistence
    # ------------------------------------------------------------------

    @staticmethod
    def fingerprint(
        texts: list[str],
        titles: Optional[list[str]] = None,
        headings: Optional[list[str]] = None,
        chunk_types: Optional[list[str]] = None,
        model_name: str = _MODEL_NAME,
        use_context: bool = True,
    ) -> str:
        """Content fingerprint used to invalidate the disk cache.

        Includes contextual metadata and model identity so a heading/title
        change or model switch correctly invalidates the cache (F4 fix).
        """
        h = hashlib.sha256()
        h.update(model_name.encode("utf-8"))
        h.update(b"|use_context=" + (b"1" if use_context else b"0"))
        # Use contextual texts if available so title/heading changes invalidate
        if use_context and titles is not None and headings is not None:
            for i, t in enumerate(texts):
                title = titles[i] if i < len(titles) else ""
                heading = headings[i] if i < len(headings) else ""
                ctype = chunk_types[i] if chunk_types and i < len(chunk_types) else ""
                ctx = DenseSemanticIndex.build_context_text(t, title, heading, ctype)
                h.update(len(ctx).to_bytes(4, "little"))
                h.update(ctx.encode("utf-8"))
        else:
            for t in texts:
                h.update(len(t).to_bytes(4, "little"))
                h.update(t.encode("utf-8"))
        return h.hexdigest()

    def build(
        self,
        ids: list[str],
        texts: list[str],
        titles: Optional[list[str]] = None,
        headings: Optional[list[str]] = None,
        chunk_types: Optional[list[str]] = None,
    ) -> None:
        """Embed *texts* (aligned with *ids*) and store the L2-normalised matrix.

        If titles/headings/chunk_types are provided, builds contextual text
        for each chunk before embedding (Anthropic-style contextual retrieval).
        """
        if not ids:
            self._ids = []
            self._matrix = np.zeros((0, 384), dtype=np.float32)
            return

        # Build contextual texts if metadata provided
        if self._use_context and titles and headings:
            contextual_texts = []
            for i, text in enumerate(texts):
                title = titles[i] if i < len(titles) else ""
                heading = headings[i] if i < len(headings) else ""
                chunk_type = chunk_types[i] if chunk_types and i < len(chunk_types) else ""
                ctx = self.build_context_text(text, title, heading, chunk_type)
                contextual_texts.append(ctx)
            texts_to_embed = contextual_texts
        else:
            texts_to_embed = texts

        vecs = self._encode(texts_to_embed)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        # Guard: renormalise rows defensively.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = (vecs / norms).astype(np.float32)
        self._ids = list(ids)

    def load(self, path: Path) -> bool:
        """Load a cached index from *path* (``.npz``).  Returns False on failure."""
        try:
            data = np.load(str(path), allow_pickle=False)
            ids = [x.decode("utf-8") for x in data["ids"]]
            self._matrix = np.asarray(data["vectors"], dtype=np.float32)
            self._ids = ids
            return True
        except (OSError, KeyError, ValueError):
            return False

    def save(self, path: Path, fingerprint: str) -> None:
        """Persist the index and its fingerprint to *path*."""
        if self._matrix is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(path),
            ids=np.asarray([x.encode("utf-8") for x in self._ids]),
            vectors=self._matrix,
            fingerprint=np.asarray([fingerprint.encode("utf-8")]),
        )

    def cached_valid(self, path: Path, fingerprint: str) -> bool:
        """Check whether *path* holds an index matching *fingerprint*."""
        try:
            data = np.load(str(path), allow_pickle=False)
            stored = data["fingerprint"][0].decode("utf-8")
            return stored == fingerprint
        except (OSError, KeyError, ValueError, IndexError):
            return False

    # ------------------------------------------------------------------
    # Query time
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def is_built(self) -> bool:
        return self._matrix is not None and len(self._ids) > 0

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, cosine_sim)]`` sorted best-first."""
        if not self.is_built:
            return []
        qv = self._encode([query])[0].astype(np.float32)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        qv = qv / qn
        sims = self._matrix @ qv  # rows already normalised -> cosine
        order = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in order]


def load_or_build(
    cache_path: Path,
    ids: list[str],
    texts: list[str],
    titles: Optional[list[str]] = None,
    headings: Optional[list[str]] = None,
    chunk_types: Optional[list[str]] = None,
    model_name: str = _MODEL_NAME,
    batch_size: int = 64,
    embed_fn: Any | None = None,
    use_context: bool = True,
) -> DenseSemanticIndex:
    """Return a dense index from cache if valid, else build and persist.

    The corpus is fingerprinted so a changed KB transparently triggers a
    rebuild.  When ``embed_fn`` is provided (e.g. a test double) it is used
    instead of the sentence-transformer model and caching is skipped.
    """
    fp = DenseSemanticIndex.fingerprint(texts, titles, headings, chunk_types, model_name, use_context)
    index = DenseSemanticIndex(
        model_name=model_name, batch_size=batch_size, embed_fn=embed_fn, use_context=use_context
    )
    if embed_fn is not None:
        index.build(ids, texts, titles, headings, chunk_types)
        return index
    if cache_path and index.cached_valid(cache_path, fp):
        if index.load(cache_path) and len(index._ids) == len(ids):
            return index
    index.build(ids, texts, titles, headings, chunk_types)
    if cache_path:
        index.save(cache_path, fp)
    return index