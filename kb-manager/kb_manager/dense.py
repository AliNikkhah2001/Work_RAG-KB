"""Dense semantic index built on sentence-transformers with disk caching.

Provides a lightweight numpy-backed cosine sim index over precomputed chunk
embeddings so the search pipeline can add a true semantic leg on top of the
lexical BM25 leg. Embeddings are L2-normalised at build time, and the whole
matrix is persisted to disk as an ``.npz`` file keyed by a content fingerprint
so rebuilds are skipped when the KB is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class DenseSemanticIndex:
    """Cosine-similarity index over precomputed chunk embeddings.

    Args:
        model_name: HuggingFace identifier used to embed the query at search
            time.  The corpus itself is embedded lazily via
            :meth:`build`.  An external (mock) embedder can be supplied
            through :attr:`embed_fn` for testing.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        batch_size: int = 64,
        embed_fn: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._embed_fn = embed_fn
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # (n, dim), L2-normalised rows
        self._model = None

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
    def fingerprint(texts: list[str]) -> str:
        """Content fingerprint used to invalidate the disk cache."""
        h = hashlib.sha256()
        for t in texts:
            h.update(len(t).to_bytes(4, "little"))
            h.update(t.encode("utf-8"))
        return h.hexdigest()

    def build(self, ids: list[str], texts: list[str]) -> None:
        """Embed *texts* (aligned with *ids*) and store the L2-normalised matrix."""
        if not ids:
            self._ids = []
            self._matrix = np.zeros((0, 384), dtype=np.float32)
            return
        vecs = self._encode(texts)
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
    model_name: str = _MODEL_NAME,
    batch_size: int = 64,
    embed_fn: Any | None = None,
) -> DenseSemanticIndex:
    """Return a dense index from cache if valid, else build and persist.

    The corpus is fingerprinted so a changed KB transparently triggers a
    rebuild.  When ``embed_fn`` is provided (e.g. a test double) it is used
    instead of the sentence-transformer model and caching is skipped.
    """
    fp = DenseSemanticIndex.fingerprint(texts)
    index = DenseSemanticIndex(
        model_name=model_name, batch_size=batch_size, embed_fn=embed_fn
    )
    if embed_fn is not None:
        index.build(ids, texts)
        return index
    if cache_path and index.cached_valid(cache_path, fp):
        if index.load(cache_path) and len(index._ids) == len(ids):
            return index
    index.build(ids, texts)
    if cache_path:
        index.save(cache_path, fp)
    return index