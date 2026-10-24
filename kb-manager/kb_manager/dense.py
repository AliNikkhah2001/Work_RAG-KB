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
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from kb_manager.model_cache import resolve_local_model_path

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Query-embedding LRU bounds.  Batches larger than this are treated as
# corpus-build traffic (see build()) and bypass the cache so a full
# re-embedding never evicts the hot query set.
_QUERY_CACHE_MAX = 2048
_QUERY_CACHE_MAX_BATCH = 8

logger = logging.getLogger(__name__)


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
        use_context: bool = False,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._embed_fn = embed_fn
        self._use_context = use_context
        self._device = device
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # (n, dim), L2-normalised rows
        self._model = None
        self._fp16 = False
        # Bounded LRU of query text -> 1-D float32 embedding (row-normalised).
        # Plain single-threaded access from the event loop; OrderedDict
        # move_to_end/popitem keeps it cheap without locks.
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()

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
        if self._device:
            self._model = SentenceTransformer(
                resolve_local_model_path(self._model_name), device=self._device
            )
        else:
            self._model = SentenceTransformer(resolve_local_model_path(self._model_name))
        # fp16 halves VRAM and speeds up the encode pass on CUDA.  Checking
        # the *loaded* model's device (rather than self._device) also covers
        # the auto-select path where device was left None.
        if str(getattr(self._model.device, "type", "")) == "cuda":
            try:
                self._model.half()
                self._fp16 = True
            except Exception:  # noqa: BLE001 - keeps service alive in fp32
                logger.warning("model.half() failed; keeping float32", exc_info=True)
        return self._model

    def _model_encode(self, model, texts: list[str]) -> np.ndarray:
        try:
            vecs = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception:  # noqa: BLE001 - fp16 revert guard
            if not self._fp16:
                raise
            # half() can succeed at load time yet break encode for some
            # pipeline combos; revert weights and retry once in float32.
            logger.warning(
                "encode failed with fp16 weights; reverting model to float32",
                exc_info=True,
            )
            model.float()
            self._fp16 = False
            vecs = model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        # The corpus .npz is float32, so fp16-computed vectors are cast back
        # to float32 here: this keeps the search-time matmul
        # (matrix @ qv) dtype-homogeneous, which is both correct for numpy
        # (mixed dtypes upcast) and faster than casting the whole corpus to
        # fp16 for a single query dot-product.  The fp16 win is purely in the
        # encode pass on the GPU.
        return np.asarray(vecs, dtype=np.float32)

    def _cache_get(self, text: str) -> np.ndarray | None:
        vec = self._query_cache.get(text)
        if vec is not None:
            self._query_cache.move_to_end(text)  # LRU: refresh recency
        return vec

    def _cache_put(self, text: str, vec: np.ndarray) -> None:
        # copy() so later downstream arithmetic can never mutate the cache
        self._query_cache[text] = vec.copy()
        self._query_cache.move_to_end(text)
        while len(self._query_cache) > _QUERY_CACHE_MAX:
            self._query_cache.popitem(last=False)  # evict least-recent

    def warmup(self) -> None:
        """Pre-load the model with one dummy encode at startup.

        No-op when an external embed_fn is supplied or the model is already
        loaded, so callers can call it unconditionally during lifespan
        pre-warm.
        """
        if self._embed_fn is not None or self._model is not None:
            return
        # Goes through _model_encode (not _encode) so a broken fp16 model is
        # detected and reverted here, and the dummy text stays out of the LRU.
        self._model_encode(self._ensure_model(), ["warmup"])

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return np.asarray(self._embed_fn(texts), dtype=np.float32)
        model = self._ensure_model()
        # LRU fast path for small (query-sized) batches: serve repeat queries
        # from cache and only send the misses to the GPU.
        if 1 <= len(texts) <= _QUERY_CACHE_MAX_BATCH:
            rows: list[np.ndarray | None] = [self._cache_get(t) for t in texts]
            miss_idx = [i for i, v in enumerate(rows) if v is None]
            if miss_idx:
                new_vecs = self._model_encode(model, [texts[i] for i in miss_idx])
                if new_vecs.ndim == 1:
                    new_vecs = new_vecs.reshape(1, -1)
                for i, v in zip(miss_idx, new_vecs):
                    rows[i] = v
                    self._cache_put(texts[i], v)
            return np.asarray(np.stack(rows), dtype=np.float32)
        return self._model_encode(model, texts)

    # ------------------------------------------------------------------
    # Index construction / persistence
    # ------------------------------------------------------------------

    @staticmethod
    def fingerprint(
        texts: list[str],
        ids: list[str] = [],
        titles: Optional[list[str]] = None,
        headings: Optional[list[str]] = None,
        chunk_types: Optional[list[str]] = None,
        model_name: str = _MODEL_NAME,
        use_context: bool = False,
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

    def search(self, query: str, top_k: int = 30, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        """Return ``[(chunk_id, cosine_sim)]`` sorted best-first."""
        if not self.is_built:
            return []
        qv = self._encode([query])[0].astype(np.float32)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return []
        qv = qv / qn
        sims = self._matrix @ qv  # rows already normalised -> cosine

        if allowed_ids is not None:
            # Mask out non-allowed ids by setting their similarity to -inf
            # use a simple list comprehension for boolean mask to avoid complex numpy string ops
            mask = np.array([cid not in allowed_ids for cid in self._ids])
            sims[mask] = -np.inf

        order = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in order if sims[i] != -np.inf]

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 30,
        allowed_ids: set[str] | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Batched multi-query search: one GPU encode + one stacked matmul.

        The beam-5 query path used to encode each variant separately (5 GPU
        round-trips) — a single batched encode plus a ``(n_q, n_chunks)``
        matmul is 3-5x faster and shares the query-embedding LRU cache.
        Returns one ``search()``-shaped list per input query, same order.
        """
        if not queries:
            return []
        if not self.is_built:
            return [[] for _ in queries]
        qv = self._encode(list(queries)).astype(np.float32)  # (n_q, dim)
        norms = np.linalg.norm(qv, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        qv = qv / norms
        sims = qv @ self._matrix.T  # (n_q, n_chunks) cosine

        if allowed_ids is not None:
            mask = np.array([cid not in allowed_ids for cid in self._ids])
            sims[:, mask] = -np.inf

        results: list[list[tuple[str, float]]] = []
        for row in sims:
            order = np.argsort(-row)[:top_k]
            results.append(
                [(self._ids[i], float(row[i])) for i in order if row[i] != -np.inf]
            )
        return results


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
    use_context: bool = False,
    device: str | None = None,
) -> DenseSemanticIndex:
    """Return a dense index from cache if valid, else build and persist.

    The corpus is fingerprinted so a changed KB transparently triggers a
    rebuild.  When ``embed_fn`` is provided (e.g. a test double) it is used
    instead of the sentence-transformer model and caching is skipped.
    """
    fp = DenseSemanticIndex.fingerprint(texts, ids, titles, headings, chunk_types, model_name, use_context)
    index = DenseSemanticIndex(
        model_name=model_name, batch_size=batch_size, embed_fn=embed_fn, use_context=use_context, device=device
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