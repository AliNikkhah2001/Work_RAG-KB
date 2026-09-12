"""Cross-encoder reranker for precision refinement in retrieval pipeline.

Uses a multilingual cross-encoder (mDeBERTa-v3-base-xsmall) to re-score
top candidates from the hybrid BM25+Dense retrieval stage.

Model selection:
    ``KB_RERANKER_MODEL`` env var selects the HuggingFace model id
    (default ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``). Known ids are
    listed in :data:`RERANKER_REGISTRY` with their loader requirements.

Candidate pool:
    ``KB_RERANK_POOL`` env var (int, default 0). ``0`` keeps the legacy
    cap (``min(50, top_k*3)`` — the 50 comes from the search-route
    pre-slice, ``top_k*3`` from :meth:`CrossEncoderReranker.rerank`);
    a value >0 overrides the candidate pool cap instead.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_DEFAULT_POOL = 0  # 0 = keep legacy min(50, top_k*3) pool logic
_MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps known reranker ids to loader specs. Two loaders are implemented:
#   'crossencoder' — standard AutoModelForSequenceClassification path with
#       trust_remote_code passthrough.
#   'flag' / 'flag-llm' — FlagEmbedding path (see FlagEmbeddingReranker):
#       'flag' uses FlagReranker (encoder-only sequence-classification
#       head); 'flag-llm' uses FlagLLMReranker (decoder-only causal LM,
#       scores the P('Yes') logit with the default A:/B: prompt template).
# The 'manual' loader (layerwise models, e.g. MiniCPM layerwise) needs
# custom per-layer scoring code and raises a clear NotImplementedError —
# support is NOT faked (see _ensure_model for the exact blockers).
RERANKER_REGISTRY: dict[str, dict[str, Any]] = {
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": {
        "loader": "crossencoder",
        "needs_trust_remote_code": False,
    },
    "BAAI/bge-reranker-v2-m3": {
        "loader": "crossencoder",
        "needs_trust_remote_code": False,
    },
    "Alibaba-NLP/gte-multilingual-reranker-base": {
        "loader": "crossencoder",
        "needs_trust_remote_code": False,
    },
    "jinaai/jina-reranker-v3": {
        "loader": "crossencoder",
        "needs_trust_remote_code": True,
    },
    "Qwen/Qwen3-Reranker-0.6B": {
        "loader": "flag-llm",
        "needs_trust_remote_code": False,
    },
    "Qwen/Qwen3-Reranker-4B": {
        "loader": "flag-llm",
        "needs_trust_remote_code": False,
    },
    "BAAI/bge-reranker-v2-gemma": {
        "loader": "flag-llm",
        "needs_trust_remote_code": False,
    },
    "BAAI/bge-reranker-v2-minicpm-layerwise": {
        "loader": "manual",
        "needs_trust_remote_code": True,
    },
}


def get_reranker_model_name() -> str:
    """Return the configured reranker model id (``KB_RERANKER_MODEL``)."""
    return os.getenv("KB_RERANKER_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL


def get_rerank_pool() -> int:
    """Return the configured candidate pool cap (``KB_RERANK_POOL``).

    Returns 0 when unset/invalid, meaning "keep the legacy pool logic".
    """
    try:
        return int(os.getenv("KB_RERANK_POOL", str(_DEFAULT_POOL)) or str(_DEFAULT_POOL))
    except (TypeError, ValueError):
        return _DEFAULT_POOL


def resolve_reranker_spec(model_name: str) -> dict[str, Any]:
    """Return the loader spec for a reranker model id.

    Unknown ids fall back to the standard cross-encoder path without
    trust_remote_code (backwards compatible for custom BERT-like models).
    """
    spec = RERANKER_REGISTRY.get(model_name)
    if spec is not None:
        return dict(spec)
    logger.warning(
        "Unknown reranker model %r — using standard cross-encoder path "
        "(trust_remote_code=False)",
        model_name,
    )
    return {"loader": "crossencoder", "needs_trust_remote_code": False}


class CrossEncoderReranker:
    """Rerank candidates using a cross-encoder model.

    The cross-encoder takes (query, passage) pairs and outputs a relevance score.
    This is more accurate than bi-encoder cosine similarity but slower.
    Used as a final precision refinement step on top-k candidates.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = _MAX_LENGTH,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._max_length = max_length

        self._model: Any = None
        self._tokenizer: Any = None
        self.last_rerank_ms: float = 0.0

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the cross-encoder model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        spec = resolve_reranker_spec(self._model_name)
        loader = spec.get("loader", "crossencoder")
        trust_remote_code = bool(spec.get("needs_trust_remote_code", False))

        if loader in ("flag", "flag-llm"):
            raise NotImplementedError(
                f"Reranker model {self._model_name!r} needs loader={loader!r}, which "
                "is implemented by FlagEmbeddingReranker, not CrossEncoderReranker. "
                "Use get_reranker() (it dispatches on the registry) instead of "
                "instantiating CrossEncoderReranker directly. "
                "Requires the FlagEmbedding package (pip install FlagEmbedding)."
            )
        if loader == "manual":
            raise NotImplementedError(
                f"Reranker model {self._model_name!r} needs loader='manual', which "
                "is not implemented: layerwise models (e.g. "
                "BAAI/bge-reranker-v2-minicpm-layerwise) expose per-layer "
                "representations and require the upstream LayerWise scoring logic. "
                "Blocked on transformers>=5 incompatibilities (verified 2026-09-12 "
                "with transformers 5.17.0 + FlagEmbedding 1.4.2, CPU-only torch): "
                "(1) the model's remote modeling_minicpm_reranker.py "
                "(transformers-4.38 era) does "
                "'from transformers.utils.import_utils import "
                "is_torch_fx_available', which no longer exists; "
                "(2) FlagEmbedding's bundled copy fails because transformers 5 "
                "normalizes config.rope_scaling=None to "
                "{'rope_theta': ..., 'rope_type': 'default'}, so "
                "MiniCPMAttention._init_rope raises KeyError: 'type' (it expects "
                "the old {'type': ..., 'factor': ...} format); "
                "(3) after patching rope_scaling=None, model __init__ fails in "
                "post_init/get_expanded_tied_weights_keys with "
                "AttributeError: 'list' object has no attribute 'keys' (old "
                "list-style tie_weights vs the dict-style transformers 5 "
                "expects). To support it, port the layerwise modeling code to "
                "transformers 5 (or pin transformers<5 in a separate env) and "
                "wire it up for loader='manual' in RERANKER_REGISTRY. Meanwhile, "
                "set KB_RERANKER_MODEL to a 'crossencoder' or 'flag-llm' entry "
                f"(e.g. the default {_DEFAULT_MODEL!r})."
            )
        if loader != "crossencoder":
            raise NotImplementedError(
                f"Reranker model {self._model_name!r} needs unknown loader "
                f"{loader!r}: add an implementation and register it in "
                "RERANKER_REGISTRY. Meanwhile, set KB_RERANKER_MODEL to a "
                "'crossencoder' entry (e.g. the default "
                f"{_DEFAULT_MODEL!r})."
            )

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for cross-encoder reranking. "
                "Install with: pip install transformers torch"
            ) from exc

        logger.info(
            "Loading cross-encoder model %s (trust_remote_code=%s) …",
            self._model_name,
            trust_remote_code,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, trust_remote_code=trust_remote_code
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
            trust_remote_code=trust_remote_code,
        )

        if self._device:
            self._model.to(self._device)
        else:
            import torch
            self._model.to("cuda" if torch.cuda.is_available() else "cpu")

        self._model.eval()
        logger.info(
            "Cross-encoder loaded (device=%s, dtype=%s)",
            next(self._model.parameters()).device,
            next(self._model.parameters()).dtype,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 10,
        score_key: str = "hybrid_score",
        pool: Optional[int] = None,
    ) -> List[dict]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: The search query.
            candidates: List of candidate dicts with at least 'content' or 'content_preview'.
            top_k: Number of top candidates to return after reranking.
            score_key: Key in candidate dict to use for initial ranking (before rerank).
            pool: Candidate pool cap. None (default) reads ``KB_RERANK_POOL``;
                0 keeps the legacy ``min(top_k*3, len)`` logic (the search
                route additionally pre-slices to 50, giving the effective
                legacy ``min(50, top_k*3)``); >0 overrides the pool cap.

        Returns:
            Top-k candidates sorted by cross-encoder score (descending),
            with 'rerank_score' added to each dict. The call latency is
            recorded on ``self.last_rerank_ms`` and logged at info level.
        """
        t0 = time.perf_counter()
        if not candidates:
            self.last_rerank_ms = 0.0
            logger.info("rerank model=%s n=0 ms=0.0", self._model_name)
            return []

        self._ensure_model()

        if pool is None:
            pool = get_rerank_pool()

        # Sort by initial score and take top candidates for reranking
        # (cross-encoder is expensive, so we only rerank a subset)
        ordered = sorted(
            candidates,
            key=lambda x: x.get(score_key, 0),
            reverse=True,
        )
        if pool and pool > 0:
            cap = min(pool, len(ordered))
        else:
            cap = min(top_k * 3, len(ordered))
        rerank_pool = ordered[:cap]

        # Prepare (query, passage) pairs
        pairs = []
        for cand in rerank_pool:
            text = cand.get("content") or cand.get("content_preview") or ""
            pairs.append((query, text))

        # Batch inference
        scores = self._score_pairs(pairs)

        # Attach scores and sort
        for cand, score in zip(rerank_pool, scores):
            cand["rerank_score"] = float(score)

        reranked = sorted(rerank_pool, key=lambda x: x["rerank_score"], reverse=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.last_rerank_ms = elapsed_ms
        logger.info(
            "rerank model=%s n=%d ms=%.1f",
            self._model_name,
            len(rerank_pool),
            elapsed_ms,
        )
        return reranked[:top_k]

    def _score_pairs(self, pairs: List[tuple[str, str]]) -> np.ndarray:
        """Score (query, passage) pairs in batches."""
        import torch

        all_scores = []
        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]

            inputs = self._tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )

            if self._device:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            else:
                import torch
                inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

            with torch.no_grad():
                logits = self._model(**inputs).logits
                # For binary classification, use sigmoid on the positive class
                if logits.shape[-1] == 1:
                    scores = torch.sigmoid(logits.squeeze(-1))
                else:
                    scores = torch.softmax(logits, dim=-1)[:, 1]

            all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores) if all_scores else np.array([])

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self._model_name!r}, batch_size={self._batch_size})"


class FlagEmbeddingReranker:
    """Rerank candidates using a FlagEmbedding reranker model.

    Same public interface as :class:`CrossEncoderReranker` — ``rerank(query,
    candidates, top_k, score_key, pool)`` returns top-k dicts with
    ``cand["rerank_score"]=float`` and records ``self.last_rerank_ms``.

    Loader dispatch (from :data:`RERANKER_REGISTRY`):

    * ``'flag'`` — encoder-only models via ``FlagEmbedding.FlagReranker``.
    * ``'flag-llm'`` — decoder-only causal-LM models (Qwen3-Reranker,
      bge-reranker-v2-gemma) via ``FlagEmbedding.FlagLLMReranker``, which
      scores the ``'Yes'`` logit under the default ``A:``/``B:`` prompt
      template. Scores are raw (unbounded) logits: only the rank order is
      meaningful, which is all :meth:`rerank` uses.

    Models always load in fp32 on CPU (``use_fp16=False``): fp16 on CPU is
    slow and unsupported in places; fp32 is kept for correctness. GPU hosts
    still work (FlagEmbedding moves the model to the requested device), the
    weights just stay fp32.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = _MAX_LENGTH,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._max_length = max_length

        self._reranker: Any = None
        self.last_rerank_ms: float = 0.0

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the FlagEmbedding reranker if not already loaded."""
        if self._reranker is not None:
            return

        spec = resolve_reranker_spec(self._model_name)
        loader = spec.get("loader", "flag-llm")

        try:
            import FlagEmbedding
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for reranker loader "
                f"{loader!r} (model {self._model_name!r}). "
                "Install with: pip install FlagEmbedding"
            ) from exc

        logger.info(
            "Loading FlagEmbedding reranker %s (loader=%s, fp32) …",
            self._model_name,
            loader,
        )

        devices = self._device if self._device else None
        if loader == "flag":
            self._reranker = FlagEmbedding.FlagReranker(
                self._model_name,
                use_fp16=False,
                max_length=self._max_length,
                devices=devices,
            )
        elif loader == "flag-llm":
            self._reranker = FlagEmbedding.FlagLLMReranker(
                self._model_name,
                use_fp16=False,
                max_length=self._max_length,
                devices=devices,
            )
        elif loader == "crossencoder":
            raise NotImplementedError(
                f"Reranker model {self._model_name!r} needs loader='crossencoder', "
                "which is implemented by CrossEncoderReranker, not "
                "FlagEmbeddingReranker. Use get_reranker() (it dispatches on the "
                "registry) instead of instantiating FlagEmbeddingReranker directly."
            )
        else:
            # 'manual' (layerwise MiniCPM) and anything unknown: reuse the
            # precise error from the cross-encoder path so there is exactly
            # one place documenting the blockers.
            CrossEncoderReranker(model_name=self._model_name)._ensure_model()
            raise AssertionError("unreachable: _ensure_model must raise")

        logger.info(
            "FlagEmbedding reranker loaded (%s)",
            type(self._reranker).__name__,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 10,
        score_key: str = "hybrid_score",
        pool: Optional[int] = None,
    ) -> List[dict]:
        """Rerank candidates using FlagEmbedding scores.

        Same semantics as :meth:`CrossEncoderReranker.rerank`: sort by
        ``score_key``, keep the pool (``KB_RERANK_POOL`` override or legacy
        ``min(top_k*3, len)``), score ``(query, passage)`` pairs, attach
        ``rerank_score`` as float, return top-k sorted descending. Records
        ``self.last_rerank_ms``.
        """
        t0 = time.perf_counter()
        if not candidates:
            self.last_rerank_ms = 0.0
            logger.info("rerank model=%s n=0 ms=0.0", self._model_name)
            return []

        self._ensure_model()

        if pool is None:
            pool = get_rerank_pool()

        ordered = sorted(
            candidates,
            key=lambda x: x.get(score_key, 0),
            reverse=True,
        )
        if pool and pool > 0:
            cap = min(pool, len(ordered))
        else:
            cap = min(top_k * 3, len(ordered))
        rerank_pool = ordered[:cap]

        pairs = []
        for cand in rerank_pool:
            text = cand.get("content") or cand.get("content_preview") or ""
            pairs.append([query, text])

        scores = self._reranker.compute_score(pairs, batch_size=self._batch_size)

        for cand, score in zip(rerank_pool, scores):
            cand["rerank_score"] = float(score)

        reranked = sorted(rerank_pool, key=lambda x: x["rerank_score"], reverse=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.last_rerank_ms = elapsed_ms
        logger.info(
            "rerank model=%s n=%d ms=%.1f",
            self._model_name,
            len(rerank_pool),
            elapsed_ms,
        )
        return reranked[:top_k]

    def __repr__(self) -> str:
        return f"FlagEmbeddingReranker(model={self._model_name!r}, batch_size={self._batch_size})"


def get_reranker(
    model_name: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> "CrossEncoderReranker | FlagEmbeddingReranker":
    """Factory function to create a reranker instance.

    ``model_name`` defaults to the ``KB_RERANKER_MODEL`` env var (falling
    back to the built-in default), so an explicit argument always wins.
    Dispatches on the registry loader: ``'crossencoder'`` (and unknown ids,
    which fall back to that path) → :class:`CrossEncoderReranker`;
    ``'flag'`` / ``'flag-llm'`` → :class:`FlagEmbeddingReranker`. Any other
    loader raises the registry's NotImplementedError.
    """
    name = model_name or get_reranker_model_name()
    loader = resolve_reranker_spec(name).get("loader", "crossencoder")
    if loader in ("flag", "flag-llm"):
        return FlagEmbeddingReranker(
            model_name=name,
            batch_size=batch_size,
            device=device,
        )
    return CrossEncoderReranker(
        model_name=name,
        batch_size=batch_size,
        device=device,
    )