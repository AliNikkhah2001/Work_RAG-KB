"""Cross-encoder reranker for precision refinement in retrieval pipeline.

Uses a multilingual cross-encoder (mDeBERTa-v3-base-xsmall) to re-score
top candidates from the hybrid BM25+Dense retrieval stage.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
_MAX_LENGTH = 512


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

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load the cross-encoder model and tokenizer if not already loaded."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for cross-encoder reranking. "
                "Install with: pip install transformers torch"
            ) from exc

        logger.info("Loading cross-encoder model %s …", self._model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16 if self._device != "cpu" else torch.float32,
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
    ) -> List[dict]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: The search query.
            candidates: List of candidate dicts with at least 'content' or 'content_preview'.
            top_k: Number of top candidates to return after reranking.
            score_key: Key in candidate dict to use for initial ranking (before rerank).

        Returns:
            Top-k candidates sorted by cross-encoder score (descending),
            with 'rerank_score' added to each dict.
        """
        if not candidates:
            return []

        self._ensure_model()

        # Sort by initial score and take top candidates for reranking
        # (cross-encoder is expensive, so we only rerank a subset)
        rerank_pool = sorted(
            candidates,
            key=lambda x: x.get(score_key, 0),
            reverse=True,
        )[: min(top_k * 3, len(candidates))]

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


def get_reranker(
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> CrossEncoderReranker:
    """Factory function to create a reranker instance."""
    return CrossEncoderReranker(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )