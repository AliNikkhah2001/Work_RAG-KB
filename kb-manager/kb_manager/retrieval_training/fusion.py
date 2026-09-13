"""Score-fusion ablation contracts (stage 6).

Covers RRF (``k=60``) and reranker/RRF interpolation (``fusion_alpha``)
over the BM25 + dense (+ HyDE) legs, including the merged-top3 / BM25-top3
pin-guard semantics from ``kb_manager/web/routes/search.py``. Later agents
own the implementations; this module only fixes the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "FusionConfig",
    "fuse_rank_lists",
    "interpolate_rerank_rrf",
    "apply_pin_guard",
]


@dataclass
class FusionConfig:
    """Knobs for a fusion ablation."""

    rrf_k: int = 60
    fusion_alpha: float = 0.7
    pin_guard: bool = True
    top_k: int = 10


def fuse_rank_lists(
    ranked_lists: list[list[tuple[str, float]]],
    config: FusionConfig,
    seed: int,
    provenance: str,
) -> list[tuple[str, float]]:
    """Fuse per-leg ranked lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: One ``[(chunk_id, score)]`` list per retrieval leg
            (BM25, dense, optional HyDE), best-first.
        config: Fusion knobs (``rrf_k``).
        seed: Deterministic seed for tie-breaking.
        provenance: Run tag recorded with the fusion output.

    Returns:
        Fused ``[(chunk_id, rrf_score)]`` list, best-first.
    """
    raise NotImplementedError


def interpolate_rerank_rrf(
    rerank_scores: dict[str, float],
    rrf_scores: dict[str, float],
    config: FusionConfig,
    seed: int,
    provenance: str,
) -> list[tuple[str, float]]:
    """Interpolate min-max normalized rerank and RRF scores.

    ``final = alpha * norm(rerank) + (1 - alpha) * norm(rrf)`` over the
    rerank input set (ordering only; raw fields untouched).

    Args:
        rerank_scores: Cross-encoder scores by chunk id.
        rrf_scores: RRF hybrid scores by chunk id.
        config: Fusion knobs (``fusion_alpha``).
        seed: Deterministic seed for tie-breaking.
        provenance: Run tag recorded with the fusion output.

    Returns:
        ``[(chunk_id, fused_score)]`` list, best-first.
    """
    raise NotImplementedError


def apply_pin_guard(
    ranked_ids: list[str],
    pinned_ids: list[str],
    config: FusionConfig,
    seed: int,
    provenance: str,
) -> list[str]:
    """Re-insert pinned ids (merged-top3 + BM25-top3) above rank 10.

    Args:
        ranked_ids: Fused ranking (chunk ids, best-first).
        pinned_ids: Ids that may not fall below final rank 10.
        config: Fusion knobs (``pin_guard`` toggle).
        seed: Deterministic seed (recorded for reproducibility).
        provenance: Run tag recorded with the guard output.

    Returns:
        Adjusted ranking with violators moved to rank <= 10.
    """
    raise NotImplementedError
