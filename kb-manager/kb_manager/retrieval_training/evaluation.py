"""Retrieval evaluation contracts (stage 5).

Scores trained reranker / dense checkpoints with the existing IR metrics
(P/R/Hit/MRR/nDCG/MAP from ``kb_manager/evaluation/metrics.py``) and the
multi-gold benchmark runner (``kb_manager/evaluation/benchmark.py``),
diffed against ``artifacts/retrieval_training/baseline_freeze.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "RetrievalEvalConfig",
    "evaluate_checkpoint",
    "compare_against_baseline",
    "reciprocal_rank",
    "hit_indicator",
    "mean_of",
]


@dataclass
class RetrievalEvalConfig:
    """Knobs for a retrieval evaluation run."""

    top_k: int = 10
    baseline_path: str = "artifacts/retrieval_training/baseline_freeze.json"
    dataset_path: str = ""


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_path: str,
    config: RetrievalEvalConfig,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Run IR metrics for a trained checkpoint over the eval dataset.

    Args:
        checkpoint_path: Exported reranker or dense model directory.
        dataset_path: Eval dataset (queries + ``expected_chunk_ids``,
            multi-gold supported).
        config: Eval knobs (top_k, baseline path, ...).
        seed: Deterministic seed for any sampling / tie-breaking.
        provenance: Run tag recorded with the results.

    Returns:
        Dict with per-query ranks plus aggregated
        ``precision`` / ``recall`` / ``hit_rate`` / ``mrr`` /
        ``ndcg`` / ``map``.
    """
    raise NotImplementedError


def compare_against_baseline(
    current_results: dict[str, Any],
    baseline_path: str,
    config: RetrievalEvalConfig,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Diff *current_results* against the frozen baseline metrics.

    Args:
        current_results: Output of :func:`evaluate_checkpoint`.
        baseline_path: Path to ``baseline_freeze.json`` (or a metrics
            snapshot in the same schema).
        config: Eval knobs.
        seed: Deterministic seed (recorded in the comparison report).
        provenance: Run tag recorded in the comparison report.

    Returns:
        Dict with per-metric deltas and a pass/fail verdict.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Tiny pure helpers (implemented; exercised by scaffolding tests)
# ---------------------------------------------------------------------------


def reciprocal_rank(rank: int) -> float:
    """Return the reciprocal rank for a 1-based *rank* (0.0 for a miss).

    Args:
        rank: 1-based rank of the first relevant result, or -1/0 for a miss.

    Returns:
        ``1.0 / rank`` for hits, ``0.0`` for misses.
    """
    if rank <= 0:
        return 0.0
    return 1.0 / float(rank)


def hit_indicator(rank: int, k: int = 10) -> int:
    """Return 1 if *rank* is a hit within top-*k*, else 0.

    Args:
        rank: 1-based rank of the first relevant result (-1/0 = miss).
        k: Retrieval depth.

    Returns:
        1 on hit, 0 on miss.
    """
    return 1 if 0 < rank <= k else 0


def mean_of(values: list[float]) -> float:
    """Return the arithmetic mean (0.0 for an empty list).

    Args:
        values: Scores to average.

    Returns:
        Mean of *values*, or 0.0 when empty.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)
