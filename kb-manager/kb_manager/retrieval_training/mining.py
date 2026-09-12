"""Hard-negative mining contracts (stage 1).

Reads the frozen retrieval pipeline (BM25 + dense + RRF + cross-encoder
reranker, see ``kb_manager/web/routes/search.py::search_knowledge_base``)
and emits :class:`~kb_manager.retrieval_training.schemas.MinedNegative`
records across Tier1..Tier4. Heavy pipeline work is left to later agents;
this module defines the contracts plus tiny deterministic pure helpers used
by the scaffolding tests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from kb_manager.retrieval_training.schemas import MinedNegative, TierRatio

__all__ = [
    "MiningConfig",
    "mine_negatives",
    "mine_tier1_negatives",
    "mine_tier2_negatives",
    "mine_tier3_negatives",
    "mine_tier4_negatives",
    "collect_failure_records",
    "validate_tier_ratios",
    "deterministic_sample",
    "rank_to_tier",
]


@dataclass
class MiningConfig:
    """Knobs for a mining run (no hard-coded ratios or pool sizes here)."""

    tier_ratios: TierRatio = field(default_factory=TierRatio)
    candidate_pool_size: int = 100
    negatives_per_query: int = 4
    rrf_k: int = 60
    top_k: int = 10


def mine_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier1..Tier4 hard negatives for *queries*.

    Args:
        queries: Query dicts with at least ``query_id``, ``query``, and
            ``positive_chunk_ids`` keys.
        config: Mining knobs including the :class:`TierRatio` mixture.
        seed: Deterministic sampling seed for reproducibility.
        provenance: Free-text run tag (code rev + db sha + config snapshot)
            stamped onto every emitted record.

    Returns:
        List of :class:`MinedNegative` records (``validation_status`` left
        as ``"pending"`` for the validation stage).
    """
    raise NotImplementedError


def mine_tier1_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier1 (near-duplicate / paraphrase) negatives.

    Args:
        queries: Query dicts (see :func:`mine_negatives`).
        config: Mining knobs.
        seed: Deterministic sampling seed.
        provenance: Run tag stamped onto emitted records.

    Returns:
        Tier1 :class:`MinedNegative` records.
    """
    raise NotImplementedError


def mine_tier2_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier2 (BM25 lexical hard) negatives.

    Args:
        queries: Query dicts (see :func:`mine_negatives`).
        config: Mining knobs.
        seed: Deterministic sampling seed.
        provenance: Run tag stamped onto emitted records.

    Returns:
        Tier2 :class:`MinedNegative` records.
    """
    raise NotImplementedError


def mine_tier3_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier3 (dense semantic hard) negatives.

    Args:
        queries: Query dicts (see :func:`mine_negatives`).
        config: Mining knobs.
        seed: Deterministic sampling seed.
        provenance: Run tag stamped onto emitted records.

    Returns:
        Tier3 :class:`MinedNegative` records.
    """
    raise NotImplementedError


def mine_tier4_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier4 (cross-encoder top-confuser) negatives.

    Args:
        queries: Query dicts (see :func:`mine_negatives`).
        config: Mining knobs.
        seed: Deterministic sampling seed.
        provenance: Run tag stamped onto emitted records.

    Returns:
        Tier4 :class:`MinedNegative` records.
    """
    raise NotImplementedError


def collect_failure_records(
    benchmark_queries: list[dict[str, Any]],
    top_k: int,
    seed: int,
    provenance: str,
) -> list[dict[str, Any]]:
    """Collect retrieval failures feeding the mining pool.

    Args:
        benchmark_queries: Benchmark items with ``query`` /
            ``expected_chunk_ids`` keys (multi-gold supported).
        top_k: Retrieval depth used when reproducing failures.
        seed: Deterministic seed for any sampling.
        provenance: Run tag stamped onto emitted records.

    Returns:
        Failure dicts convertible to
        :class:`~kb_manager.retrieval_training.schemas.FailureRecord`.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Tiny pure helpers (implemented; exercised by scaffolding tests)
# ---------------------------------------------------------------------------


def validate_tier_ratios(ratios: TierRatio) -> None:
    """Validate a :class:`TierRatio` mixture; raise :class:`ValueError` if bad.

    Args:
        ratios: Mixture to check (shares must be >= 0 and sum to 1.0).
    """
    ratios.validate()


def deterministic_sample(candidate_ids: list[str], k: int, seed: int) -> list[str]:
    """Sample up to *k* ids deterministically under *seed* (no replacement).

    Args:
        candidate_ids: Pool to sample from (order-independent).
        k: Number of ids to draw.
        seed: Seed for :class:`random.Random`.

    Returns:
        Sorted-input sample of at most *k* ids; deterministic for a given
        pool and seed.
    """
    pool = sorted(candidate_ids)
    if k >= len(pool):
        return list(pool)
    rng = random.Random(seed)
    return rng.sample(pool, k)


def rank_to_tier(bm25_rank: int, dense_rank: int, reranker_rank: int) -> str:
    """Map per-leg ranks to the most specific applicable tier label.

    Tier1 is assigned by the caller (needs paraphrase detection, not ranks),
    so this helper only distinguishes Tier2..Tier4:

    * reranker surfaced it (``reranker_rank >= 0``) -> ``"Tier4"``.
    * else dense surfaced it (``dense_rank >= 0``) -> ``"Tier3"``.
    * else -> ``"Tier2"`` (BM25-leg hard negative).

    Args:
        bm25_rank: 0-based BM25 rank, or -1 if absent.
        dense_rank: 0-based dense rank, or -1 if absent.
        reranker_rank: 0-based reranker rank, or -1 if absent.

    Returns:
        One of ``"Tier2"``, ``"Tier3"``, ``"Tier4"``.
    """
    if reranker_rank >= 0:
        return "Tier4"
    if dense_rank >= 0:
        return "Tier3"
    return "Tier2"
