"""Hard-negative mining contracts (stage 1).

Reads the frozen retrieval pipeline (BM25 + dense + RRF + cross-encoder
reranker, see ``kb_manager/web/routes/search.py::search_knowledge_base``)
and emits :class:`~kb_manager.retrieval_training.schemas.MinedNegative`
records across Tier1..Tier4. Heavy pipeline work (running
``search_knowledge_base`` per query, DB lookups) belongs to the run scripts
under ``scripts/retrieval_training/``; this module implements the
deterministic, pipeline-free selection contracts those scripts (and the
tests) build on.

Tier semantics (task chain, NOT the paraphrase/BM25/dense/reranker split
described in the original scaffold docstring):

* Tier1 ``reranker_mistake``: gold merged (RRF) rank <= 20 and an incorrect
  chunk ranked above the gold in the final (reranked) list. Up to 3
  above-gold chunks.
* Tier2 ``hard_rrf``: non-gold chunks in merged top-10. Up to 4.
* Tier3 ``medium``: same-document (``source_document`` / ``gold_document_id``
  match), non-gold, merged rank > 20 or outside the pool. Up to 3.
* Tier4 ``easy``: random other-document chunks, seeded sample. Up to 2.

Pool limit (public API): ``search_knowledge_base(q, top_k=100)`` truncates
every exposed leg (``bm25_results`` / ``dense_results`` /
``merged_candidates`` / ``final_results``) to ``top_k`` rows, so the minable
RRF pool is the ``merged_candidates`` slice of 100. Ranks deeper than 100
are unreachable via the public API; Tier3 "outside pool" candidates are
therefore sourced from DB same-doc lookups by the run script, not from the
search response.

Determinism: NEVER use ``hash()`` here (salted per process; the beam-4
query-expansion shuffle in ``query_expansion.py`` already jitters leg ranks
across processes, so labels must be pinned with ``PYTHONHASHSEED=0`` at the
interpreter level). Per-query sampling derives a stable sub-seed from
``zlib.crc32`` (see :func:`per_query_seed`).
"""

from __future__ import annotations

import random
import re
import zlib
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
    "per_query_seed",
    "TIER_CAPS",
    "TIER_PRIORITY",
]

#: Per-tier per-query caps (Tier1..Tier4). ``mine_negatives`` additionally
#: caps the union at ``MiningConfig.negatives_per_query`` preserving
#: ``TIER_PRIORITY`` order (run scripts set ``negatives_per_query=12`` to
#: keep the full pool; the default 4 matches the CE training width).
TIER_CAPS: dict[str, int] = {"Tier1": 3, "Tier2": 4, "Tier3": 3, "Tier4": 2}

#: Priority used when trimming the union to ``negatives_per_query`` and
#: when resolving a chunk claimed by several tiers (earlier wins).
TIER_PRIORITY: tuple[str, ...] = ("Tier1", "Tier2", "Tier3", "Tier4")

#: Candidate-dict keys that may carry extractable question text for the
#: normalizer dedup (first hit wins).
_QUESTION_KEYS: tuple[str, ...] = ("question_text", "question", "content", "content_preview")

#: Candidate-dict keys that may carry the chunk id.
_ID_KEYS: tuple[str, ...] = ("negative_chunk_id", "chunk_id", "id")


def normalize_question_text(q: str) -> str:
    """Normalize question text with the chunker normalizer when available.

    Reuses ``SemanticChunker._normalize_question`` (Arabic yeh/kaf unify,
    ZWNJ -> space, question marks stripped, whitespace collapsed); falls
    back to an identical inline implementation if the chunker cannot be
    imported (keeps this module import-light for tests).
    """
    try:
        from kb_manager.chunker.semantic import SemanticChunker

        return SemanticChunker._normalize_question(q)
    except Exception:
        s = (
            q.strip()
            .replace("\u064a", "\u06cc")  # Arabic yeh -> Persian yeh
            .replace("\u0643", "\u06a9")  # Arabic kaf -> Persian kaf
            .replace("\u0671", "\u0627")  # alif wasla -> alef
            .replace("\u200c", " ")  # ZWNJ -> space
            .replace("؟", "")
            .replace("?", "")
        )
        return re.sub(r"\s+", " ", s).strip()


def per_query_seed(seed: int, query_id: str) -> int:
    """Derive a stable per-query sub-seed (crc32, NOT ``hash()``).

    ``hash(str)`` is salted per process (PYTHONHASHSEED) and would jitter
    Tier4 sampling across processes; crc32 is stable, so labels are
    reproducible under any hash seed (still pin ``PYTHONHASHSEED=0`` for
    the retrieval legs themselves).
    """
    return (int(seed) & 0xFFFFFFFF) ^ zlib.crc32(f"{seed}:{query_id}".encode("utf-8"))


@dataclass
class MiningConfig:
    """Knobs for a mining run (no hard-coded ratios or pool sizes here)."""

    tier_ratios: TierRatio = field(default_factory=TierRatio)
    candidate_pool_size: int = 100
    negatives_per_query: int = 4
    rrf_k: int = 60
    top_k: int = 10


# ---------------------------------------------------------------------------
# Input validation / candidate helpers
# ---------------------------------------------------------------------------


def _require_query(query: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Validate a query dict; return ``(query_id, query, gold_ids)``.

    Raises:
        ValueError: If ``query_id``/``query`` are missing/blank or
            ``positive_chunk_ids`` is missing/empty/blank.
    """
    if not isinstance(query, dict):
        raise ValueError(f"query must be a dict, got {type(query).__name__}")
    qid = query.get("query_id", "")
    qtext = query.get("query", "")
    golds = query.get("positive_chunk_ids", [])
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"query missing non-blank 'query_id': {query!r}")
    if not isinstance(qtext, str) or not qtext.strip():
        raise ValueError(f"query {qid!r} missing non-blank 'query'")
    if not isinstance(golds, list) or not golds or not all(
        isinstance(g, str) and g.strip() for g in golds
    ):
        raise ValueError(
            f"query {qid!r} missing non-empty 'positive_chunk_ids' "
            "(list of non-blank chunk-id strings)"
        )
    return qid.strip(), qtext.strip(), [g.strip() for g in golds]


def _candidate_id(cand: dict[str, Any]) -> str:
    for k in _ID_KEYS:
        v = cand.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _candidate_question(cand: dict[str, Any]) -> str:
    for k in _QUESTION_KEYS:
        v = cand.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _candidate_doc(cand: dict[str, Any]) -> str:
    v = cand.get("source_document", cand.get("doc_id", ""))
    return v.strip() if isinstance(v, str) else ""


def _rank(cand: dict[str, Any], key: str) -> int:
    try:
        return int(cand.get(key, -1))
    except (TypeError, ValueError):
        return -1


def _gold_doc_of(query: dict[str, Any]) -> str:
    for k in ("gold_document_id", "gold_source_document", "source_document", "doc_id"):
        v = query.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _dedup_candidates(
    candidates: list[dict[str, Any]], gold_ids: set[str]
) -> list[dict[str, Any]]:
    """Remove golds, drop id-less rows, dedup by chunk id and by normalized
    question text (first occurrence = best rank wins; callers pre-sort)."""
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    out: list[dict[str, Any]] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        cid = _candidate_id(cand)
        if not cid or cid in gold_ids or cid in seen_ids:
            continue
        qtext = _candidate_question(cand)
        if qtext:
            norm = normalize_question_text(qtext)
            if norm and norm in seen_questions:
                continue
            if norm:
                seen_questions.add(norm)
        seen_ids.add(cid)
        out.append(cand)
    return out


def _build_record(
    query_id: str,
    query_text: str,
    gold_ids: list[str],
    cand: dict[str, Any],
    tier: str,
    provenance: str,
    seed: int,
) -> MinedNegative:
    prov = (
        f"{provenance}|mine:tier={tier}:seed={seed}"
        f":bm25={_rank(cand, 'bm25_rank')}:dense={_rank(cand, 'dense_rank')}"
        f":rrf={_rank(cand, 'rrf_rank')}:rerank={_rank(cand, 'reranker_rank')}"
    )
    return MinedNegative(
        query_id=query_id,
        query=query_text,
        positive_chunk_ids=list(gold_ids),
        negative_chunk_id=_candidate_id(cand),
        negative_type=tier,  # type: ignore[arg-type]
        bm25_rank=_rank(cand, "bm25_rank"),
        dense_rank=_rank(cand, "dense_rank"),
        rrf_rank=_rank(cand, "rrf_rank"),
        reranker_rank=_rank(cand, "reranker_rank"),
        source_document=_candidate_doc(cand),
        validation_status="pending",
        provenance=prov,
    )


def _pool_of(query: dict[str, Any]) -> list[dict[str, Any]]:
    pool = query.get("candidates", query.get("pool", []))
    return list(pool) if isinstance(pool, list) else []


# ---------------------------------------------------------------------------
# Tier miners
# ---------------------------------------------------------------------------


def mine_tier1_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier1 (reranker_mistake) negatives.

    Eligible when the gold merged rank (``gold_merged_rank`` / ``gold_rrf_rank``,
    -1 = absent) is in [0, 20]: non-gold candidates ranked above the gold in
    the final list (``reranker_rank`` < ``gold_final_rank``; when the gold is
    absent from the final list all reranked non-gold candidates count as
    above-gold), ordered by ``reranker_rank``, up to 3 per query. Queries
    without gold-rank keys fall back to plain top-reranked order.
    """
    out: list[MinedNegative] = []
    for query in queries:
        qid, qtext, golds = _require_query(query)
        gold_set = set(golds)
        gold_merged = query.get("gold_merged_rank", query.get("gold_rrf_rank", None))
        try:
            gold_merged_i = int(gold_merged) if gold_merged is not None else None
        except (TypeError, ValueError):
            gold_merged_i = None
        if gold_merged_i is not None and not (0 <= gold_merged_i <= 20):
            continue
        try:
            gold_final = int(query.get("gold_final_rank", query.get("gold_reranker_rank", -1)))
        except (TypeError, ValueError):
            gold_final = -1
        ranked = sorted(
            (c for c in _pool_of(query) if isinstance(c, dict) and _rank(c, "reranker_rank") >= 0),
            key=lambda c: _rank(c, "reranker_rank"),
        )
        if gold_final >= 0:
            ranked = [c for c in ranked if _rank(c, "reranker_rank") < gold_final]
        pool = _dedup_candidates(ranked, gold_set)[: TIER_CAPS["Tier1"]]
        out.extend(_build_record(qid, qtext, golds, c, "Tier1", provenance, seed) for c in pool)
    return out


def mine_tier2_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier2 (hard_rrf) negatives: non-gold merged top-10 ordered by
    ``rrf_rank``, up to 4 per query."""
    out: list[MinedNegative] = []
    for query in queries:
        qid, qtext, golds = _require_query(query)
        ranked = sorted(
            (
                c
                for c in _pool_of(query)
                if isinstance(c, dict) and 0 <= _rank(c, "rrf_rank") <= 9
            ),
            key=lambda c: _rank(c, "rrf_rank"),
        )
        pool = _dedup_candidates(ranked, set(golds))[: TIER_CAPS["Tier2"]]
        out.extend(_build_record(qid, qtext, golds, c, "Tier2", provenance, seed) for c in pool)
    return out


def mine_tier3_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier3 (medium, same-document) negatives: non-gold candidates whose
    ``source_document`` matches the query's gold document
    (``gold_document_id`` / ``gold_source_document``) and whose merged rank is
    > 20 or absent (outside the pool, e.g. DB same-doc fill), up to 3 per
    query ordered with in-pool ranks first."""
    out: list[MinedNegative] = []
    for query in queries:
        qid, qtext, golds = _require_query(query)
        gdoc = _gold_doc_of(query)
        cands = [
            c
            for c in _pool_of(query)
            if isinstance(c, dict)
            and (not gdoc or _candidate_doc(c) == gdoc)
            and (_rank(c, "rrf_rank") < 0 or _rank(c, "rrf_rank") > 20)
        ]
        cands.sort(key=lambda c: (_rank(c, "rrf_rank") < 0, _rank(c, "rrf_rank")))
        pool = _dedup_candidates(cands, set(golds))[: TIER_CAPS["Tier3"]]
        out.extend(_build_record(qid, qtext, golds, c, "Tier3", provenance, seed) for c in pool)
    return out


def mine_tier4_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier4 (easy, other-document) negatives: non-gold candidates from a
    different document than the gold, seeded random sample up to 2 per query
    (stable ``crc32`` sub-seed; see :func:`per_query_seed`)."""
    out: list[MinedNegative] = []
    for query in queries:
        qid, qtext, golds = _require_query(query)
        gdoc = _gold_doc_of(query)
        cands = _dedup_candidates(
            [c for c in _pool_of(query) if isinstance(c, dict)], set(golds)
        )
        if gdoc:
            others = [c for c in cands if _candidate_doc(c) and _candidate_doc(c) != gdoc]
            # Candidates without doc info cannot be proven other-doc; use them
            # only when no proven other-doc candidates exist.
            pool_in = others if others else [c for c in cands if not _candidate_doc(c)]
        else:
            pool_in = cands
        ids = [_candidate_id(c) for c in pool_in]
        picked = set(deterministic_sample(ids, TIER_CAPS["Tier4"], per_query_seed(seed, qid)))
        pool = [c for c in pool_in if _candidate_id(c) in picked]
        pool.sort(key=_candidate_id)  # stable emission order
        out.extend(_build_record(qid, qtext, golds, c, "Tier4", provenance, seed) for c in pool)
    return out


def mine_negatives(
    queries: list[dict[str, Any]],
    config: MiningConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Mine Tier1..Tier4 hard negatives for *queries*.

    Runs each tier miner over the per-query ``candidates`` pools, merges with
    cross-tier dedup (a chunk claimed by several tiers keeps its highest
    priority tier), then trims each query's union to
    ``config.negatives_per_query`` preserving ``TIER_PRIORITY`` order.

    Args:
        queries: Query dicts with at least ``query_id``, ``query``, and
            ``positive_chunk_ids`` keys plus an optional ``candidates`` pool
            (see tier miners for rank/doc keys).
        config: Mining knobs including the :class:`TierRatio` mixture.
        seed: Deterministic sampling seed for reproducibility.
        provenance: Free-text run tag (code rev + db sha + config snapshot)
            stamped onto every emitted record.

    Returns:
        List of :class:`MinedNegative` records (``validation_status`` left
        as ``"pending"`` for the validation stage).

    Raises:
        ValueError: On malformed query rows (missing/blank ``query_id`` /
            ``query`` / ``positive_chunk_ids``).
    """
    validate_tier_ratios(config.tier_ratios)
    per_tier = [
        mine_tier1_negatives(queries, config, seed, provenance),
        mine_tier2_negatives(queries, config, seed, provenance),
        mine_tier3_negatives(queries, config, seed, provenance),
        mine_tier4_negatives(queries, config, seed, provenance),
    ]
    by_query: dict[str, list[MinedNegative]] = {}
    claimed: set[tuple[str, str]] = set()
    for tier_records in per_tier:  # priority order Tier1..Tier4
        for rec in tier_records:
            key = (rec.query_id, rec.negative_chunk_id)
            if key in claimed:
                continue
            claimed.add(key)
            by_query.setdefault(rec.query_id, []).append(rec)
    out: list[MinedNegative] = []
    for qid in [q.get("query_id", "").strip() for q in queries if isinstance(q, dict)]:
        recs = by_query.get(qid, [])
        out.extend(recs[: max(0, int(config.negatives_per_query))])
    return out


def collect_failure_records(
    benchmark_queries: list[dict[str, Any]],
    top_k: int,
    seed: int,
    provenance: str,
) -> list[dict[str, Any]]:
    """Collect retrieval failures feeding the mining pool.

    Converts benchmark items (``query`` / ``expected_chunk_ids``, multi-gold
    supported) into failure dicts convertible to
    :class:`~kb_manager.retrieval_training.schemas.FailureRecord`. Rank
    reproduction needs the live pipeline, so ``retrieved_ids`` is empty and
    ``rank`` is -1; the run scripts fill them from
    ``search_knowledge_base`` output.

    Args:
        benchmark_queries: Benchmark items with ``query`` /
            ``expected_chunk_ids`` keys (multi-gold supported).
        top_k: Retrieval depth used when reproducing failures.
        seed: Deterministic seed for any sampling.
        provenance: Run tag stamped onto emitted records.

    Returns:
        Failure dicts convertible to
        :class:`~kb_manager.retrieval_training.schemas.FailureRecord`.

    Raises:
        ValueError: On malformed rows (missing/blank ``query`` or empty
            ``expected_chunk_ids``); nothing is silently skipped.
    """
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(benchmark_queries):
        if not isinstance(item, dict):
            raise ValueError(f"benchmark row {idx}: must be a dict, got {type(item).__name__}")
        qtext = item.get("query", "")
        expected = item.get("expected_chunk_ids", [])
        if not isinstance(qtext, str) or not qtext.strip():
            raise ValueError(f"benchmark row {idx}: missing non-blank 'query'")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(e, str) and e.strip() for e in expected)
        ):
            raise ValueError(
                f"benchmark row {idx}: missing non-empty 'expected_chunk_ids' "
                "(list of non-blank chunk-id strings)"
            )
        qid = item.get("query_id", f"q{idx}")
        out.append(
            {
                "query_id": qid.strip() if isinstance(qid, str) and qid.strip() else f"q{idx}",
                "query": qtext.strip(),
                "expected_chunk_ids": [e.strip() for e in expected],
                "retrieved_ids": list(item.get("retrieved_ids", [])),
                "rank": int(item.get("rank", -1)),
                "top_k": top_k,
                "failure_type": str(item.get("failure_type", "miss")),
                "provenance": f"{provenance}|collect:seed={seed}:row={idx}",
            }
        )
    return out


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
