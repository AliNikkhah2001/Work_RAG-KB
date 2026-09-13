"""Dataset-build contracts (stage 3).

Turns validated negatives + failure records into versioned train/val/test
splits of :class:`~kb_manager.retrieval_training.schemas.DatasetSplit`
with a pinned :class:`~kb_manager.retrieval_training.schemas.RunManifest`.

Output record shapes (every split row carries BOTH shapes for CLI compat):

* CE (cross-encoder triples): ``query`` / ``positive_text`` /
  ``negatives_text`` (max ``negatives_per_positive``, ordered Tier1 > Tier2
  > Tier3 > Tier4).
* Dense (bi-encoder): ``query`` / ``positive`` (= ``positive_text``) /
  ``hard`` (Tier1+Tier2) / ``medium`` (Tier3) / ``easy`` (Tier4). The
  ``TierRatio`` mixture guides sampling priority, not strict per-row counts
  (a query rarely yields all four tiers; priority order keeps the hardest
  available negatives).

Split rule (query-level, file-stratified; documented exact rule):

* Group queries by source file (record ``source_file``; the run script sets
  it from the ``query_id`` prefix, e.g. ``Individual_CRM_Questions``).
* Large files (``Individual*``, ``Cheque*``, ``Public*`` — and any
  ``unknown`` group) are divided 80/10/10 (``train_ratio``/``val_ratio``
  from config, test = remainder) by seeded shuffle of their query ids.
  Rationale: every QA workbook was ingested as a single document, so
  document-level confinement coincides with file level and cannot satisfy
  per-file ratios; the frozen rule therefore stratifies queries within
  large files while keeping each query in exactly one split (no leakage).
* Small files (name contains ``dispute`` or ``etebarito``, case-insensitive)
  go wholly to ``train``.
* Failure-class representatives: ``failure_records`` mark the fail query ids
  (class A/B/C); after the per-file split, if ``validation`` or ``test``
  holds zero fail queries while fails exist, one fail query is moved from
  the fail-richest split (``train`` preferred on ties) into the empty split
  (seeded choice among sorted candidates). Moves are single queries, never
  whole files; small files' queries never move out of ``train`` — if no
  large-file fail exists to move, the shortfall is recorded in
  ``stats["warnings"]``.

Determinism: shuffles use ``random.Random`` with ``zlib.crc32``-derived
sub-seeds (never ``hash()``); identical ``(records, config, seed)`` gives
identical splits. Malformed rows (missing/blank ``query_id`` / ``query`` /
``positive_text``, bad ``negative_type``, blank ``negative_text``) raise
``ValueError``. Records carrying an ``"error"`` key (e.g. missing chunk
content from the DB resolve step) are NOT silently skipped: they are
excluded from splits and counted in ``stats["errors"]`` with their reason.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Any

from kb_manager.retrieval_training.schemas import DatasetSplit, RunManifest, TierRatio

__all__ = [
    "DatasetConfig",
    "build_dataset",
    "split_dataset",
    "write_dataset_manifest",
    "validate_split_ratios",
    "deterministic_split",
    "tier_priority_key",
    "SMALL_FILE_MARKERS",
    "TIER_ORDER",
]

#: Tier priority for CE ordering (hardest first).
TIER_ORDER: tuple[str, ...] = ("Tier1", "Tier2", "Tier3", "Tier4")

#: Case-insensitive filename markers treated as small files (wholly train).
SMALL_FILE_MARKERS: tuple[str, ...] = ("dispute", "etebarito")


@dataclass
class DatasetConfig:
    """Knobs for dataset construction (ratios live here, not in code)."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    negatives_per_positive: int = 4
    tier_ratios: TierRatio = field(default_factory=TierRatio)


def tier_priority_key(tier: str) -> int:
    """Sort key for tier priority (Tier1 first; unknown tiers last)."""
    try:
        return TIER_ORDER.index(tier)
    except ValueError:
        return len(TIER_ORDER)


def _sub_seed(seed: int, tag: str) -> int:
    return (int(seed) & 0xFFFFFFFF) ^ zlib.crc32(tag.encode("utf-8"))


def _positive_text(rec: dict[str, Any], qid: str) -> str:
    for k in ("positive_text", "positive"):
        v = rec.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError(f"query {qid!r}: missing non-blank 'positive_text'")


def _require_grouped(rec: dict[str, Any], idx: int) -> tuple[str, str, str, str, str]:
    """Validate one flat negative dict -> (qid, query, pos, neg_text, tier)."""
    if not isinstance(rec, dict):
        raise ValueError(f"validated_negatives[{idx}]: must be a dict")
    if "error" in rec:
        raise _ErrorEntry(rec)
    qid = rec.get("query_id", "")
    qtext = rec.get("query", "")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"validated_negatives[{idx}]: missing non-blank 'query_id'")
    if not isinstance(qtext, str) or not qtext.strip():
        raise ValueError(f"query {qid!r}: missing non-blank 'query'")
    qid, qtext = qid.strip(), qtext.strip()
    pos = _positive_text(rec, qid)
    neg = rec.get("negative_text", rec.get("negative", ""))
    if not isinstance(neg, str) or not neg.strip():
        raise ValueError(f"query {qid!r}: missing non-blank 'negative_text'")
    tier = rec.get("negative_type", rec.get("tier", ""))
    if tier not in TIER_ORDER:
        raise ValueError(
            f"query {qid!r}: bad 'negative_type' {tier!r} (expected one of {list(TIER_ORDER)})"
        )
    return qid, qtext, pos, neg, tier


class _ErrorEntry(Exception):
    def __init__(self, rec: dict[str, Any]) -> None:
        super().__init__(str(rec.get("error", "unknown error")))
        self.rec = rec


def _source_file(rec: dict[str, Any]) -> str:
    for k in ("source_file", "sourceFile", "file"):
        v = rec.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def _is_small_file(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in SMALL_FILE_MARKERS)


def _fail_ids(failure_records: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for fr in failure_records:
        if not isinstance(fr, dict):
            continue
        ftype = str(fr.get("failure_type", "")).upper()
        qid = str(fr.get("query_id", "")).strip()
        if qid and (ftype in ("A", "B", "C") or "fail" in str(fr.get("failure_type", "")).lower()):
            out.add(qid)
    return out


def build_dataset(
    validated_negatives: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
    config: DatasetConfig,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Assemble the training dataset from validated negatives and failures.

    Args:
        validated_negatives: Flat per-negative dicts with ``query_id``,
            ``query``, ``positive_text``, ``negative_text``, ``negative_type``
            (Tier1..Tier4) and optional ``source_file``. Only
            ``validation_status="passed"`` rows should be passed (the run
            script filters); rows with an ``"error"`` key are tallied into
            ``stats["errors"]`` instead of raising.
        failure_records: Failure dicts (``query_id`` + ``failure_type`` A/B/C
            mark fail representatives for split forcing).
        config: Dataset knobs (split ratios, negatives-per-positive, tiers).
        seed: Deterministic seed for shuffling / splitting.
        provenance: Run tag recorded in the dataset manifest.

    Returns:
        Dict with ``splits`` (``{"train": [...], "validation": [...],
        "test": [...]}`` row dicts), ``manifest`` (``RunManifest`` dict),
        and ``stats``.

    Raises:
        ValueError: On malformed rows (missing keys, bad tiers, bad ratios).
    """
    validate_split_ratios(config.train_ratio, config.val_ratio, config.test_ratio)
    config.tier_ratios.validate()

    errors: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for idx, rec in enumerate(validated_negatives):
        try:
            qid, qtext, pos, neg, tier = _require_grouped(rec, idx)
        except _ErrorEntry as exc:
            errors.append(
                {
                    "query_id": str(exc.rec.get("query_id", "")),
                    "reason": str(exc),
                    "record": {k: v for k, v in exc.rec.items() if k != "record"},
                }
            )
            continue
        entry = grouped.get(qid)
        if entry is None:
            doc = ""
            if isinstance(rec, dict):
                for k in ("gold_document_id", "gold_doc", "gold_source_document"):
                    v = rec.get(k, "")
                    if isinstance(v, str) and v.strip():
                        doc = v.strip()
                        break
            entry = {
                "query_id": qid,
                "query": qtext,
                "positive_text": pos,
                "source_file": _source_file(rec),
                "gold_doc": doc or qid,  # fallback: every query its own group
                "negatives": [],
            }
            grouped[qid] = entry
            order.append(qid)
        entry["negatives"].append({"text": neg, "tier": tier})

    rows: list[dict[str, Any]] = []
    tier_counts = {t: 0 for t in TIER_ORDER}
    for qid in order:
        entry = grouped[qid]
        negs = sorted(entry["negatives"], key=lambda n: (tier_priority_key(n["tier"]), n["text"]))
        kept = negs[: max(0, int(config.negatives_per_positive))]
        hard = [n["text"] for n in kept if n["tier"] in ("Tier1", "Tier2")]
        medium = [n["text"] for n in kept if n["tier"] == "Tier3"]
        easy = [n["text"] for n in kept if n["tier"] == "Tier4"]
        for n in kept:
            tier_counts[n["tier"]] += 1
        rows.append(
            {
                "query_id": qid,
                "query": entry["query"],
                "positive_text": entry["positive_text"],
                "positive": entry["positive_text"],
                "negatives_text": [n["text"] for n in kept],
                "tiers": [n["tier"] for n in kept],
                "hard": hard,
                "medium": medium,
                "easy": easy,
                "source_file": entry["source_file"],
                "gold_document_id": entry["gold_doc"],
            }
        )

    # --- file-stratified query split (no leakage) ----------------------
    by_file: dict[str, list[str]] = {}
    for qid in order:
        by_file.setdefault(grouped[qid]["source_file"], []).append(qid)
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for fname in sorted(by_file):
        qids = sorted(by_file[fname])
        if _is_small_file(fname):
            splits["train"].extend(qids)
            continue
        rng = random.Random(_sub_seed(seed, f"file:{fname}"))
        pool = list(qids)
        rng.shuffle(pool)
        n = len(pool)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        splits["train"].extend(pool[:n_train])
        splits["validation"].extend(pool[n_train : n_train + n_val])
        splits["test"].extend(pool[n_train + n_val :])

    # --- force fail representatives into validation+test ----------------
    # Single queries move (donor prefers the fail-richest split, train on
    # ties; small-file queries never move out of train).
    warnings: list[str] = []
    fail_ids = _fail_ids(failure_records) & set(order)
    for target in ("validation", "test"):
        if not fail_ids:
            break
        if any(q in fail_ids for q in splits[target]):
            continue
        counts = {
            name: sorted(q for q in qids if q in fail_ids)
            for name, qids in splits.items()
            if name != target
        }
        ranked_donors = sorted(counts, key=lambda n: (-len(counts[n]), 0 if n == "train" else 1))
        moved = False
        for donor in ranked_donors:
            movable = [
                q for q in counts[donor]
                if not _is_small_file(grouped[q]["source_file"])
            ]
            if not movable:
                continue
            rng = random.Random(_sub_seed(seed, f"failmove:{target}"))
            pick = rng.choice(movable)
            splits[donor].remove(pick)
            splits[target].append(pick)
            moved = True
            break
        if not moved:
            warnings.append(f"no movable fail query for split {target!r}; fails stay in train")

    split_rows = {
        name: [r for r in rows if r["query_id"] in set(qids)]
        for name, qids in splits.items()
    }
    # stable row order: file, then query_id
    for name in split_rows:
        split_rows[name].sort(key=lambda r: (r["source_file"], r["query_id"]))
    for r in rows:
        r["split"] = next(name for name, qids in splits.items() if r["query_id"] in qids)

    total_negs = sum(len(r["negatives_text"]) for r in rows)
    stats: dict[str, Any] = {
        "queries": len(rows),
        "negatives": total_negs,
        "avg_negatives_per_query": (total_negs / len(rows)) if rows else 0.0,
        "tier_counts": tier_counts,
        "splits": {name: len(qids) for name, qids in splits.items()},
        "split_tier_counts": {
            name: {
                t: sum(1 for r in split_rows[name] for x in r["tiers"] if x == t)
                for t in TIER_ORDER
            }
            for name in split_rows
        },
        "fail_ids": sorted(fail_ids),
        "errors": errors,
        "warnings": warnings,
        "seed": seed,
        "provenance": provenance,
    }
    manifest = RunManifest(
        code_rev=provenance or "unknown",
        db_sha="",
        config={
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "negatives_per_positive": config.negatives_per_positive,
            "tier_ratios": config.tier_ratios.as_dict(),
            "seed": seed,
            "provenance": provenance,
        },
        seed=seed,
    )
    manifest_dict = manifest.to_dict()
    by_id = {r["query_id"]: r for r in rows}
    split_objs = {
        name: DatasetSplit(name=name, query_ids=sorted(qids), provenance=provenance)
        for name, qids in splits.items()
    }
    return {
        "splits": {name: [by_id[q] for q in sorted(qids)] for name, qids in splits.items()},
        "split_objects": split_objs,
        "manifest": manifest_dict,
        "stats": stats,
    }


def split_dataset(
    query_ids: list[str],
    config: DatasetConfig,
    seed: int,
    provenance: str,
) -> dict[str, DatasetSplit]:
    """Split *query_ids* into ``train`` / ``val`` / ``test`` splits.

    Plain ratio split (no file grouping; that lives in :func:`build_dataset`).

    Args:
        query_ids: Query ids to partition (no leakage across splits).
        config: Dataset knobs carrying the split ratios.
        seed: Deterministic shuffle seed.
        provenance: Run tag stamped onto each split.

    Returns:
        Mapping ``{"train": ..., "val": ..., "test": ...}``.
    """
    validate_split_ratios(config.train_ratio, config.val_ratio, config.test_ratio)
    pool = list(query_ids)
    rng = random.Random(seed)
    rng.shuffle(pool)
    n = len(pool)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)
    return {
        "train": DatasetSplit(name="train", query_ids=pool[:n_train], provenance=provenance),
        "val": DatasetSplit(name="val", query_ids=pool[n_train : n_train + n_val], provenance=provenance),
        "test": DatasetSplit(name="test", query_ids=pool[n_train + n_val :], provenance=provenance),
    }


def write_dataset_manifest(
    splits: dict[str, DatasetSplit],
    config: DatasetConfig,
    code_rev: str,
    db_sha: str,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Persist the dataset manifest (splits + ``RunManifest``) to disk.

    Pure (no file IO): builds the manifest dict; the run script writes the
    sidecar JSON. Kept signature-compatible with the CLI contract.

    Args:
        splits: Named splits from :func:`split_dataset`.
        config: Dataset knobs snapshotted into the manifest.
        code_rev: Git revision of the mining code.
        db_sha: SHA256 of the source DB snapshot.
        seed: Deterministic seed of the build.
        provenance: Run tag recorded in the manifest.

    Returns:
        The manifest dict that was written.
    """
    manifest = RunManifest(
        code_rev=code_rev,
        db_sha=db_sha,
        config={
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "negatives_per_positive": config.negatives_per_positive,
            "tier_ratios": config.tier_ratios.as_dict(),
            "seed": seed,
            "provenance": provenance,
            "splits": {name: {"size": sp.size, "query_ids": list(sp.query_ids)} for name, sp in splits.items()},
        },
        seed=seed,
    )
    return manifest.to_dict()


# ---------------------------------------------------------------------------
# Tiny pure helpers (implemented; exercised by scaffolding tests)
# ---------------------------------------------------------------------------


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate split ratios; raise :class:`ValueError` if bad.

    Args:
        train_ratio: Train share (>= 0).
        val_ratio: Validation share (>= 0).
        test_ratio: Test share (>= 0).

    Raises:
        ValueError: If any share is negative or they do not sum to 1.0.
    """
    values = (train_ratio, val_ratio, test_ratio)
    if any(v < 0 for v in values):
        raise ValueError(f"Split ratios must be >= 0, got {values!r}")
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(values)!r}")


def deterministic_split(
    query_ids: list[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    provenance: str = "",
) -> dict[str, DatasetSplit]:
    """Deterministically partition *query_ids* into train/val/test splits.

    Args:
        query_ids: Ids to partition.
        seed: Shuffle seed.
        train_ratio: Train share.
        val_ratio: Validation share (test gets the remainder).
        provenance: Tag stamped onto each split.

    Returns:
        Mapping ``{"train": ..., "val": ..., "test": ...}`` with disjoint,
        covering splits.
    """
    validate_split_ratios(train_ratio, val_ratio, round(1.0 - train_ratio - val_ratio, 9))
    pool = list(query_ids)
    rng = random.Random(seed)
    rng.shuffle(pool)
    n = len(pool)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": DatasetSplit(name="train", query_ids=pool[:n_train], provenance=provenance),
        "val": DatasetSplit(name="val", query_ids=pool[n_train : n_train + n_val], provenance=provenance),
        "test": DatasetSplit(name="test", query_ids=pool[n_train + n_val :], provenance=provenance),
    }
