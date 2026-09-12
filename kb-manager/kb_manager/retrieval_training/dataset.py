"""Dataset-build contracts (stage 3).

Turns validated negatives + failure records into versioned train/val/test
splits of :class:`~kb_manager.retrieval_training.schemas.DatasetSplit`
with a pinned :class:`~kb_manager.retrieval_training.schemas.RunManifest`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from kb_manager.retrieval_training.schemas import DatasetSplit, TierRatio

__all__ = [
    "DatasetConfig",
    "build_dataset",
    "split_dataset",
    "write_dataset_manifest",
    "validate_split_ratios",
    "deterministic_split",
]


@dataclass
class DatasetConfig:
    """Knobs for dataset construction (ratios live here, not in code)."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    negatives_per_positive: int = 4
    tier_ratios: TierRatio = field(default_factory=TierRatio)


def build_dataset(
    validated_negatives: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
    config: DatasetConfig,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Assemble the training dataset from validated negatives and failures.

    Args:
        validated_negatives: Records with ``validation_status="passed"``.
        failure_records: Failure dicts convertible to ``FailureRecord``.
        config: Dataset knobs (split ratios, negatives-per-positive, tiers).
        seed: Deterministic seed for shuffling / splitting.
        provenance: Run tag recorded in the dataset manifest.

    Returns:
        Dict with ``splits`` (list of ``DatasetSplit``), ``manifest``
        (``RunManifest``), and ``stats``.
    """
    raise NotImplementedError


def split_dataset(
    query_ids: list[str],
    config: DatasetConfig,
    seed: int,
    provenance: str,
) -> dict[str, DatasetSplit]:
    """Split *query_ids* into ``train`` / ``val`` / ``test`` splits.

    Args:
        query_ids: Query ids to partition (no leakage across splits).
        config: Dataset knobs carrying the split ratios.
        seed: Deterministic shuffle seed.
        provenance: Run tag stamped onto each split.

    Returns:
        Mapping ``{"train": ..., "val": ..., "test": ...}``.
    """
    raise NotImplementedError


def write_dataset_manifest(
    splits: dict[str, DatasetSplit],
    config: DatasetConfig,
    code_rev: str,
    db_sha: str,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Persist the dataset manifest (splits + ``RunManifest``) to disk.

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
    raise NotImplementedError


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
