"""Typed records for retrieval-training stages.

All records are plain :mod:`dataclasses` with JSON round-trip helpers so
later agents can persist mined negatives, splits, and run manifests without
pulling in heavy dependencies. This module must stay lightweight: no
``torch`` / ``transformers`` / ``sentence-transformers`` imports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

NegativeTier = Literal["Tier1", "Tier2", "Tier3", "Tier4"]
"""Hard-negative difficulty tier (see ``docs/retrieval_training/hard_negative_mining.md``)."""

NEGATIVE_TIERS: tuple[str, ...] = ("Tier1", "Tier2", "Tier3", "Tier4")
"""Allowed values of :data:`NegativeTier` as runtime-checkable data."""

ValidationStatus = Literal["pending", "passed", "failed"]
"""Lifecycle state of a mined negative after the validation stage."""

VALIDATION_STATUSES: tuple[str, ...] = ("pending", "passed", "failed")
"""Allowed values of :data:`ValidationStatus` as runtime-checkable data."""

__all__ = [
    "NegativeTier",
    "NEGATIVE_TIERS",
    "ValidationStatus",
    "VALIDATION_STATUSES",
    "TierRatio",
    "MinedNegative",
    "FailureRecord",
    "DatasetSplit",
    "RunManifest",
    "to_json",
    "from_json",
]


# ---------------------------------------------------------------------------
# Tier mixture config (no hard-coded ratios in stage signatures)
# ---------------------------------------------------------------------------


@dataclass
class TierRatio:
    """Target mixture over hard-negative tiers; must sum to 1.0.

    Attributes:
        tier1: Share of near-duplicate / paraphrase negatives.
        tier2: Share of BM25 lexical hard negatives.
        tier3: Share of dense semantic hard negatives.
        tier4: Share of cross-encoder (reranker) top confusers.
    """

    tier1: float = 0.25
    tier2: float = 0.25
    tier3: float = 0.25
    tier4: float = 0.25

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ValueError` if ratios are negative or do not sum to 1."""
        values = (self.tier1, self.tier2, self.tier3, self.tier4)
        if any(v < 0 for v in values):
            raise ValueError(f"TierRatio values must be >= 0, got {values!r}")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError(f"TierRatio values must sum to 1.0, got {sum(values)!r}")

    def as_dict(self) -> dict[str, float]:
        """Return the ratios as a plain dict."""
        return {"Tier1": self.tier1, "Tier2": self.tier2, "Tier3": self.tier3, "Tier4": self.tier4}

    def to_json(self) -> str:
        """Serialize this ratio config to a JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TierRatio:
        """Build a :class:`TierRatio` from a plain dict (validates)."""
        return cls(
            tier1=float(data["tier1"]),
            tier2=float(data["tier2"]),
            tier3=float(data["tier3"]),
            tier4=float(data["tier4"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> TierRatio:
        """Deserialize a :class:`TierRatio` from a JSON string (validates)."""
        return cls.from_dict(json.loads(payload))


# ---------------------------------------------------------------------------
# Stage records
# ---------------------------------------------------------------------------


@dataclass
class MinedNegative:
    """One mined hard negative linked to its query and positive chunk(s)."""

    query_id: str
    query: str
    positive_chunk_ids: list[str] = field(default_factory=list)
    negative_chunk_id: str = ""
    negative_type: NegativeTier = "Tier2"
    bm25_rank: int = -1
    dense_rank: int = -1
    rrf_rank: int = -1
    reranker_rank: int = -1
    source_document: str = ""
    validation_status: ValidationStatus = "pending"
    provenance: str = ""

    def __post_init__(self) -> None:
        if self.negative_type not in NEGATIVE_TIERS:
            raise ValueError(
                f"negative_type must be one of {NEGATIVE_TIERS!r}, got {self.negative_type!r}"
            )
        if self.validation_status not in VALIDATION_STATUSES:
            raise ValueError(
                f"validation_status must be one of {VALIDATION_STATUSES!r}, "
                f"got {self.validation_status!r}"
            )
        self.positive_chunk_ids = list(self.positive_chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this record."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize this record to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MinedNegative:
        """Build a :class:`MinedNegative` from a plain dict (validates tiers)."""
        return cls(
            query_id=str(data["query_id"]),
            query=str(data["query"]),
            positive_chunk_ids=list(data.get("positive_chunk_ids", [])),
            negative_chunk_id=str(data.get("negative_chunk_id", "")),
            negative_type=data.get("negative_type", "Tier2"),
            bm25_rank=int(data.get("bm25_rank", -1)),
            dense_rank=int(data.get("dense_rank", -1)),
            rrf_rank=int(data.get("rrf_rank", -1)),
            reranker_rank=int(data.get("reranker_rank", -1)),
            source_document=str(data.get("source_document", "")),
            validation_status=data.get("validation_status", "pending"),
            provenance=str(data.get("provenance", "")),
        )

    @classmethod
    def from_json(cls, payload: str) -> MinedNegative:
        """Deserialize a :class:`MinedNegative` from a JSON string."""
        return cls.from_dict(json.loads(payload))


@dataclass
class FailureRecord:
    """A retrieval failure: expected chunk(s) missed or ranked poorly."""

    query_id: str
    query: str
    expected_chunk_ids: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    rank: int = -1
    top_k: int = 10
    failure_type: str = "miss"
    provenance: str = ""

    def __post_init__(self) -> None:
        self.expected_chunk_ids = list(self.expected_chunk_ids)
        self.retrieved_ids = list(self.retrieved_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this record."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize this record to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureRecord:
        """Build a :class:`FailureRecord` from a plain dict."""
        return cls(
            query_id=str(data["query_id"]),
            query=str(data["query"]),
            expected_chunk_ids=list(data.get("expected_chunk_ids", [])),
            retrieved_ids=list(data.get("retrieved_ids", [])),
            rank=int(data.get("rank", -1)),
            top_k=int(data.get("top_k", 10)),
            failure_type=str(data.get("failure_type", "miss")),
            provenance=str(data.get("provenance", "")),
        )

    @classmethod
    def from_json(cls, payload: str) -> FailureRecord:
        """Deserialize a :class:`FailureRecord` from a JSON string."""
        return cls.from_dict(json.loads(payload))


@dataclass
class DatasetSplit:
    """A named train/val/test split over query ids."""

    name: str
    query_ids: list[str] = field(default_factory=list)
    provenance: str = ""

    def __post_init__(self) -> None:
        self.query_ids = list(self.query_ids)

    @property
    def size(self) -> int:
        """Number of query ids in this split."""
        return len(self.query_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this split."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize this split to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSplit:
        """Build a :class:`DatasetSplit` from a plain dict."""
        return cls(
            name=str(data["name"]),
            query_ids=list(data.get("query_ids", [])),
            provenance=str(data.get("provenance", "")),
        )

    @classmethod
    def from_json(cls, payload: str) -> DatasetSplit:
        """Deserialize a :class:`DatasetSplit` from a JSON string."""
        return cls.from_dict(json.loads(payload))


@dataclass
class RunManifest:
    """Reproducibility manifest pinned to a code revision and DB snapshot."""

    code_rev: str
    db_sha: str
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def __post_init__(self) -> None:
        self.config = dict(self.config)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of this manifest."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize this manifest to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        """Build a :class:`RunManifest` from a plain dict."""
        return cls(
            code_rev=str(data["code_rev"]),
            db_sha=str(data["db_sha"]),
            config=dict(data.get("config", {})),
            seed=int(data.get("seed", 42)),
        )

    @classmethod
    def from_json(cls, payload: str) -> RunManifest:
        """Deserialize a :class:`RunManifest` from a JSON string."""
        return cls.from_dict(json.loads(payload))


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def to_json(record: Any) -> str:
    """Serialize any retrieval-training dataclass record to a JSON string.

    Args:
        record: A :class:`TierRatio`, :class:`MinedNegative`,
            :class:`FailureRecord`, :class:`DatasetSplit`, or
            :class:`RunManifest` instance.

    Returns:
        JSON string of the record.
    """
    return json.dumps(asdict(record), ensure_ascii=False)


def from_json(cls: Any, payload: str) -> Any:
    """Deserialize *payload* into an instance of dataclass *cls*.

    Args:
        cls: One of the record classes defined in this module.
        payload: JSON string previously produced by :func:`to_json` or a
            record ``to_json()`` method.

    Returns:
        An instance of *cls*.
    """
    return cls.from_dict(json.loads(payload))
