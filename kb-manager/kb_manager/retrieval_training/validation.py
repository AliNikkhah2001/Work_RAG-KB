"""Validation contracts (stage 2).

Drops false negatives (a "negative" that is actually a gold positive) and
incomplete QA rows, then marks
:class:`~kb_manager.retrieval_training.schemas.MinedNegative.validation_status`.
Reuses the dedup normalizer semantics from
``kb_manager/chunker/semantic.py::_normalize_question`` (Arabic yeh/kaf,
ZWNJ to space, strip question marks, collapse whitespace).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kb_manager.retrieval_training.schemas import MinedNegative

__all__ = [
    "ValidationConfig",
    "validate_negatives",
    "filter_false_negatives",
    "filter_incomplete_qa",
    "is_false_negative",
    "normalize_validation_status",
    "is_complete_qa_record",
]


@dataclass
class ValidationConfig:
    """Knobs for the validation stage."""

    drop_false_negatives: bool = True
    drop_incomplete_qa: bool = True
    dedup_via_normalizer: bool = True


def validate_negatives(
    candidates: list[MinedNegative],
    config: ValidationConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Run the full validation pipeline over mined *candidates*.

    Args:
        candidates: Raw mined negatives (``validation_status="pending"``).
        config: Validation knobs.
        seed: Deterministic seed for any sampling / tie-breaking.
        provenance: Run tag appended to each record's provenance chain.

    Returns:
        Validated records with ``validation_status`` set to ``"passed"``
        or ``"failed"``.
    """
    raise NotImplementedError


def filter_false_negatives(
    candidates: list[MinedNegative],
    config: ValidationConfig,
    seed: int,
    provenance: str,
) -> list[MinedNegative]:
    """Drop candidates whose negative id is actually a gold positive.

    Args:
        candidates: Raw mined negatives.
        config: Validation knobs.
        seed: Deterministic seed.
        provenance: Run tag for the filtering step.

    Returns:
        Candidates with false negatives marked ``validation_status="failed"``.
    """
    raise NotImplementedError


def filter_incomplete_qa(
    candidates: list[dict[str, Any]],
    config: ValidationConfig,
    seed: int,
    provenance: str,
) -> list[dict[str, Any]]:
    """Drop QA rows missing a question or an answer (``skipped_incomplete``).

    Args:
        candidates: Candidate dicts carrying lowercased ``fields`` maps
            (chunker convention: ``question``/``answer`` et al.).
        config: Validation knobs.
        seed: Deterministic seed.
        provenance: Run tag for the filtering step.

    Returns:
        Only complete QA candidate dicts.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Tiny pure helpers (implemented; exercised by scaffolding tests)
# ---------------------------------------------------------------------------


def is_false_negative(negative_chunk_id: str, positive_chunk_ids: list[str]) -> bool:
    """Return True if the "negative" id is actually a gold positive.

    Args:
        negative_chunk_id: Candidate negative chunk id.
        positive_chunk_ids: Gold positive chunk ids for the query.

    Returns:
        True when *negative_chunk_id* is a member of *positive_chunk_ids*.
    """
    return negative_chunk_id in set(positive_chunk_ids)


def normalize_validation_status(status: str) -> str:
    """Normalize a validation status string to ``pending``/``passed``/``failed``.

    Args:
        status: Raw status string (case-insensitive, stripped).

    Returns:
        Canonical status; unknown values map to ``"pending"``.
    """
    s = status.strip().lower()
    if s in ("pending", "passed", "failed"):
        return s
    return "pending"


def is_complete_qa_record(fields: dict[str, Any]) -> bool:
    """Return True if lowercased QA *fields* carry both question and answer.

    Mirrors the chunker row filter: question keys ``question`` / ``پرسش`` /
    ``سوال`` / ``متن سوال`` / ``متن_سوال``; answer keys ``answer`` /
    ``briefanswer`` / ``پاسخ`` / ``متن پاسخ`` / ``متن_پاسخ`` /
    ``پاسخ کوتاه`` / ``پاسخ کامل``. Empty/blank values do not count.

    Args:
        fields: Lowercased field map from chunk ``metadata.fields``.

    Returns:
        True when at least one question key and one answer key are present
        and non-blank.
    """
    question_keys = ("question", "پرسش", "سوال", "متن سوال", "متن_سوال")
    answer_keys = (
        "answer",
        "briefanswer",
        "پاسخ",
        "متن پاسخ",
        "متن_پاسخ",
        "پاسخ کوتاه",
        "پاسخ کامل",
    )
    has_q = any(str(fields.get(k, "")).strip() for k in question_keys)
    has_a = any(str(fields.get(k, "")).strip() for k in answer_keys)
    return bool(has_q and has_a)
