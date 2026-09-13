"""Validation contracts (stage 2).

Drops false negatives (a "negative" that is actually a gold positive) and
incomplete QA rows, then marks
:class:`~kb_manager.retrieval_training.schemas.MinedNegative.validation_status`.
Reuses the dedup normalizer semantics from
``kb_manager/chunker/semantic.py::_normalize_question`` (Arabic yeh/kaf,
ZWNJ to space, strip question marks, collapse whitespace).

8-rule filter (all deterministic; verdicts ``accepted``/``uncertain`` map to
``validation_status`` ``passed``/``failed``; uncertain rows are DROPPED from
training pools but KEPT in the provenance file with their reason):

* R1 ``self_gold`` — negative id is one of the query's own
  ``positive_chunk_ids``.
* R2 ``cross_gold`` — negative id is a gold for the same normalized question
  in ANY other record. The normalized-question -> goldset map is built from
  the full candidate batch FIRST (run scripts pass all rows plus every gold
  from ``retrieval_failures.jsonl``, so same-question golds across files are
  caught).
* R3 ``parent_child`` — the negative chunk is a parent chunk
  (``metadata.parent_key`` / ``is_parent`` / ``chunk_type`` ending
  ``_parent``; parents are already excluded from the search index, still
  checked). Parent signals travel in the record's ``provenance`` JSON
  envelope (``chunk_type`` / ``is_parent`` keys) stamped by the run script
  from DB SELECTs.
* R4 ``near_dup`` — char-3-gram Jaccard between the negative's question text
  and the gold question text > 0.85 -> ``uncertain``, NOT auto-accepted.
* R5 ``same_doc_overlap`` — negative from the same document as the gold AND
  high word-level overlap with the gold question (token-set Jaccard > 0.7)
  -> ``uncertain``.
* R6 ``exact_question`` — the negative's question normalizes exactly to the
  query (same question, different chunk id: a definite false negative).
* R7 ``duplicate`` — the same ``(query_id, negative_chunk_id)`` pair already
  judged in this batch (second occurrence fails).
* R8 ``malformed`` — blank ``negative_chunk_id`` or blank ``query``.

Question texts (``neg_question`` / ``gold_question``) and document ids
(``neg_doc`` / ``gold_doc``) are read from the ``provenance`` JSON envelope
when present (run script resolves them via DB content by chunk id); when
absent, R3/R4/R5 fall back to ``source_document`` equality (R5) and the
record's own ``query`` as the gold question (R4/R6), and are skipped only if
no text is available at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
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
    "char3_jaccard",
    "word_jaccard",
    "JACCARD_NEAR_DUP_THRESHOLD",
    "SAME_DOC_OVERLAP_THRESHOLD",
]

#: Char-3-gram Jaccard above this -> ``uncertain`` (R4), never auto-accept.
JACCARD_NEAR_DUP_THRESHOLD: float = 0.85

#: Same-doc word-overlap Jaccard above this -> ``uncertain`` (R5).
SAME_DOC_OVERLAP_THRESHOLD: float = 0.7


@dataclass
class ValidationConfig:
    """Knobs for the validation stage."""

    drop_false_negatives: bool = True
    drop_incomplete_qa: bool = True
    dedup_via_normalizer: bool = True


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def normalize_question_text(q: str) -> str:
    """Normalize with the chunker normalizer when available (fallback inline)."""
    try:
        from kb_manager.chunker.semantic import SemanticChunker

        return SemanticChunker._normalize_question(q)
    except Exception:
        s = (
            q.strip()
            .replace("\u064a", "\u06cc")
            .replace("\u0643", "\u06a9")
            .replace("\u0671", "\u0627")
            .replace("\u200c", " ")
            .replace("؟", "")
            .replace("?", "")
        )
        return re.sub(r"\s+", " ", s).strip()


def _char3grams(s: str) -> set[str]:
    s = normalize_question_text(s)
    if len(s) < 3:
        return {s} if s else set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


def char3_jaccard(a: str, b: str) -> float:
    """Char-3-gram Jaccard similarity in [0, 1] (1.0 when both empty)."""
    sa, sb = _char3grams(a), _char3grams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def word_jaccard(a: str, b: str) -> float:
    """Whitespace-token Jaccard over normalized text in [0, 1]."""
    sa = set(normalize_question_text(a).split())
    sb = set(normalize_question_text(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _payload(rec: MinedNegative) -> dict[str, Any]:
    """Decode the provenance JSON envelope; ``{}`` when absent/invalid.

    Tolerates chain affixes: when the full string is not JSON, the substring
    from the first ``{`` to the last ``}`` is tried (the mining chain is a
    plain ``a|b:c`` string while the validation run script stores a JSON
    envelope; post-stamp chains append ``|validate:...`` after the JSON).
    """
    text = rec.provenance or ""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (ValueError, AttributeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else {}
        except ValueError:
            return {}
    return {}


def _neg_question(rec: MinedNegative, env: dict[str, Any]) -> str:
    for k in ("neg_question", "negative_question", "candidate_question"):
        v = env.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _gold_question(rec: MinedNegative, env: dict[str, Any]) -> str:
    for k in ("gold_question", "positive_question"):
        v = env.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    return rec.query or ""


def _neg_doc(rec: MinedNegative, env: dict[str, Any]) -> str:
    for k in ("neg_doc", "neg_document_id", "negative_document_id"):
        v = env.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return (rec.source_document or "").strip()


def _gold_doc(rec: MinedNegative, env: dict[str, Any]) -> str:
    for k in ("gold_doc", "gold_document_id", "gold_source_document"):
        v = env.get(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _is_parent(env: dict[str, Any]) -> bool:
    ctype = env.get("chunk_type", "")
    if isinstance(ctype, str) and ctype.strip().endswith("_parent"):
        return True
    if env.get("is_parent") is True:
        return True
    if isinstance(env.get("parent_key", ""), str) and env.get("parent_key", "").strip():
        # A present parent_key alone marks sheet-scoped children too; only
        # treat as parent when is_parent/chunk_type says so.
        return False
    return False


def build_gold_map(candidates: list[MinedNegative]) -> dict[str, set[str]]:
    """Build normalized-question -> gold-id-set over the batch (R2 needs the
    FULL batch first so cross-file same-question golds are caught)."""
    gold_map: dict[str, set[str]] = {}
    for rec in candidates:
        norm = normalize_question_text(rec.query or "")
        if not norm:
            continue
        gold_map.setdefault(norm, set()).update(rec.positive_chunk_ids or [])
    return gold_map


def _judge(
    rec: MinedNegative,
    env: dict[str, Any],
    gold_map: dict[str, set[str]],
    seen_pairs: set[tuple[str, str]],
) -> tuple[str, str]:
    """Return ``(verdict, reason)`` with verdict in {accepted, uncertain}."""
    if not (rec.negative_chunk_id or "").strip() or not (rec.query or "").strip():
        return "uncertain", "R8:malformed"
    neg_id = rec.negative_chunk_id.strip()
    own_golds = set(rec.positive_chunk_ids or [])
    if neg_id in own_golds:
        return "uncertain", "R1:self_gold"
    norm_q = normalize_question_text(rec.query or "")
    if norm_q:
        cross = set(gold_map.get(norm_q, set())) - own_golds
        if neg_id in cross:
            return "uncertain", "R2:cross_gold"
    if _is_parent(env):
        return "uncertain", "R3:parent_child"
    neg_q = _neg_question(rec, env)
    gold_q = _gold_question(rec, env)
    if neg_q and (gold_q or norm_q):
        nq_norm = normalize_question_text(neg_q)
        if (gold_q and nq_norm == normalize_question_text(gold_q)) or (
            norm_q and nq_norm == norm_q
        ):
            # Same question text under a different chunk id: the negative
            # answers this exact query -> definite false negative.
            return "uncertain", "R6:exact_question"
        if gold_q and char3_jaccard(neg_q, gold_q) > JACCARD_NEAR_DUP_THRESHOLD:
            return "uncertain", "R4:near_dup"
        neg_doc, gold_doc = _neg_doc(rec, env), _gold_doc(rec, env)
        if (
            neg_doc
            and gold_doc
            and neg_doc == gold_doc
            and word_jaccard(neg_q, gold_q) > SAME_DOC_OVERLAP_THRESHOLD
        ):
            return "uncertain", "R5:same_doc_overlap"
    elif neg_q and norm_q and normalize_question_text(neg_q) == norm_q:
        return "uncertain", "R6:exact_question"
    pair = (rec.query_id, neg_id)
    if pair in seen_pairs:
        return "uncertain", "R7:duplicate"
    return "accepted", "ok"


def _stamp(rec: MinedNegative, verdict: str, reason: str, seed: int, provenance: str) -> MinedNegative:
    status = "passed" if verdict == "accepted" else "failed"
    chain = f"{rec.provenance}|validate:{verdict}:{reason}:seed={seed}:{provenance}"
    return replace(rec, validation_status=status, provenance=chain)  # type: ignore[arg-type]


def validate_negatives(
    candidates: list[MinedNegative],
    config: ValidationConfig,
    seed: int,
    provenance: str,
    extra_gold_map: dict[str, list[str]] | None = None,
) -> list[MinedNegative]:
    """Run the full validation pipeline over mined *candidates*.

    Builds the normalized-question -> goldset map over the whole batch
    first (R2), merged with *extra_gold_map* (e.g. every gold from
    ``retrieval_failures.jsonl`` so same-question golds across files are
    caught even for queries absent from this batch), then judges every
    record with the 8-rule filter. ALL records are returned (uncertain ones
    marked ``failed`` with their reason in the provenance chain — kept for
    provenance, dropped from training pools by the dataset stage).

    Args:
        candidates: Raw mined negatives (``validation_status="pending"``).
        config: Validation knobs.
        seed: Deterministic seed for any sampling / tie-breaking.
        provenance: Run tag appended to each record's provenance chain.
        extra_gold_map: Optional normalized-question -> gold-ids mapping
            merged into the batch-derived map (keys must already be
            normalized with the chunker normalizer).

    Returns:
        Validated records with ``validation_status`` set to ``"passed"``
        or ``"failed"``.
    """
    gold_map = build_gold_map(candidates)
    if extra_gold_map:
        for k, v in extra_gold_map.items():
            gold_map.setdefault(k, set()).update(v)
    seen: set[tuple[str, str]] = set()
    out: list[MinedNegative] = []
    for rec in candidates:
        env = _payload(rec)
        if config.drop_false_negatives:
            verdict, reason = _judge(rec, env, gold_map, seen)
        else:
            verdict, reason = ("accepted", "ok")
        seen.add((rec.query_id, (rec.negative_chunk_id or "").strip()))
        out.append(_stamp(rec, verdict, reason, seed, provenance))
    return out


def filter_false_negatives(
    candidates: list[MinedNegative],
    config: ValidationConfig,
    seed: int,
    provenance: str,
    extra_gold_map: dict[str, list[str]] | None = None,
) -> list[MinedNegative]:
    """Drop candidates whose negative id is actually a gold positive.

    Applies the id-based guards (R1 self-gold, R2 cross-query gold, R6 exact
    normalized-question match when the text is available); text-overlap rules
    (R4/R5) belong to :func:`validate_negatives`.

    Args:
        candidates: Raw mined negatives.
        config: Validation knobs.
        seed: Deterministic seed.
        provenance: Run tag for the filtering step.
        extra_gold_map: Optional normalized-question -> gold-ids mapping
            merged into the batch-derived map.

    Returns:
        Candidates with false negatives marked ``validation_status="failed"``.
    """
    if not config.drop_false_negatives:
        return list(candidates)
    gold_map = build_gold_map(candidates)
    if extra_gold_map:
        for k, v in extra_gold_map.items():
            gold_map.setdefault(k, set()).update(v)
    seen: set[tuple[str, str]] = set()
    out: list[MinedNegative] = []
    for rec in candidates:
        env = _payload(rec)
        neg_id = (rec.negative_chunk_id or "").strip()
        verdict, reason = "accepted", "ok"
        if not neg_id or not (rec.query or "").strip():
            verdict, reason = "uncertain", "R8:malformed"
        elif neg_id in set(rec.positive_chunk_ids or []):
            verdict, reason = "uncertain", "R1:self_gold"
        else:
            norm_q = normalize_question_text(rec.query or "")
            if norm_q and neg_id in (set(gold_map.get(norm_q, set())) - set(rec.positive_chunk_ids or [])):
                verdict, reason = "uncertain", "R2:cross_gold"
            else:
                neg_q = _neg_question(rec, env)
                if neg_q and norm_q and normalize_question_text(neg_q) == norm_q:
                    verdict, reason = "uncertain", "R6:exact_question"
                elif (rec.query_id, neg_id) in seen:
                    verdict, reason = "uncertain", "R7:duplicate"
        seen.add((rec.query_id, neg_id))
        out.append(_stamp(rec, verdict, reason, seed, provenance))
    return out


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
        Only complete QA candidate dicts (all rows when
        ``config.drop_incomplete_qa`` is False).
    """
    if not config.drop_incomplete_qa:
        return list(candidates)
    out: list[dict[str, Any]] = []
    for cand in candidates:
        fields = cand.get("fields", {}) if isinstance(cand, dict) else {}
        if isinstance(fields, dict) and is_complete_qa_record(fields):
            out.append(cand)
    return out


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
