"""Validation stage tests: contracts + pure-helper tests."""

from __future__ import annotations

import json

from kb_manager.retrieval_training.schemas import MinedNegative
from kb_manager.retrieval_training.validation import (
    filter_false_negatives,
    filter_incomplete_qa,
    is_complete_qa_record,
    is_false_negative,
    normalize_validation_status,
    validate_negatives,
)
from kb_manager.retrieval_training.validation import ValidationConfig


def _rec(qid, query, golds, neg, tier="Tier2", doc="doc-x", **env):
    base = f"mine:tier={tier}:seed=1"
    prov = base if not env else json.dumps({"mine": base, **env}, ensure_ascii=False)
    return MinedNegative(
        query_id=qid, query=query, positive_chunk_ids=list(golds),
        negative_chunk_id=neg, negative_type=tier, source_document=doc,
        provenance=prov,
    )


def test_validate_negatives_contract() -> None:
    q = "what is my credit score"
    cands = [
        _rec("q1", q, ["g1"], "n-ok", doc="doc-a",
             neg_question="how do loans work", gold_question=q,
             neg_doc="doc-a", gold_doc="doc-gold"),
        _rec("q1", q, ["g1"], "g1"),  # R1 self gold
        # R2 cross-query gold: same normalized question, gold of sibling row
        _rec("q2", q + "?", ["g2"], "g1"),
        _rec("q1b", q, ["g9"], "g1"),
        # R4 near-dup of gold question (>0.85 char-3-gram)
        _rec("q3", "credit score report", ["g3"], "n-near", doc="doc-b",
             neg_question="credit score reports", gold_question="credit score report",
             neg_doc="doc-b", gold_doc="doc-g3"),
        # R6 exact question, other chunk id
        _rec("q4", "exact same question", ["g4"], "n-exact", doc="doc-c",
             neg_question="exact same question", gold_question="totally different words here",
             neg_doc="doc-c", gold_doc="doc-g4"),
        # R5 same-doc + high word overlap (char-3-gram below R4 threshold)
        _rec("q5b", "bills timing query", ["g5b"], "n-samedoc", doc="doc-e",
             neg_question="every month pay your bills on time",
             gold_question="pay your bills on time every month",
             neg_doc="doc-e", gold_doc="doc-e"),
        _rec("q5", "parent query here", ["g5"], "n-par", doc="doc-d",
             chunk_type="qa_pair_parent", is_parent=True,
             neg_question="something else entirely", gold_question="parent query here",
             neg_doc="doc-d", gold_doc="doc-g5"),
        # R7 duplicate pair
        _rec("q6", "unique query six", ["g6"], "n-dup"),
        _rec("q6", "unique query six", ["g6"], "n-dup"),
        # R8 malformed
        MinedNegative(query_id="q7", query="has query", positive_chunk_ids=["g7"],
                      negative_chunk_id="  ", provenance="mine:x"),
    ]
    out = validate_negatives(cands, ValidationConfig(), seed=11, provenance="run1")
    by_id = {(r.query_id, r.negative_chunk_id): r for r in out}
    assert len(out) == len(cands)  # kept in provenance file, never dropped
    assert by_id[("q1", "n-ok")].validation_status == "passed"
    for key in [("q1", "g1"), ("q2", "g1"), ("q1b", "g1"), ("q3", "n-near"),
                ("q4", "n-exact"), ("q5", "n-par"), ("q5b", "n-samedoc"), ("q7", "  ")]:
        assert by_id[key].validation_status == "failed", key
        assert "validate:uncertain:R" in by_id[key].provenance
    dups = [r for r in out if (r.query_id, r.negative_chunk_id) == ("q6", "n-dup")]
    assert {r.validation_status for r in dups} == {"passed", "failed"}
    # uncertain reasons recorded for the provenance report
    reasons = [r.provenance for r in out if r.validation_status == "failed"]
    assert any("R1:self_gold" in p for p in reasons)
    assert any("R2:cross_gold" in p for p in reasons)
    assert any("R4:near_dup" in p for p in reasons)
    assert any("R5:same_doc_overlap" in p for p in reasons)


def test_filter_false_negatives_contract() -> None:
    cands = [
        _rec("a", "query a", ["g1", "g2"], "g2"),  # multi-gold hit
        _rec("a", "query a", ["g1", "g2"], "n1"),  # clean
        _rec("b", "query a?", ["g3"], "g1"),  # cross-query gold, same norm question
    ]
    out = filter_false_negatives(cands, ValidationConfig(), seed=5, provenance="ffn")
    status = {(r.query_id, r.negative_chunk_id): r.validation_status for r in out}
    assert status[("a", "g2")] == "failed"
    assert status[("a", "n1")] == "passed"
    assert status[("b", "g1")] == "failed"
    # disabled knob passes everything through
    out2 = filter_false_negatives(cands, ValidationConfig(drop_false_negatives=False),
                                  seed=5, provenance="ffn")
    assert all(r.validation_status == "pending" for r in out2)


def test_validation_deterministic_and_seed_stamped() -> None:
    cands = [_rec("q1", "some query text", ["g1"], "n1"),
             _rec("q1", "some query text", ["g1"], "g1")]
    first = validate_negatives(cands, ValidationConfig(), seed=42, provenance="pv")
    second = validate_negatives(cands, ValidationConfig(), seed=42, provenance="pv")
    assert [r.to_json() for r in first] == [r.to_json() for r in second]
    assert all(":seed=42:pv" in r.provenance for r in first)
    assert [r.validation_status for r in first] == ["passed", "failed"]


def test_filter_incomplete_qa_drops_answerless() -> None:
    rows = [
        {"fields": {"question": "q?", "answer": "a"}},
        {"fields": {"question": "q?"}},
        {"fields": {"answer": "a"}},
    ]
    assert len(filter_incomplete_qa(rows, ValidationConfig(), seed=0, provenance="p")) == 1


def test_is_false_negative_hit() -> None:
    assert is_false_negative("c1", ["c1", "c2"]) is True


def test_is_false_negative_miss() -> None:
    assert is_false_negative("c9", ["c1", "c2"]) is False


def test_normalize_validation_status() -> None:
    assert normalize_validation_status(" Passed ") == "passed"
    assert normalize_validation_status("FAILED") == "failed"
    assert normalize_validation_status("bogus") == "pending"


def test_is_complete_qa_record_english() -> None:
    assert is_complete_qa_record({"question": "q?", "answer": "a"}) is True
    assert is_complete_qa_record({"question": "q?"}) is False
    assert is_complete_qa_record({"answer": "a"}) is False


def test_is_complete_qa_record_persian_and_blank() -> None:
    assert is_complete_qa_record({"پرسش": "سوال", "پاسخ": "جواب"}) is True
    assert is_complete_qa_record({"question": "   ", "answer": "a"}) is False
