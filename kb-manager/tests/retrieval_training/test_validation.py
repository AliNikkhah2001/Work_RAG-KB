"""Validation stage tests: contract skeletons (skipped) + pure-helper tests (pass now)."""

from __future__ import annotations

import pytest

from kb_manager.retrieval_training.validation import (
    is_complete_qa_record,
    is_false_negative,
    normalize_validation_status,
)


@pytest.mark.skip(reason="implemented by mining/validation agents")
def test_validate_negatives_contract() -> None:
    raise AssertionError("validate_negatives not implemented yet")


@pytest.mark.skip(reason="implemented by mining/validation agents")
def test_filter_false_negatives_contract() -> None:
    raise AssertionError("filter_false_negatives not implemented yet")


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
