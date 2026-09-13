"""Evaluation stage tests: contract skeletons (skipped) + pure-helper tests (pass now)."""

from __future__ import annotations

import pytest

from kb_manager.retrieval_training.evaluation import (
    hit_indicator,
    mean_of,
    reciprocal_rank,
)


@pytest.mark.skip(reason="implemented by mining/validation agents")
def test_evaluate_checkpoint_contract() -> None:
    raise AssertionError("evaluate_checkpoint not implemented yet")


@pytest.mark.skip(reason="implemented by mining/validation agents")
def test_compare_against_baseline_contract() -> None:
    raise AssertionError("compare_against_baseline not implemented yet")


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(1) == pytest.approx(1.0)
    assert reciprocal_rank(4) == pytest.approx(0.25)
    assert reciprocal_rank(-1) == 0.0
    assert reciprocal_rank(0) == 0.0


def test_hit_indicator() -> None:
    assert hit_indicator(1, k=10) == 1
    assert hit_indicator(10, k=10) == 1
    assert hit_indicator(11, k=10) == 0
    assert hit_indicator(-1, k=10) == 0


def test_mean_of() -> None:
    assert mean_of([1.0, 0.0, 0.5]) == pytest.approx(0.5)
    assert mean_of([]) == 0.0
