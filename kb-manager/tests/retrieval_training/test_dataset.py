"""Dataset stage tests: contract skeletons (skipped) + pure-helper tests (pass now)."""

from __future__ import annotations

import pytest

from kb_manager.retrieval_training.dataset import (
    deterministic_split,
    validate_split_ratios,
)


@pytest.mark.skip(reason="implemented by mining/validation agents")
def test_build_dataset_contract() -> None:
    raise AssertionError("build_dataset not implemented yet")


@pytest.mark.skip(reason="implemented by mining/validation agents")
def test_split_dataset_contract() -> None:
    raise AssertionError("split_dataset not implemented yet")


def test_validate_split_ratios_ok() -> None:
    validate_split_ratios(0.8, 0.1, 0.1)


def test_validate_split_ratios_bad_sum() -> None:
    with pytest.raises(ValueError):
        validate_split_ratios(0.5, 0.5, 0.5)


def test_deterministic_split_cover_and_disjoint() -> None:
    ids = [f"q{i}" for i in range(10)]
    splits = deterministic_split(ids, seed=42, train_ratio=0.8, val_ratio=0.1, provenance="p")
    assert set(splits) == {"train", "val", "test"}
    assert splits["train"].size == 8
    assert splits["val"].size == 1
    assert splits["test"].size == 1
    all_ids = splits["train"].query_ids + splits["val"].query_ids + splits["test"].query_ids
    assert sorted(all_ids) == sorted(ids)


def test_deterministic_split_reproducible() -> None:
    ids = [f"q{i}" for i in range(10)]
    first = deterministic_split(ids, seed=7, train_ratio=0.8, val_ratio=0.1)
    second = deterministic_split(ids, seed=7, train_ratio=0.8, val_ratio=0.1)
    assert first["train"].query_ids == second["train"].query_ids
    assert first["test"].query_ids == second["test"].query_ids
