"""Mining stage tests: contract skeletons (skipped) + pure-helper tests (pass now)."""

from __future__ import annotations

import pytest

from kb_manager.retrieval_training.mining import (
    MiningConfig,
    deterministic_sample,
    rank_to_tier,
    validate_tier_ratios,
)
from kb_manager.retrieval_training.schemas import TierRatio


@pytest.mark.skip(reason="implemented by mining/validation agents")
def test_mine_negatives_contract() -> None:
    raise AssertionError("mine_negatives not implemented yet")


@pytest.mark.skip(reason="implemented by mining/validation agents")
def test_collect_failure_records_contract() -> None:
    raise AssertionError("collect_failure_records not implemented yet")


def test_validate_tier_ratios_ok() -> None:
    validate_tier_ratios(TierRatio(0.25, 0.25, 0.25, 0.25))


def test_validate_tier_ratios_bad_sum() -> None:
    ratio = TierRatio(0.25, 0.25, 0.25, 0.25)
    ratio.tier1 = 0.9  # break the sum post-construction
    with pytest.raises(ValueError):
        validate_tier_ratios(ratio)


def test_deterministic_sample_reproducible() -> None:
    pool = [f"c{i}" for i in range(20)]
    assert deterministic_sample(pool, 5, seed=42) == deterministic_sample(pool, 5, seed=42)
    assert len(deterministic_sample(pool, 5, seed=42)) == 5


def test_deterministic_sample_capped_at_pool() -> None:
    assert deterministic_sample(["a", "b"], 10, seed=1) == ["a", "b"]


def test_rank_to_tier_mapping() -> None:
    assert rank_to_tier(2, -1, -1) == "Tier2"
    assert rank_to_tier(2, 4, -1) == "Tier3"
    assert rank_to_tier(2, 4, 1) == "Tier4"


def test_mining_config_defaults_use_tier_ratio() -> None:
    cfg = MiningConfig()
    assert abs(cfg.tier_ratios.tier1 + cfg.tier_ratios.tier2 + cfg.tier_ratios.tier3 + cfg.tier_ratios.tier4 - 1.0) < 1e-6
