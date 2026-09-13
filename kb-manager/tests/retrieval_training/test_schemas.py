"""Round-trip + literal validation tests for retrieval_training schemas (must pass now)."""

from __future__ import annotations

import pytest

from kb_manager.retrieval_training.schemas import (
    NEGATIVE_TIERS,
    VALIDATION_STATUSES,
    DatasetSplit,
    FailureRecord,
    MinedNegative,
    RunManifest,
    TierRatio,
    from_json,
    to_json,
)


def _sample_negative() -> MinedNegative:
    return MinedNegative(
        query_id="q1",
        query="sample query",
        positive_chunk_ids=["c-pos-1"],
        negative_chunk_id="c-neg-9",
        negative_type="Tier2",
        bm25_rank=3,
        dense_rank=12,
        rrf_rank=5,
        reranker_rank=7,
        source_document="doc-1",
        validation_status="pending",
        provenance="rev/db/cfg",
    )


def test_mined_negative_round_trip() -> None:
    rec = _sample_negative()
    clone = MinedNegative.from_json(rec.to_json())
    assert clone == rec


def test_module_helpers_round_trip() -> None:
    rec = _sample_negative()
    assert from_json(MinedNegative, to_json(rec)) == rec
    manifest = RunManifest(code_rev="abc", db_sha="def", config={"k": 60}, seed=7)
    assert from_json(RunManifest, to_json(manifest)) == manifest


def test_tier_literals_accepted() -> None:
    for tier in NEGATIVE_TIERS:
        assert MinedNegative(query_id="q", query="q", negative_type=tier).negative_type == tier  # type: ignore[arg-type]
    assert tuple(NEGATIVE_TIERS) == ("Tier1", "Tier2", "Tier3", "Tier4")


def test_tier_literal_rejected() -> None:
    with pytest.raises(ValueError):
        MinedNegative(query_id="q", query="q", negative_type="Tier5")  # type: ignore[arg-type]


def test_validation_status_literal_rejected() -> None:
    with pytest.raises(ValueError):
        MinedNegative(query_id="q", query="q", validation_status="maybe")  # type: ignore[arg-type]
    assert tuple(VALIDATION_STATUSES) == ("pending", "passed", "failed")


def test_tier_ratio_round_trip_and_validate() -> None:
    ratio = TierRatio(tier1=0.1, tier2=0.2, tier3=0.3, tier4=0.4)
    assert TierRatio.from_json(ratio.to_json()) == ratio
    assert ratio.as_dict() == {"Tier1": 0.1, "Tier2": 0.2, "Tier3": 0.3, "Tier4": 0.4}


def test_tier_ratio_rejects_bad_sum() -> None:
    with pytest.raises(ValueError):
        TierRatio(tier1=0.5, tier2=0.5, tier3=0.5, tier4=0.5)


def test_failure_record_round_trip() -> None:
    rec = FailureRecord(
        query_id="q9", query="q", expected_chunk_ids=["a"], retrieved_ids=["b"],
        rank=-1, top_k=10, failure_type="miss", provenance="p",
    )
    assert FailureRecord.from_json(rec.to_json()) == rec


def test_dataset_split_round_trip_and_size() -> None:
    split = DatasetSplit(name="train", query_ids=["q1", "q2"], provenance="p")
    assert split.size == 2
    assert DatasetSplit.from_json(split.to_json()) == split


def test_run_manifest_round_trip() -> None:
    manifest = RunManifest(code_rev="96438c1", db_sha="9eac43ae", config={"rrf_k": 60}, seed=42)
    clone = RunManifest.from_json(manifest.to_json())
    assert clone == manifest
    assert clone.config["rrf_k"] == 60
    assert clone.seed == 42
