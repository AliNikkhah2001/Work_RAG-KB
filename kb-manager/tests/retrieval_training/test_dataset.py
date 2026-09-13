"""Dataset stage tests: contracts + pure-helper tests."""

from __future__ import annotations

import pytest

from kb_manager.retrieval_training.dataset import (
    DatasetConfig,
    build_dataset,
    deterministic_split,
    split_dataset,
    validate_split_ratios,
    write_dataset_manifest,
)


def _neg(qid, tier, idx, pos="POS", f="Individual_CRM_Questions.xlsx", q=None, doc=None):
    return {
        "query_id": qid,
        "query": q if q is not None else f"question {qid}",
        "positive_text": pos,
        "negative_text": f"neg-{tier}-{idx}-{qid}",
        "negative_type": tier,
        "source_file": f,
        "gold_document_id": doc if doc is not None else f"doc-{qid}",
    }


def _batch():
    recs = []
    # large file, 10 queries x 4 tiers
    for i in range(10):
        qid = f"IND#{i}"
        for t in ("Tier1", "Tier2", "Tier3", "Tier4"):
            recs.append(_neg(qid, t, 0))
        recs.append(_neg(qid, "Tier2", 1))  # extra Tier2 (priority trim keeps Tier1)
    # small files wholly to train
    recs.append(_neg("DIS#0", "Tier2", 0, f="DisputeQuestions.xlsx"))
    recs.append(_neg("ETE#0", "Tier3", 0, f="EtebaritoProblems.xlsx"))
    # another large file, 10 queries
    for i in range(10):
        qid = f"CHQ#{i}"
        for t in ("Tier2", "Tier4"):
            recs.append(_neg(qid, t, 0, f="ChequeQuestions.xlsx"))
    fails = [
        {"query_id": "IND#0", "failure_type": "A"},
        {"query_id": "IND#1", "failure_type": "B"},
        {"query_id": "CHQ#0", "failure_type": "C"},
    ]
    return recs, fails


def test_build_dataset_contract() -> None:
    recs, fails = _batch()
    cfg = DatasetConfig(negatives_per_positive=4)
    out = build_dataset(recs, fails, cfg, seed=9, provenance="ds1")
    assert set(out["splits"]) == {"train", "validation", "test"}
    rows = {r["query_id"]: r for s in out["splits"].values() for r in s}
    assert len(rows) == 22  # 10 IND + 10 CHQ + DIS + ETE
    # CE shape: max 4, Tier1 first
    ind0 = rows["IND#0"]
    assert set(ind0) >= {"query", "positive_text", "negatives_text", "positive",
                         "hard", "medium", "easy", "tiers", "source_file", "split"}
    assert len(ind0["negatives_text"]) == 4
    assert ind0["tiers"][0] == "Tier1"
    # dense mapping: hard=Tier1+Tier2, medium=Tier3, easy=Tier4
    assert len(ind0["hard"]) == 3 and len(ind0["medium"]) == 1 and len(ind0["easy"]) == 0
    # no-leakage: every query in exactly one split (cover + disjoint)
    seen = [r["query_id"] for s in out["splits"].values() for r in s]
    assert sorted(seen) == sorted(rows)
    # file stratification: large files divided ~80/10/10, small wholly train
    by_file_split = {}
    for name, srows in out["splits"].items():
        for r in srows:
            by_file_split.setdefault(r["source_file"], {}).setdefault(name, 0)
            by_file_split[r["source_file"]][name] += 1
    assert by_file_split["Individual_CRM_Questions.xlsx"]["train"] in (7, 8)
    assert sum(by_file_split["Individual_CRM_Questions.xlsx"].values()) == 10
    assert by_file_split["ChequeQuestions.xlsx"]["train"] in (7, 8)
    assert sum(by_file_split["ChequeQuestions.xlsx"].values()) == 10
    assert set(by_file_split["DisputeQuestions.xlsx"]) == {"train"}
    assert set(by_file_split["EtebaritoProblems.xlsx"]) == {"train"}
    # small files wholly in train
    assert rows["DIS#0"]["split"] == "train" and rows["ETE#0"]["split"] == "train"
    # fail representatives in validation+test
    val_ids = {r["query_id"] for r in out["splits"]["validation"]}
    test_ids = {r["query_id"] for r in out["splits"]["test"]}
    assert val_ids & {"IND#0", "IND#1", "CHQ#0"}
    assert test_ids & {"IND#0", "IND#1", "CHQ#0"}
    # manifest + stats keys
    assert out["manifest"]["seed"] == 9
    assert out["stats"]["queries"] == 22
    assert out["stats"]["tier_counts"]["Tier1"] == 10
    assert out["stats"]["avg_negatives_per_query"] > 0
    # determinism
    out2 = build_dataset(recs, fails, cfg, seed=9, provenance="ds1")
    assert out["stats"] == out2["stats"]
    assert [r["query_id"] for r in out["splits"]["train"]] == \
        [r["query_id"] for r in out2["splits"]["train"]]


def test_build_dataset_errors_and_malformed() -> None:
    cfg = DatasetConfig()
    # error entries tallied, not silent
    recs = [_neg("Q1", "Tier2", 0),
            {"query_id": "Q2", "error": "missing chunk content for c-xyz"}]
    out = build_dataset(recs, [], cfg, seed=1, provenance="p")
    assert out["stats"]["queries"] == 1
    assert len(out["stats"]["errors"]) == 1
    # malformed rows rejected with clear errors
    with pytest.raises(ValueError):
        build_dataset([{"query_id": "Q", "query": "q", "positive_text": "p",
                        "negative_text": "n", "negative_type": "Tier9"}],
                      [], cfg, seed=1, provenance="p")
    with pytest.raises(ValueError):
        build_dataset([{"query_id": "Q", "query": "q", "negative_text": "n",
                        "negative_type": "Tier1"}],
                      [], cfg, seed=1, provenance="p")
    with pytest.raises(ValueError):
        build_dataset([{"query": "q", "positive_text": "p", "negative_text": "n",
                        "negative_type": "Tier1"}],
                      [], cfg, seed=1, provenance="p")


def test_split_dataset_contract() -> None:
    cfg = DatasetConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    ids = [f"q{i}" for i in range(20)]
    splits = split_dataset(ids, cfg, seed=13, provenance="sp")
    assert set(splits) == {"train", "val", "test"}
    all_ids = splits["train"].query_ids + splits["val"].query_ids + splits["test"].query_ids
    assert sorted(all_ids) == sorted(ids)  # cover, no leakage/duplication
    assert len(splits["train"].query_ids) == 16
    assert len(splits["val"].query_ids) == 2
    assert len(splits["test"].query_ids) == 2
    again = split_dataset(ids, cfg, seed=13, provenance="sp")
    assert again["train"].query_ids == splits["train"].query_ids
    other = split_dataset(ids, cfg, seed=99, provenance="sp")
    assert other["train"].query_ids != splits["train"].query_ids
    man = write_dataset_manifest(splits, cfg, code_rev="abc123", db_sha="deadbeef",
                                 seed=13, provenance="sp")
    assert man["code_rev"] == "abc123" and man["db_sha"] == "deadbeef" and man["seed"] == 13


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
