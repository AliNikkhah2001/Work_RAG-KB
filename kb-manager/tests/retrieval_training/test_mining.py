"""Mining stage tests: contracts + pure-helper tests."""

from __future__ import annotations

import pytest

from kb_manager.retrieval_training.mining import (
    MiningConfig,
    collect_failure_records,
    deterministic_sample,
    mine_negatives,
    per_query_seed,
    rank_to_tier,
    validate_tier_ratios,
)
from kb_manager.retrieval_training.schemas import TierRatio


def _cand(cid, **kw):
    rec = {"chunk_id": cid, "bm25_rank": -1, "dense_rank": -1, "rrf_rank": -1,
           "reranker_rank": -1, "source_document": "doc-other"}
    rec.update(kw)
    return rec


def _queries():
    # q1: gold g1 (merged 5, final 6); above-gold reranked bad1/bad2;
    # merged top-10 hard1/hard2; same-doc mid1 (rrf 30); other-doc easy pool.
    q1 = {
        "query_id": "Q1",
        "query": "sample question one",
        "positive_chunk_ids": ["g1"],
        "gold_merged_rank": 5,
        "gold_final_rank": 6,
        "gold_document_id": "doc-gold",
        "candidates": [
            _cand("g1", rrf_rank=5, reranker_rank=6, bm25_rank=1, dense_rank=2,
                  source_document="doc-gold", question_text="sample question one"),
            _cand("bad1", reranker_rank=0, rrf_rank=0, source_document="doc-x"),
            _cand("bad2", reranker_rank=2, rrf_rank=1, source_document="doc-x"),
            _cand("hard1", rrf_rank=3, bm25_rank=0, source_document="doc-x"),
            _cand("hard2", rrf_rank=4, bm25_rank=3, source_document="doc-x"),
            # exact dup question text of hard1 -> dedup removal keeps first
            _cand("hard1dup", rrf_rank=6, bm25_rank=4, source_document="doc-x",
                  question_text="shared wording here",
                  ),
            _cand("mid1", rrf_rank=30, source_document="doc-gold"),
            _cand("easy1", source_document="doc-e1"),
            _cand("easy2", source_document="doc-e2"),
            _cand("easy3", source_document="doc-e3"),
        ],
    }
    q1["candidates"][3]["question_text"] = "shared wording here"  # hard1 dup pair
    # q2: multi-gold g2a/g2b, both must be excluded
    q2 = {
        "query_id": "Q2",
        "query": "second question",
        "positive_chunk_ids": ["g2a", "g2b"],
        "gold_document_id": "doc-g2",
        "candidates": [
            _cand("g2a", rrf_rank=0, reranker_rank=0, source_document="doc-g2"),
            _cand("g2b", rrf_rank=1, reranker_rank=1, source_document="doc-g2"),
            _cand("n1", rrf_rank=2, source_document="doc-z"),
            _cand("n2", source_document="doc-e9"),
        ],
    }
    return [q1, q2]


def test_mine_negatives_contract() -> None:
    cfg = MiningConfig(negatives_per_query=12)
    recs = mine_negatives(_queries(), cfg, seed=7, provenance="rev/db/cfg")
    by_q: dict[str, list] = {}
    for r in recs:
        by_q.setdefault(r.query_id, []).append(r)
    # gold exclusion incl multi-gold
    got_q1 = {r.negative_chunk_id for r in by_q["Q1"]}
    assert "g1" not in got_q1
    got_q2 = {r.negative_chunk_id for r in by_q["Q2"]}
    assert not ({"g2a", "g2b"} & got_q2)
    # dup removal: hard1dup dropped (same normalized question as hard1)
    assert "hard1dup" not in got_q1
    # tier presence + priority caps (Tier4 checked via direct call below:
    # the union may legitimately yield <2 Tier4 when the seeded sample hits
    # chunks already claimed by higher tiers)
    tiers_q1 = {r.negative_type for r in by_q["Q1"]}
    assert {"Tier1", "Tier2", "Tier3"} <= tiers_q1
    assert sum(1 for r in by_q["Q1"] if r.negative_type == "Tier1") <= 3
    assert sum(1 for r in by_q["Q1"] if r.negative_type == "Tier2") <= 4
    assert sum(1 for r in by_q["Q1"] if r.negative_type == "Tier3") <= 3
    assert sum(1 for r in by_q["Q1"] if r.negative_type == "Tier4") <= 2
    # Tier1 only above-gold final ranks
    for r in by_q["Q1"]:
        if r.negative_type == "Tier1":
            assert 0 <= r.reranker_rank < 6
    # Tier4 direct: seeded other-doc sample, cap 2, gold-free, deterministic
    from kb_manager.retrieval_training.mining import mine_tier4_negatives
    t4 = mine_tier4_negatives(_queries(), cfg, seed=7, provenance="p")
    q1t4 = [r for r in t4 if r.query_id == "Q1"]
    assert 0 < len(q1t4) <= 2
    assert all(r.negative_chunk_id != "g1" for r in q1t4)
    t4b = mine_tier4_negatives(_queries(), cfg, seed=7, provenance="p")
    assert [x.negative_chunk_id for x in t4] == [x.negative_chunk_id for x in t4b]
    for r in recs:
        assert r.validation_status == "pending"
        assert "rev/db/cfg" in r.provenance
    # determinism under seed
    again = mine_negatives(_queries(), cfg, seed=7, provenance="rev/db/cfg")
    assert [x.to_json() for x in recs] == [x.to_json() for x in again]
    # negatives_per_query cap respected with Tier1 priority
    capped = mine_negatives(_queries(), MiningConfig(negatives_per_query=2), seed=7, provenance="p")
    q1c = [r for r in capped if r.query_id == "Q1"]
    assert len(q1c) == 2
    assert q1c[0].negative_type == "Tier1"


def test_collect_failure_records_contract() -> None:
    rows = [
        {"query_id": "a#1", "query": "q one", "expected_chunk_ids": ["g1", "g2"]},
        {"query": "q two", "expected_chunk_ids": ["h1"], "failure_type": "A"},
    ]
    out = collect_failure_records(rows, top_k=100, seed=3, provenance="pv")
    assert out[0]["expected_chunk_ids"] == ["g1", "g2"]  # multi-gold kept
    assert out[0]["top_k"] == 100 and out[1]["query_id"] == "q1"
    assert all(o["provenance"].startswith("pv") for o in out)
    with pytest.raises(ValueError):
        collect_failure_records([{"query": "  ", "expected_chunk_ids": ["g"]}], 10, 0, "p")
    with pytest.raises(ValueError):
        collect_failure_records([{"query": "q", "expected_chunk_ids": []}], 10, 0, "p")
    with pytest.raises(ValueError):
        collect_failure_records([{"query": "q"}], 10, 0, "p")


def test_mine_determinism_seeded_no_hash_jitter() -> None:
    # NOTE: PYTHONHASHSEED only affects hash(); Tier4 sampling derives its
    # sub-seed from zlib.crc32 (per_query_seed), so labels are identical
    # under any hash seed. The retrieval legs themselves still need
    # PYTHONHASHSEED=0 pinned at interpreter startup.
    assert per_query_seed(42, "Q1") == per_query_seed(42, "Q1")
    assert per_query_seed(42, "Q1") != per_query_seed(43, "Q1")
    pool = [f"c{i}" for i in range(30)]
    first = deterministic_sample(pool, 2, per_query_seed(99, "QX"))
    assert deterministic_sample(list(reversed(pool)), 2, per_query_seed(99, "QX")) == first


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


def test_mine_negatives_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        mine_negatives([{"query": "no id", "positive_chunk_ids": ["g"]}],
                       MiningConfig(), seed=0, provenance="p")
    with pytest.raises(ValueError):
        mine_negatives([{"query_id": "x", "query": "q", "positive_chunk_ids": []}],
                       MiningConfig(), seed=0, provenance="p")
