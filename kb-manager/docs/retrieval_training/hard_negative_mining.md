# Hard-Negative Mining (tier design + proportions + provenance)

## Tier design

Mined per query against the frozen pipeline
(`search_knowledge_base`: BM25 content+keywords → dense → RRF k=60 →
cross-encoder rerank of the top-100 pool → fusion α=0.7 → pin-guard).
Ranks from each leg are stored on the record (`bm25_rank`, `dense_rank`,
`rrf_rank`, `reranker_rank`; `-1` = absent from that leg's pool).

| Tier | Name | Source | Intuition |
|---|---|---|---|
| Tier1 | Near-duplicate / paraphrase | Normalizer collisions + dense near-ties that are NOT the gold (`_normalize_question` semantics: yeh/kaf unify, ZWNJ→space, ؟/? stripped) | Hardest: orthographic/phrasing variants of the gold |
| Tier2 | BM25 lexical hard | High BM25 rank, low dense rank | Shares terms, differs in meaning (tests semantic leg) |
| Tier3 | Dense semantic hard | High dense rank, low BM25 rank | Shares meaning-space, few terms (tests lexical leg) |
| Tier4 | Reranker confuser | Top reranker ranks without being gold | What the final stage almost picked (tests discriminator) |

`rank_to_tier()` in `mining.py` encodes the rank→tier rule (Tier1 is assigned
by the paraphrase path, not by ranks). Validation (stage 2) drops any
candidate whose id is in `positive_chunk_ids` (false negative) and any
answer-less QA row, mirroring the chunker's `skipped_incomplete` gate.

## Proportions

No hard-coded ratios: every mining/dataset signature takes a
`TierRatio` (`schemas.py`: `tier1..tier4`, validated to sum to 1.0).
Default seeding mixture: 0.25 / 0.25 / 0.25 / 0.25 — later agents tune from
eval deltas (e.g. overweight Tier4 if reranker overfits, Tier2 if BM25
regressions appear). `validate_tier_ratios()` / `deterministic_sample()`
keep sampling reproducible under an explicit `seed`.

## Provenance

Every `MinedNegative` carries `provenance: str` — recommended format
`"<code_rev>/<db_sha>/<config_snapshot>/<seed>"` — plus `source_document`
and `RunManifest(code_rev, db_sha, config, seed)` pinned at dataset build.
This chains each negative back to the exact code, DB snapshot
(`baseline_freeze.json`: rev `96438c1…`, `kb_9_7_2026.db` sha), tier mixture,
and seed that produced it.

## Observed yields (frozen v10 baseline, seed 42, `PYTHONHASHSEED=0`)

Mined with `search_knowledge_base(q, top_k=100)` over all 511 evaluated rows
(`scripts/retrieval_training/run_mining.py`, resume-capable, 600 s/query
timeout, 0 errors; two parallel shards, ~74 min wall). Pool = the
`merged_candidates` slice of 100 (RRF ranks deeper than 100 are unreachable
via the public API; Tier3 "outside pool" candidates come from DB same-doc
lookups). Caps per query: Tier1 ≤ 3, Tier2 ≤ 4, Tier3 ≤ 3, Tier4 ≤ 2;
`mine_negatives` union capped at 12/query (`negatives_per_query=12`).

Accepted (post-validation) tier counts per failure class (4601 mined, 17
uncertain → 4584 accepted):

| Class | Queries | Tier1 reranker_mistake | Tier2 hard_rrf | Tier3 medium | Tier4 easy |
|---|---|---|---|---|---|
| A retriever_failure | 18 | 0 | 72 | 54 | 36 |
| B fusion_failure | 2 | 0 | 8 | 6 | 4 |
| C reranker_failure | 2 | 6 | 6 | 6 | 4 |
| D success | 489 | 58 | 1892 | 1457 | 975 |
| Total accepted | 511 | 64 | 1978 | 1523 | 1019 |

Tier1 is empty for A/B by construction (needs gold merged ≤ 20; A has gold
outside merged top-100, B has merged > 20). C yields the full Tier1 set
(above-gold final confusers).

Validation (stage 2, 8-rule filter): 4584/4601 accepted (99.6 %). Uncertain
(kept in `mined_negatives.jsonl` with reason, dropped from training pools):
17 total — `R5:same_doc_overlap` × 12, `R4:near_dup` × 5, all in class D.
No `R1`/`R2`/`R3`/`R6`/`R7` fires (mining already removes gold ids, dedups
normalized questions, and the index excludes `*_parent` chunks).
