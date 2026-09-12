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
