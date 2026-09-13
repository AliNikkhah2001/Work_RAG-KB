# Training (design stub)

Two fine-tuning tracks consume the stage-3 dataset; both are contract-only
here (`reranker_training.py`, `dense_training.py`) — heavy compute belongs
to later agents.

## Track A — cross-encoder reranker

- Base: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` via
  `kb_manager/reranker.py::get_reranker` / `CrossEncoderReranker.rerank(query, [dict], top_k)`.
- Input: `(query, positive, negatives)` triples from `build_triples()`
  (TierRatio mixture preserved per query).
- Knobs: `RerankerTrainingConfig` (model, epochs=3, batch=32, lr=2e-5,
  max_length=512, output_dir under `artifacts/retrieval_training/reranker`).
- Export: `export_reranker()` → serving directory + manifest; drop-in for the
  `_RERANKER_MODEL` slot in `search.py`.

## Track B — dense bi-encoder

- Base: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via
  `kb_manager/dense.py::DenseSemanticIndex` (+ `load_or_build`).
- Input: mined query/passage pairs; keep `use_context=True` (Title/Heading/Type
  prefixing) so the fingerprint semantics don't change silently.
- Knobs: `DenseTrainingConfig` (model, epochs=3, batch=64, lr=2e-5,
  `use_context`, output_dir under `artifacts/retrieval_training/dense`).
- Rebuild: `rebuild_dense_index()` regenerates the `.npz` cache so its stored
  fingerprint matches the new model + corpus (never hand-edit the cache).

## Shared rules

- Every entry point takes `(config, seed, provenance)`; record a `RunManifest`.
- No `torch` imports at scaffolding scope; training agents isolate heavy deps.
- Gate both tracks on stage 5: promote only on frozen-baseline improvement.

## Stage-3 dataset stats (seed 42, `PYTHONHASHSEED=0`)

Built by `scripts/retrieval_training/run_dataset.py` from the 4584 accepted
negatives (17 uncertain dropped, 0 missing-chunk errors):
`artifacts/retrieval_training/datasets/{train,validation,test}.jsonl` +
`provenance.json` sidecar (manifest + stats + code rev + DB sha).

Split rule (exact): group by source file (`query_id` prefix); large files
(`Individual*`, `Cheque*`, `Public*`) divided 80/10/10 by seeded shuffle of
query ids; small files (`Dispute*`, `Etebarito*`) wholly to train; fail-class
(A/B/C) representatives forced into validation+test (seeded single-query
moves, train preferred as donor). No query appears in more than one split.

| Split | Queries | Per-file breakdown |
|---|---|---|
| train | 411 | Individual 297, Cheque 54, Public 41, Etebarito 11, Dispute 8 |
| validation | 48 | Individual 37, Cheque 6, Public 5 |
| test | 52 | Individual 38, Cheque 8, Public 6 |

Fail representatives: 22 total → train 18, validation 2
(`Individual_CRM_Questions#6`, `Individual_CRM_Questions#99`), test 2
(`Individual_CRM_Questions#239`, `PublicQuestions#38`).

Row shapes: CE `{query, positive_text, negatives_text}` (max 4/query,
Tier1 > Tier2 > Tier3 > Tier4 priority) + dense `{query, positive, hard
(Tier1+Tier2), medium (Tier3), easy (Tier4)}`. Counts per tier/split (after
the width-4 priority trim): train Tier1 48 / Tier2 1584 / Tier3 12 /
Tier4 0; validation 11 / 180 / 1 / 0; test 5 / 201 / 2 / 0; totals 64 /
1965 / 15 / 0. Average negatives/query: 4.00 (every query kept 4: Tier1
first, then Tier2; Tier3/4 survive only where a query yields fewer than 4
Tier1+Tier2 negatives). `TierRatio` (default 0.25⁴) is preserved in the
manifest for downstream triple sampling, not enforced as per-row quotas.
