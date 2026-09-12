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
