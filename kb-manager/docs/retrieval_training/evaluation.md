# Evaluation (design stub)

## Metrics

Reuse `kb_manager/evaluation/metrics.py`: `RetrievalMetrics.compute_all`
(precision/recall/hit_rate @K, MRR, nDCG@K, MAP@K) with the ranx-backed
`RanxRetrievalEvaluator` where installed, and `BenchmarkRunner` /
`AsyncBenchmarkRunner` from `kb_manager/evaluation/benchmark.py`
(multi-gold: a hit is any `expected_chunk_ids` member in the top-K).

## Flow (stage 5, `evaluation.py`)

1. `evaluate_checkpoint(checkpoint_path, dataset_path, config, seed, provenance)` —
   run the eval dataset through the pipeline with the candidate checkpoint
   swapped in (reranker model or dense cache), emit per-query ranks +
   aggregated metrics.
2. `compare_against_baseline(...)` — diff against
   `artifacts/retrieval_training/baseline_freeze.json`
   (`retrieval_config`: rrf_k=60, rerank_top_k=100, keyword_boost=3.0,
   synonym_beam=5, fusion_alpha=0.7, pin merged-top3+bm25-top3<=10) and report
   per-metric deltas + pass/fail.
3. Scaffolding pure helpers (`reciprocal_rank`, `hit_indicator`, `mean_of`)
   pin the metric edge semantics (miss = 0, 1-based ranks).

## Fusion ablations (stage 6, `fusion.py`)

Sweep `FusionConfig(rrf_k, fusion_alpha, pin_guard, top_k)` over
`fuse_rank_lists` (RRF), `interpolate_rerank_rrf` (α blend) and
`apply_pin_guard`, scored through the same stage-5 harness so fusion changes
are comparable to model changes.
