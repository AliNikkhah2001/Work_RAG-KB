# Evaluation (deprecated package)

This package no longer runs benchmarks. **Work_RAG-Evaluation_Suite** is the sole backend:

- HTTP: `EVAL_SUITE_URL` (default `http://127.0.0.1:8300`)
- Datasets: `components/evaluation/datasets/`
- Quality plan: [`docs/EVAL_QUALITY_IMPROVEMENT_PLAN.md`](../../docs/EVAL_QUALITY_IMPROVEMENT_PLAN.md)
- KB UI: `/benchmarks` proxies the suite

Importing `kb_manager.evaluation` raises `ImportError` by design.
