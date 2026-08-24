# KB Manager — Implementation Plan v3

## Current State (v5 — August 2026)

### What's Done

| Workstream | Status | Details |
|-----------|--------|---------|
| **W1: Fix broken QA chunks** | ✅ DONE | Incomplete QA rows filtered during ingestion, cleanup tool (CLI + web) |
| **W2: Structured chunk format** | ✅ DONE | Persian field names (سوال/پاسخ/کلیدواژه), metadata.fields for filtering |
| **W3: Hierarchical chunking** | ✅ DONE | Parent-child chunks (sheet/document scope), parent_id FK |
| **W4: Retrieval pipeline v4** | ✅ DONE | BM25+ngram + Dense MiniLM + RRF + Cross-encoder reranker |
| **W5: Search pre-warm** | ✅ DONE | BM25 index + dense embeddings + reranker loaded at startup |
| **W6: Benchmark framework** | ✅ DONE | 120 queries (6 formats), async execution, web UI |
| **W7: QA cleanup tool** | ✅ DONE | CLI + web dashboard for filtering incomplete QA chunks |
| **W8: Web UI fixes** | ✅ DONE | Versions tab, comparison charts, monitoring dashboard |
| **W9: Git submodule** | ✅ DONE | `kb-source/` → Work_RAG-KB-SourceFiles (78 XLSX) |

### Benchmark Results (v5 — 120 queries)

| Format | Hit@5 | MRR | Notes |
|--------|-------|-----|-------|
| verbatim | 95.0% | 0.867 | Baseline |
| paraphrase | 90.0% | 0.842 | Synonym swaps + reordering |
| typo | 95.0% | 0.917 | Char 3-grams handle orthographic variants |
| reworded | 75.0% | 0.725 | 40% token drop + shuffle |
| conversational | 85.0% | 0.792 | Formal→informal |
| keyword_only | 65.0% | 0.367 | Hardest format |
| **Overall** | **84.2%** | **0.751** | **4.2s avg latency** |

---

## Next Steps

### Phase 10: Latency Optimization

**Goal:** Reduce avg latency from 4.2s to <2s per query.

| Task | Complexity | Impact |
|------|-----------|--------|
| Quantize dense embeddings (INT8) | 🟢 Low | 2x faster dense search |
| Reduce RERANKER_TOP_K 50→30 | 🟢 Low | 40% less reranking |
| Cache BM25 index across server restarts | 🟡 Medium | Skip 5s rebuild |
| Async cross-encoder batching | 🟡 Medium | Pipeline reranking |

### Phase 11: keyword_only Improvement

**Goal:** Improve keyword_only Hit@5 from 65% to >80%.

| Task | Complexity | Impact |
|------|-----------|--------|
| HyDE (Hypothetical Document Embeddings) | 🟡 Medium | +10-15% MRR |
| Multi-query rewriting (beam 5) | 🟡 Medium | +3-6% MRR |
| Query expansion with synonyms | 🟢 Low | +5% Hit |

### Phase 12: Corpus Deduplication

**Goal:** Reduce 6,208 chunks to ~5,000 by removing near-duplicates.

| Task | Complexity | Impact |
|------|-----------|--------|
| MinHash LSH dedup | 🟡 Medium | +5-10% Hit |
| QA pair dedup (normalized question) | 🟢 Low | Cleaner corpus |

### Phase 13: FaMTEB Benchmark

**Goal:** Compare against Persian IR leaderboard.

| Task | Complexity | Impact |
|------|-----------|--------|
| Run FaMTEB datasets (synper_qa, nq_fa, miracle_fa) | 🟡 Medium | External validation |
| Publish results to FaMTEB leaderboard | 🟢 Low | Community visibility |

### Phase 14: Documentation & Release

| Task | Complexity | Impact |
|------|-----------|--------|
| Update README with v5 results | 🟢 Low | ✅ DONE |
| Write technical blog post | 🟡 Medium | Knowledge sharing |
| Package as pip-installable | 🟡 Medium | Easy deployment |

---

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| DB | SQLite (dev) / PostgreSQL (prod) | SQLite for simplicity, PostgreSQL for scale |
| Embedding model | MiniLM L12 (384-dim) | Best multilingual quality/size tradeoff |
| Reranker | mMiniLMv2 cross-encoder | Multilingual, matches embedding model |
| Fusion | RRF (k=60) | Rank-based, robust to score distribution differences |
| Chunking | Semantic (structure-aware) | Preserves QA pair boundaries, heading paths |
| Parent scope | Per-sheet | Matches natural document structure |

## Key Files

| File | Purpose |
|------|---------|
| `kb_manager/web/routes/search.py` | BM25 + Dense + RRF + Reranker search pipeline |
| `kb_manager/chunker/semantic.py` | Structure-aware chunking with QA filtering |
| `kb_manager/evaluation/benchmark.py` | BenchmarkRunner for retrieval evaluation |
| `kb_manager/evaluation/query_formats.py` | 6 query format transformations |
| `kb_manager/web/app.py` | FastAPI app with startup pre-warm |
| `regen_test_questions.py` | Regenerate benchmark dataset from current KB |
