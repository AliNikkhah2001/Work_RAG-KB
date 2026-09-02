# KB Manager - Knowledge Base Management System

> **ICS Credit Scoring Knowledge Base** — Process, version, and manage your Persian-language knowledge base for RAG agents.

## Current Status — v7 (1405-05-31 KB, 2026-09-02)

| Metric | Value |
|--------|-------|
| **Version** | `v7_iva_1405-05-31` (fresh build from `kb-source/1405-05-31`, tag `v7`) |
| **Documents / Chunks** | 34 docs, 2,074 chunks (585 QA pairs, 979 body, 499 reason_detail) |
| **DB** | `data/kb_1405.db` (fresh) + `dense_embeddings.npz` (2074×384) |
| **Source** | `D:/Code/KB/kb-source/1405-05-31` (excludes `TestQuestions_IVA` + `TestQuestion*` dirs) |
| **IVA Test** | **15 questions** from `TestQuestions_IVA/InitialTestQuestion.xlsx` — [`versions/v7_iva_1405-05-31/IVA_REPORT.md`](versions/v7_iva_1405-05-31/IVA_REPORT.md) |
| **Doc-level Hit@5** | **73.3% (11/15)** — golden-doc chunk present in top-5 |
| **Doc-level MRR** | 0.466 |
| **Answer Hit@5** | 20% (3/15, ≥70% answer tokens in top-5) |
| **Avg Latency** | ~22.7s/query (CPU cross-encoder 50-pool) |
| **Pipeline** | Full rebuild 23.3s, 34 processed / 0 failed / 307 incomplete QA skipped — `data/pipeline_summary_1405.json` |
| **Tests** | `pytest` suite passing |
| **Web UI** | http://127.0.0.1:8000 (serves `kb_1405.db`) |

> **v7 Notice:** New isolated KB from the `1405-05-31` folder (individual/corporate/cheque/saire/fanni content). Synonym beam5 + colloquial→formal expansion added (`kb_manager/query_expansion.py`, 74 entries) lifts Persian conversational queries; pipeline now skips `TestQuestion*` source dirs so test datasets are never ingested. 4 IVA misses are ranking-quality (reason-code "guaranteed loan" Q11/12 and semantic Q14/15 — see `diag_pretank.py`; reranker demotes golden chunks). Next: increase RERANKER_TOP_K 50→100 for pool, or per-domain rerank.

## Quick Start

```bash
pip install -e ".[dev]"
python run_server.py
# Open http://127.0.0.1:8000
```

The server pre-warms the search index (BM25 + dense embeddings + cross-encoder reranker) on startup (~30-60s first time).

### Ingest from Source

The pipeline page at `/pipeline` has an editable source directory (defaults to `kb-source/` submodule). Click **Full Rebuild** to ingest.

```bash
# Or via CLI
python -c "
import asyncio
from kb_manager.pipeline.orchestrator import PipelineOrchestrator
from kb_manager.models.database import Database
from kb_manager.config import load_config

async def main():
    cfg = load_config()
    db = Database(cfg.db)
    orch = PipelineOrchestrator(database=db)
    summary = await orch.run_full_rebuild(cfg.source_dir)
    print(summary.to_dict())

asyncio.run(main())
"
```

## Retrieval Pipeline

Four-stage hybrid retriever with cross-encoder reranking:

```
Query → Persian Normalization + Char 3-grams
      → BM25 (lexical) ─────────────────┐
      → Dense Semantic (MiniLM L12) ─────┤
                                         ↓
                              RRF Fusion (k=60)
                                         ↓
                         Cross-encoder Rerank (top-50)
                                         ↓
                              Final Top-K Results
```

- **BM25**: Okapi BM25 with Persian-aware tokenization, char 3-grams for typo robustness, keyword 3x boost
- **Dense**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) with contextual embeddings (title + heading prepended)
- **RRF**: Reciprocal Rank Fusion over BM25 + Dense ranked lists
- **Reranker**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` on top-50 candidates

## Benchmark Results

### v7 — IVA 15 questions, 1405-05-31 KB (verbatim, CPU)

| Metric | Doc-level | Answer-grounded (≥70% tokens in top-5) |
|--------|-----------|---------------------------------------|
| Hit@5 | **11/15 (73.3%)** | 3/15 (20%) |
| MRR | 0.466 | — |
| Avg latency | ~22.7s | — |

Per-question results: [`versions/v7_iva_1405-05-31/IVA_REPORT.md`](versions/v7_iva_1405-05-31/IVA_REPORT.md) · `data/iva_results.json`

### v5 — 120 queries, 6208 chunks (baseline, `aa5c576`)

| Format | Hit@5 | Top-1 | MRR | Avg Latency |
|--------|-------|-------|-----|-------------|
| verbatim | 95.0% | 80.0% | 0.867 | 3.3s |
| paraphrase | 90.0% | 80.0% | 0.842 | 3.6s |
| typo | 95.0% | 90.0% | 0.917 | 3.3s |
| reworded | 75.0% | 70.0% | 0.725 | 3.6s |
| conversational | 85.0% | 75.0% | 0.792 | 7.5s |
| keyword_only | 65.0% | 15.0% | 0.367 | 3.5s |
| **Overall** | **84.2%** | **68.3%** | **0.751** | **4.2s** |

## Web UI Pages

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Document/chunk counts, domain/category breakdown |
| Documents | `/documents` | Browse, filter, manage documents |
| Chunks | `/chunks` | View/edit chunks, search content |
| Pipeline | `/pipeline` | Trigger rebuild/incremental, job history |
| Search | `/search` | Interactive search with step-by-step transparency |
| Benchmarks | `/benchmarks` | Run retrieval benchmarks, view plots, create snapshots |
| Comparison | `/benchmarks/comparison` | Version comparison charts (v2→v3→v4) |
| Versions | `/versions` | Document version history and rollback |
| Cleanup QA | `/cleanup/qa` | Filter incomplete QA chunks |
| Monitoring | `/monitoring` | Staleness reports, retrieval metrics |

## Project Structure

```
kb-manager/
├── kb_manager/              # Main package
│   ├── config.py            # Configuration (env vars + defaults)
│   ├── cli.py               # CLI interface
│   ├── models/
│   │   └── database.py      # SQLAlchemy ORM (Document, Chunk, DocumentVersion, IngestionJob)
│   ├── parsers/
│   │   ├── xlsx_parser.py   # Excel with schema auto-detection
│   │   ├── pdf_parser.py    # PDF (PyMuPDF)
│   │   └── docx_parser.py   # DOCX
│   ├── preprocessor/
│   │   ├── persian.py       # ZWNJ normalization, Unicode cleanup
│   │   └── pipeline.py      # Chained preprocessing
│   ├── chunker/
│   │   ├── semantic.py      # Structure-aware chunking (QA pairs, reason codes, articles)
│   │   └── fixed.py         # Fixed-size chunking
│   ├── dense/
│   │   └── dense_index.py   # DenseSemanticIndex (MiniLM embeddings + .npz cache)
│   ├── reranker/
│   │   └── cross_encoder.py # CrossEncoderReranker (mMiniLMv2)
│   ├── evaluation/
│   │   ├── benchmark.py     # BenchmarkRunner + AsyncBenchmarkRunner
│   │   ├── generator.py     # Synthetic test data generator
│   │   ├── metrics.py       # IR metrics (ranx + pure-Python fallback)
│   │   ├── plots.py         # Performance plots (hit rate, MRR, latency)
│   │   └── query_formats.py # 6 query format transformations
│   ├── cleanup/
│   │   └── qa_cleanup.py    # Find/preview/cleanup incomplete QA chunks
│   ├── pipeline/
│   │   ├── orchestrator.py  # PipelineOrchestrator (parse→chunk→embed→store)
│   │   ├── versioning.py    # Version snapshots
│   │   └── quality.py       # Quality gates
│   └── web/                 # FastAPI web UI
│       ├── app.py           # FastAPI app with lifespan (table creation + search pre-warm)
│       ├── deps.py          # Shared DB + templates (avoids circular imports)
│       ├── routes/
│       │   ├── benchmarks.py  # Benchmark execution + version snapshots
│       │   ├── chunks.py      # Chunk management
│       │   ├── cleanup.py     # QA cleanup dashboard
│       │   ├── documents.py   # Document management
│       │   ├── monitoring.py  # Staleness + metrics
│       │   ├── pipeline.py    # Pipeline control (actually runs orchestrator)
│       │   ├── search.py      # Search API + BM25/dense/reranker
│       │   └── versions.py    # Document version history
│       ├── templates/       # Jinja2 HTML templates
│       └── static/lib/      # Local Chart.js (CDN blocked)
├── data/
│   ├── kb_test.db           # SQLite database
│   ├── test_questions.json  # 120 benchmark queries (6 formats × 20)
│   ├── benchmark_comparison.json  # v2/v3/v4 comparison data
│   └── plots/               # Generated benchmark plots
├── versions/                # Version snapshots
│   └── v4_retrieval/        # Latest versioned snapshot
├── kb-source/               # Git submodule (78 XLSX source files)
├── scripts/
│   └── cleanup_incomplete_qa.py  # CLI cleanup tool
├── tests/                   # 17 tests (chunker + pipeline)
├── run_server.py            # Start web server
├── regen_test_questions.py  # Regenerate benchmark dataset from current KB
└── pyproject.toml
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_DB_URL` | `sqlite+aiosqlite:///./data/kb_test.db` | Database URL |
| `KB_SOURCE_DIR` | `./kb-source` | Source files directory |
| `KB_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `KB_CHUNK_STRATEGY` | `semantic` | Chunking strategy |
| `KB_CHUNK_MAX` | `512` | Max tokens per chunk |
| `KB_WEB_PORT` | `8000` | Web server port |

## Running Tests

```bash
pytest tests/ -v
```

## CLI Commands

```bash
python run_server.py                    # Start web server
python -m kb_manager.cli ingest --full  # Full rebuild via CLI
python -m kb_manager.cli status         # Show KB stats
python scripts/cleanup_incomplete_qa.py --dry-run  # Preview QA cleanup
```

## Roadmap & Remediation — Updated for v7

Full audit: [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) (36 findings)

### Done
- [x] **Phase 0** Safety net — `data/test_questions.sha256` frozen, `data/versions.lock`, `tests/test_characterization.py` 5 tests
- [x] **Phase 1** Immediate crashes — ordinal propagation (`search.py` 7-tuple), CPU dtype `float32` on CPU, chunk page `content`, reranker pool 50 (`82e8e3d`)
- [x] **Phase 2** Async boundary — `llm.py:103` `async def generate`, `benchmark.py:173` await, `search.py:285` lock (`37c8c09`)
- [x] **Phase 3** Index/ingestion — fingerprint `model+context` (`dense.py:120`), invalidation fingerprint+`max(updated_at)`, parent key map O(1), dedup `true`/`kb-source` (`08aa35a`), duplicate-doc fix (`orchestrator.py:270`), CLI session fix (`cli.py:52`)
- [x] **Phase 4** Retrieval consolidation — `regex_persian.py` central maps, `persian.py` delegates, BM25 `kw*3` field weight, ZWNJ tokenizer fix (`73e7f5b` + `109d403`)
- [x] **Phase 5** Evaluation integrity — drop by index, real typo map, no `1.5` fallback, FaMTEB schema validate, `RAGEvaluator→Heuristic` rename (`b308493`)
- [x] **Phase 6** HyDE consolidation — single `hyde.py`, `query_reform.HyDE` deprecated, strict JSON, `KB_ALLOW_MOCK` gate, disabled by default (`c2a2856`)
- [x] **Phase 7** Operational — `deps.py:get_db` DI, job TTL 50, `app.py` router, logging (`40c578c`)
- [x] **Phase 8** Dead code — `C19-21/B16/B17` removed (`5dad5d1`)
- [x] **Phase 0-8 merged** to `master`, tag `v6.0`
- [x] **v7 KB build** — fresh `data/kb_1405.db` from `kb-source/1405-05-31`: 34 docs / 2074 chunks / 307 incomplete-filtered, 23.3s (`orchestrator.py` skips `TestQuestion*` dirs so test data is never indexed)
- [x] **v7 colloquial expansion** — `query_expansion.py` extended to 74 entries (قسطشون→قسط, رتبم→رتبه, چکم→چک, چی→چه, رو→را, توی→در…)
- [x] **v7 IVA test** — full 15-question run of `TestQuestions_IVA/InitialTestQuestion.xlsx`: doc-hit@5 **11/15 (73.3%)**, MRR 0.466, snapshot + `IVA_REPORT.md` + `iva_results.json` (`run_iva_eval.py`, `build_iva_dataset.py`)
- [x] **v7 live** — `run_server.py` now serves `kb_1405.db` (1405-05-31 source)

### Remaining (Next)

| Priority | Task | Owner | Est. |
|----------|------|-------|------|
| **P0** | IVA ranking — raise doc-hit@5 73.3%→85%+: `RERANKER_TOP_K 50→100` (golden chunks for Q11/12 ARE in pool but reranker demotes), per-domain rerank, or promote `reason_detail_parent`/`qa_pair_parent` aggregation for semantic Q14/15 | retrieval | 1d |
| **P1** | Full IVA answer-grounded model eval (RAGAS faithfulness/relevancy on the 15 Q+A) with Gemini/Ollama | eval | 1d |
| **P1** | Performance — `RERANKER_TOP_K 50→30` GPU (equal headroom), batch 64→128, quantize `float16` on H200, BM25 cache across restarts → target <2s warm | perf | 1w |
| **P2** | FaMTEB live 600q — `famteb.py` + `run_benchmark.py --famteb` smoke, publish to leaderboard | eval | 1d |
| **P2** | Corpus dedup apply — `python -m kb_manager.cli dedup --db-path data/kb_1405.db` (2074→~1700 target) + re-benchmark | data | 1d |
| **P2** | Multi-query rewriting (beam5 RRF with LLM) — consolidate `query_reform.MultiQueryGenerator`, mocked tests | retrieval | 2d |
| **P2** | HyDE A/B controlled — `docs/BENCHMARK_REPORT.md` hit@5/MRR/p50/p95, `hyde_enabled`, disabled default | report | 1d |
| **P3** | Ops hardening — branch cleanup 17 branches tag & delete, `config` drift doc, gitignore `SKILL.md`/parent repos | ops | 1d |
| **P3** | Documentation — `IMPLEMENTATION_PLAN.md` + `PLAN.md` Phase 10-15 status, v6/v7 comparison plots to `data/plots` | docs | 1d |

See `docs/REMEDIATION_PLAN.md` §§5,9 for PR sequence and §§11-12 DoD. Next tag `v7.1` after IVA ranking fix + GPU benchmark.

## License

Private — ICS Credit Scoring
