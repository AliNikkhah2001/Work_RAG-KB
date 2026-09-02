# KB Manager - Knowledge Base Management System

> **ICS Credit Scoring Knowledge Base** — Process, version, and manage your Persian-language knowledge base for RAG agents.

## Current Status — v6.0 (2026-09-02)

| Metric | Value |
|--------|-------|
| **Version** | `v6.0` `109d403` → `master` merge (remediation P0-P8 + Persian central + synonym/dedup) |
| **Documents** | 69 (78 XLSX source, 9 failed — empty sheets) |
| **Chunks** | 3,626 (432 QA pairs, 1,913 body, 1,255 reason_detail, 26 parents) — down from 6,208 due to dedup + empty-sheet filtering |
| **DB** | SQLite (`data/kb_test.db`, ~3.6 GB) + `dense_embeddings.npz` (3626×384, fingerprint v2) |
| **Hit@5** | **100% on 5-query smoke (v6 regenerated dataset, synonym OFF)** — full 120q benchmark pending GPU host (CPU 40-60s/query, see §Benchmark) |
| **Hit@5 v5** | 84.2% (120q, 6 formats) — baseline before v6 (verbatim 95%, keyword_only 65%, MRR 0.751, 4.2s on H200) |
| **Tests** | 45 collected — 43 passed (preprocessor+chunker+parsers+cli 28, characterization 5, eval 7, embedder 3, pipeline 2) |
| **Web UI** | http://127.0.0.1:8000 |
| **Remediation** | P0-P8 merged to `master`, tag `v6.0` — see [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) |
| **Pipeline** | Full rebuild 102.5s, 69 updated / 9 failed / 92 incomplete QA skipped / 62 versions — `data/pipeline_summary.json` |

> **v6 Notice:** Pipeline duplicate-doc bug fixed (`orchestrator.py:270` force path), Persian maps centralized (`regex_persian.py` 189L), synonym beam5 + MinHash LSH added (`dedup.py`, `query_expansion.py`). Dense cache rebuilt (57s warm). Full 120q A/B needs GPU host — current CPU cross-encoder rerank dominates latency (40-60s/q). Use `KB_SYNONYM_ENABLED=false` for baseline; `true` adds 5× BM25/dense max-pool (keyword_only lift). See `versions/v6_persian_central_synonym_dedup/snapshot.json`.

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

### v6 — preliminary (3626 chunks, regenerated 120q, CPU)

| Dataset | Hit@5 | Top-1 | MRR | Avg Latency | Notes |
|---------|-------|-------|-----|-------------|-------|
| 5-query smoke, synonym OFF | 100% | 100% | 1.0 | 67.3s (cold 173s, warm 28-48s) | `KB_SYNONYM_ENABLED=false`, cross-encoder 50-pool on CPU |
| 120q full | — | — | — | — | Pending GPU host; CPU 40-60s/q → ~80 min for 120q. Run `python run_benchmark.py` on H200 with `KB_SYNONYM_ENABLED` toggle for A/B |

> Regenerated `data/test_questions.json` from 432 QA pairs (v6 DB) — old IDs stale after dedup/pipefix. Full A/B requires frozen checksum + `tests/test_characterization.py::test_dataset_checksum_frozen` update.

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

## Roadmap & Remediation — Updated for v6

Full audit: [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) (36 findings)

### Done (merged to `master`, tag `v6.0`)
- [x] **Phase 0** Safety net — `data/test_questions.sha256` frozen, `data/versions.lock`, `tests/test_characterization.py` 5 tests
- [x] **Phase 1** Immediate crashes — ordinal propagation (`search.py` 7-tuple), CPU dtype `float32` on CPU, chunk page `content`, reranker pool 50 (`82e8e3d`)
- [x] **Phase 2** Async boundary — `llm.py:103` `async def generate`, `benchmark.py:173` await, `search.py:285` lock (`37c8c09`)
- [x] **Phase 3** Index/ingestion — fingerprint `model+context` (`dense.py:120`), invalidation fingerprint+`max(updated_at)`, parent key map O(1), dedup `true`/`kb-source` (`08aa35a`), duplicate-doc fix (`orchestrator.py:270`), CLI session fix (`cli.py:52`)
- [x] **Phase 4** Retrieval consolidation — `regex_persian.py` central maps, `persian.py` delegates, BM25 `kw*3` field weight, ZWNJ tokenizer fix (`73e7f5b` + `109d403`)
- [x] **Phase 5** Evaluation integrity — drop by index, real typo map, no `1.5` fallback, FaMTEB schema validate, `RAGEvaluator→Heuristic` rename (`b308493`)
- [x] **Phase 6** HyDE consolidation — single `hyde.py`, `query_reform.HyDE` deprecated, strict JSON, `KB_ALLOW_MOCK` gate, disabled by default (`c2a2856`)
- [x] **Phase 7** Operational — `deps.py:get_db` DI, job TTL 50, `app.py` router, logging (`40c578c`)
- [x] **Phase 8** Dead code — `C19-21/B16/B17` removed (`5dad5d1`)
- [x] **v6 Feature** Persian central — `regex_persian.py` (189L) + `validators.py` (178L) + `persian.py` delegates diacritics/repetition/digits
- [x] **v6 Feature** Synonym/dedup — `query_expansion.py` beam5 (CURATED 40 + generated 221-entry map), `synonym_eda.py`+`generator.py`, `dedup.py` MinHash 64/8, `search.py` BM25/dense max-pool beam, `cli.py dedup` command, `data/domain_profile.json`
- [x] **Pipeline** Full rebuild v6 — 78→69 docs, 3626 chunks, `pipeline_summary.json` (102s)
- [x] **Tests** 45 tests — 43 passed

### Remaining (Next)

| Priority | Task | Owner | Est. |
|----------|------|-------|------|
| **P0 Next** | Full 120q A/B on GPU H200 — `KB_SYNONYM_ENABLED` off vs beam5, 50 vs 30 pool, with frozen `test_questions.json` checksum update | infra | 1 day GPU |
| **P1** | Fix 9 XLSX parse failures (empty sheets: `PublicQuesions_Individual` etc) — add `len(headers)<2` guard or sheet fallback | parser | 2h |
| **P1** | Performance — reduce `RERANKER_TOP_K 50→30` (~40% rerank), batch 64→128, quantize `float16` on GPU, BM25 cache across restarts | perf | 1w |
| **P2** | FaMTEB live 600q — `famteb.py` + `run_benchmark.py --famteb` smoke 10-sample, publish to leaderboard | eval | 1d |
| **P2** | Corpus dedup apply — `python -m kb_manager.cli dedup` (6208→~5000 target) + rebuild + re-benchmark | data | 1d |
| **P2** | Multi-query rewriting (beam5 RRF) — consolidate `hyde.py` + `query_expansion` LLM path (`query_reform.MultiQueryGenerator`) already coded, needs wiring + mocked tests | retrieval | 2d |
| **P2** | HyDE A/B controlled report — `docs/BENCHMARK_REPORT.md` with hit@5/MRR/p50/p95, cost, variance ×3 runs, `hyde_enabled` in schema | report | 1d |
| **P3** | Ops hardening — `config` drift doc, `versions/v6` → `v7` promotion, branch cleanup 17 branches tag & delete, `SKILL.md`/`Work_Credit-RAG_Phase1` gitignore | ops | 1d |
| **P3** | Documentation — update `IMPLEMENTATION_PLAN.md` progress, `PLAN.md` Phase 10-15, publish `versions/v6` comparison plots | docs | 1d |

See `docs/REMEDIATION_PLAN.md` §§5,9 for PR sequence and §§11-12 DoD. Next tag `v6.1` after GPU A/B + 50→30 pool.

## License

Private — ICS Credit Scoring
