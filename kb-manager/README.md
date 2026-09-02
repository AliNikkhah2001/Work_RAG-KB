# KB Manager - Knowledge Base Management System

> **ICS Credit Scoring Knowledge Base** — Process, version, and manage your Persian-language knowledge base for RAG agents.

## Current Status

| Metric | Value |
|--------|-------|
| **Documents** | 355 (78 XLSX source files) |
| **Chunks** | 6,208 (974 QA pairs, 5,203 body, 31 QA parents) |
| **DB** | SQLite (`data/kb_test.db`, ~2.7 GB) |
| **Hit@5** | 84.2% (120 queries, 6 formats) — *see Remediation Plan §7 for frozen A/B protocol* |
| **MRR** | 0.751 |
| **Avg Latency** | 4.2s per query (warm) |
| **Web UI** | http://127.0.0.1:8000 |
| **Tests** | 17/17 passing |
| **Remediation Plan** | [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md) — 12-section evidence-based roadmap (36 findings, P1 hold on `feat/hyde-ragas` merge) |
| **Baseline** | `feat/hyde-ragas` `0d29fb1` · `master` `aa5c576` — frozen dataset required before any merge |

> **Stabilization Notice:** Per `docs/REMEDIATION_PLAN.md` executive assessment, `feat/hyde-ragas` is **on hold** pending Phase 0-2 (cache invalidation, async safety, ordinal fix). Do not merge until controlled HyDE A/B with frozen `data/test_questions.json` checksum is recorded. See `docs/REMEDIATION_PLAN.md` §§1,3,5,11.

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

## Benchmark Results (v5 — 120 queries, current KB)

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

## Roadmap & Remediation

Full audit and phased fix sequence: [`docs/REMEDIATION_PLAN.md`](docs/REMEDIATION_PLAN.md)

- **Phase 0** Safety net: frozen dataset + characterization tests
- **Phase 1** Immediate crashes: lost ordinal, CPU dtype, chunk page, reranker pool
- **Phase 2** Async boundary: `asyncio.run` → `await`
- **Phase 3** Index/ingestion: fingerprint, invalidation, parent linking, dedup defaults (requires corpus rebuild + cache bump)
- **Phase 4** Retrieval consolidation: shared Persian norm, BM25 weighting
- **Phase 5** Evaluation integrity: deterministic query formats, real typo map, no fabricated metrics
- **Phase 6** HyDE consolidation: one canonical `hyde.py`, mocked HTTP tests, disabled by default
- **Phase 7** Operational hardening: dependency injection, job TTL, structured logging
- **Phase 8** Dead code & branch cleanup (17 branches archived)
- **Phase 9** Measured performance (50→30 rerank pool, quantization)

See plan §§5,9 for PR sequence and §§11-12 for Definition of Done / Open Questions.

## License

Private — ICS Credit Scoring
