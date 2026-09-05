# KB Manager - Knowledge Base Management System

> **ICS Credit Scoring Knowledge Base** — Process, version, and manage your Persian-language knowledge base for RAG agents.

## Current Status — v8 (pgvector HNSW + GPU, 2026-09-05)

| Metric | Value |
|--------|-------|
| **Version** | `v8_pgvector_hnsw` (full `kb-source/clean_files` 78 files + `1405-05-31`, `KB_DB_MODE=pgvector`) |
| **Documents / Chunks** | **103 docs, 6,593 chunks** (1,240 QA pairs incl. 165 `پرسش/پاسخ` Persian, 3,200 body, 2,153 reason_detail; 1 parent/doc) |
| **DB** | `pgvector` `chunks.embedding Vector(384)` `HNSW m16 ef_construction 64` + `dense_embeddings.npz` `6593×384` (backfilled) |
| **Source** | `kb-source/clean_files` + `1405-05-31` (Persian `پرسش/پاسخ` now `crm_qa`) |
| **Tunable** | `KB_KEYWORD_BOOST=3.0` (`GET /search/config`, `POST /search/api {"keyword_boost":5}`), `KB_EMBED_DEVICE=cuda/cpu` `KB_RERANKER_DEVICE` |
| **Batch** | `GET /transparency/questions?group_by=section` + `/json` + `/questions/retrieval-check` + `GET /transparency/benchmarks/by-section` |
| **Avg Latency** | **HNSW CPU 23.1s** → **HNSW GPU 18.4s** `1.3×` (reranker 50-pool `CPU 439ms → GPU 279ms 1.6×`, HNSW `13ms`, dense `144→180ms`); file-based before `~22.7s` similar |
| **Hit@5** | `5q verbatim` `0.00` (test set mismatched after re-ingest; IVA 15 `73.3%` still baseline) – see `data/hnsw_benchmark_detailed.json` |
| **Pipeline** | Full rebuild `~90s` `103 processed` `6593 embedded` `pgvector` |
| **Web UI** | `http://127.0.0.1:8001` `pgvector` + `http://127.0.0.1:8000` via `Caddy` `32221→8000` `0.0.0.0` `supervisor:kb-manager` |

> **v7 Notice:** New isolated KB from the `1405-05-31` folder (individual/corporate/cheque/saire/fanni content). Synonym beam5 + colloquial→formal expansion added (`kb_manager/query_expansion.py`, 74 entries) lifts Persian conversational queries; pipeline now skips `TestQuestion*` source dirs so test datasets are never ingested. 4 IVA misses are ranking-quality (reason-code "guaranteed loan" Q11/12 and semantic Q14/15 — see `diag_pretank.py`; reranker demotes golden chunks). Next: increase RERANKER_TOP_K 50→100 for pool, or per-domain rerank.

## Version History & Differences

| Version | When | KB corpus | Pipeline / Retrieval | Key changes | Benchmark |
|---------|------|-----------|----------------------|-------------|-----------|
| **v1** | Aug 2026 | ~160 files (31Tir1405 zip) | Baseline ingest + BM25 | Initial KB architecture, file taxonomy, schemas | — |
| **v2** | Aug 2026 | 355 docs / 6,208 chunks | BM25 + TF-IDF (RRF k=60) | First hybrid retrieval; QA cleanup start | Hit@5 **90%**, MRR 0.736, 2.8s |
| **v3** | Aug 2026 | 355 docs / 6,208 chunks | BM25 + Dense MiniLM-L12 + RRF | Added dense semantic leg + contextual embeddings | Hit@5 **89.2%**, MRR 0.787, 1.9s |
| **v4** | Aug 2026 | 355 docs / 6,208 chunks | + char 3-grams + cross-encoder reranker | Char n-grams (typo fix), mmarco reranker, structured Persian chunking | Hit@5 **90%**, MRR 0.775, **15.8s** (rerank cost) |
| **v5** | Aug 2026 | 355 docs / 6,208 chunks | P0-P5 frozen dataset + BM25×3 | Frozen dataset checksum, BM25 keyword ×3 weight, typo map fixed, no fabricated metrics | Hit@5 **84.2%**, MRR 0.751, 4.2s |
| **v6** | Sep 2026 | 69 docs / 3,626 chunks | P0-P8 remediation + Persian central + synonym beam5 | `regex_persian.py` central maps, `dedup.py` MinHash LSH, `query_expansion.py` beam5, fingerprint/invalidation, async fixes, duplicate-doc pipefix | 10q smoke **100%** Hit@5 |
| **v7** | Sep 2026 | 34 docs / 2,074 chunks | 1405-05-31 KB + colloquial beam5 | **Fresh KB** from `kb-source/1405-05-31`; `TestQuestion*` dirs excluded; colloquial→formal synonyms; IVA 15 | **Doc-level Hit@5 73.3%** (11/15), MRR 0.466, ~22.7s |
| **v8** ⭐ current | Sep 2026 | **103 docs / 6,593 chunks** | `pgvector HNSW 384 m16` + `tunable keyword×3.0` + `batch questions/section` | `Vector(384)` `HNSW`, `pipeline` persist `pgvector`, `پرسش/پاسخ` `crm_qa` fix (165 rows), `8001 0.0.0.0` `Caddy 8000→8001` | **HNSW CPU 23.1s → GPU 18.4s 1.3×** (rerank 439→279ms 1.6×, HNSW 13ms) – `data/hnsw_benchmark_detailed.json` |

### What changed in v7 (vs v6)

1. **Brand-new isolated corpus**: ingest switched from the old `kb-source/clean_files` (78 files) to the `kb-source/1405-05-31/` tree — `(done)حقوقی`, `(done)حقیقی`, `سایر`, `فنی`. Result: 34 documents, 2,074 chunks (585 QA, 979 body, 499 reason-detail), rebuilt in 23.3s.
2. **Test datasets excluded from indexing** (`orchestrator._scan_files`): any path segment starting with `TestQuestion` is skipped — so `TestQuestions_IVA/` is never ingested.
3. **Colloquial → formal query expansion** (74-entry map): added user-facing Persian forms (چی/چه, رو/را, توی/در, قسطشون/قسط, رتبم/رتبه, چکم/چک…) to close the gap on real conversational questions.
4. **Real user-question benchmark**: `TestQuestions_IVA/InitialTestQuestion.xlsx` (15 questions) with a doc-level ground-truth (golden document that holds the answer). Reported doc-level Hit@5 = 73.3% (11/15), MRR 0.466.
5. **Web fixes shipped in v7**: pipeline jobs no longer stuck `running` (finalize now re-fetches inside the active session + stale jobs marked `interrupted` at startup); snapshot plot images fixed (`{name:path}` route → 404 was a path-segment mismatch); Comparison tab rewritten data-driven (v2→v7, not hard-coded v4); `/versions` rollback persists (missing flush).

### Known v7 gaps (next fixes)

- **4 IVA misses** (Q11/12 reason-code "guaranteed loan", Q14 rank-vs-score, Q15 negative-contract details): golden chunks ARE in the RRF pool for Q11/12 but the cross-encoder demotes them → try `RERANKER_TOP_K 50→100` or per-domain reranking.
- Latency ~22.7s/query is CPU-bound (cross-encoder 50-pool); on GPU (H200) expect the v5-era ~4s baseline. See Remediation Phase 9.
- Only `verbatim` format benchmarked for IVA; expanding to 6 formats + answer-grounded RAGAS eval is the next milestone.

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

### v8 — HNSW pgvector 6593 chunks, RTX 6000 Ada (2026-09-05)

| Variant | Device | HNSW | Avg latency | p95 | Hit@5 (5q verbatim) | Rerank 50 | Dense 10 |
|---------|--------|------|-------------|-----|---------------------|-----------|----------|
| **File-based before** | CPU | file `npz` | ~22.7s | ~34s | 0.00* | 439ms | 144ms |
| **HNSW pgvector CPU** | CPU | `pgvector HNSW m16` `13ms` | **23.1s** | 34.8s | 0.00 | 439ms | 144ms |
| **HNSW pgvector GPU** | **cuda:0** | `pgvector HNSW m16` `13ms` | **18.4s** | 28.8s | 0.00 | **279ms 1.6×** | **180ms 0.8×** |
| **Speedup GPU vs CPU** | — | — | **1.3×** | — | — | 1.6× | — |

*Hit 0/5 on 5q verbatim after re-ingest (mismatched expected ids after Persian fix, IVA 15 still 73.3% baseline). Detailed `data/hnsw_benchmark_detailed.json` + `data/hnsw_benchmark_quick.json`. HNSW `13ms` vs file dense `~5ms` similar; bottleneck is `cross-encoder 50-pool` (CPU 439ms → GPU 279ms). On H200 expect ~4s (v5-era). Tunable `KB_KEYWORD_BOOST=3.0` (`POST /search/api {"keyword_boost":5}`) adds `×5` to `bm25_kw`; `GET /search/config` shows current.

## Web UI Pages

| Page | URL | Description |
|------|-----|-------------|
| Dashboard | `/` | Document/chunk counts, domain/category breakdown |
| Documents | `/documents` | Browse, filter, manage documents — each row has **Inspect → Transparency** |
| Chunks | `/chunks` | View/edit chunks, search content |
| Pipeline | `/pipeline` | Trigger rebuild/incremental, job history |
| Search | `/search` | Interactive search with step-by-step transparency |
| **Transparency** | `/transparency` | **Excel → Chunks pipeline introspection (NEW)** — raw table (exact `_format_cell` bytes), header normalization & schema `overlap ≥60%` debug, parser text, and DB chunks per sheet. Persian-safe (Vazirmatn, `dir=rtl/auto`, UTF-8). Live `POST /transparency/parse-upload` parses any `.xlsx` without indexing. `GET /transparency/api/raw/{doc_id}` returns JSON `charset=utf-8` for byte-level proof. |
| **Questions Batch** | `/transparency/questions?group_by=section` | All `qa_pair` questions divided by hierarchy (`heading_path/parent_key/sheet`) – for `retrieval`/`guardrails`/`RAG` batch; `GET /transparency/questions/json`, `GET /transparency/questions/retrieval-check`, `GET /transparency/benchmarks/by-section` per-section hit@5/MRR |
| Benchmarks | `/benchmarks` | Run retrieval benchmarks, view plots, create snapshots |
| Comparison | `/benchmarks/comparison` | Version comparison charts (v2→v7, data-driven) |
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
| `KB_DB_MODE` | `sqlite` | `sqlite` (light) or `pgvector` (`KB_DB_HOST/PORT/NAME/USER/PASSWORD`); `KB_DB_URL` overrides |
| `KB_DB_URL` | `sqlite+aiosqlite:///./data/kb_test.db` | Database URL (if set, `KB_DB_MODE` inferred) |
| `KB_SQLITE_PATH` | `./data/kb_test.db` | SQLite file (when `KB_DB_MODE=sqlite`) |
| `KB_SOURCE_DIR` | `./kb-source` | Source files directory |
| `KB_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `KB_EMBED_DEVICE` | `cpu` | `cpu` or `cuda` for dense + reranker; `KB_RERANKER_DEVICE` overrides reranker |
| `KB_KEYWORD_BOOST` | `3.0` | Tunable weight for `bm25_kw` (`0..10`, `POST /search/api {"keyword_boost":5}`) |
| `KB_CHUNK_STRATEGY` | `semantic` | Chunking strategy |
| `KB_CHUNK_MAX` | `512` | Max tokens per chunk |
| `KB_WEB_HOST` | `0.0.0.0` | Web host (`0.0.0.0` public) |
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

## Transparency — How Excel Tables Become Chunks (and how to verify Persian)

`GET /transparency` and `GET /transparency/{doc_id}` show the pipeline step-by-step with **exact bytes** so you can prove Persian is not mojibake:

1. **Read sheets** (`parsers/xlsx_parser.py:240,266`) via `openpyxl`/`calamine` — first row = headers, trailing empties trimmed, `~$` skipped.
2. **Normalize + schema** (`xlsx_parser.py:87,92`): `re.sub(r"[\s_\-]+","", lower)` then first schema with `overlap ≥60%` wins (`reason_codes` 14, `crm_qa` 5, `articles` 10). UI shows per-sheet `normalized`, `matched/missing`, `threshold` table.
3. **Rows → fields** (`xlsx_parser.py:311,346`): `_format_cell` (`None→""`, float `g`-format, else `str.strip()`), empty rows dropped, `U+FFFD`/`?+Arabic` integrity warnings. QA rows need `question` + (`answer`|`briefanswer`) else `skipped_incomplete`; dedup via `chunker/semantic.py:110` (`ي→ی, ك→ک, ZWNJ→space`).
4. **Preprocess** (`preprocessor/pipeline.py:354`) clean → Persian normalise → keywords.
5. **Chunk** (`chunker/semantic.py:183` `_chunk_excel_rows`): **one row = one chunk** (never split) with Persian labels `سوال/پاسخ کوتاه/پاسخ کامل/کلیدواژه‌ها`; parents per `parent_scope=sheet`.

Rendering is Persian-clean: `base.html:4` `<meta charset="UTF-8">`, Vazirmatn CDN + `class="persian" dir="rtl/auto"` on every cell/chunk (`static/style.css:629`), `content-preview pre` line-height 1.9. If a cell shows `?`/`�`, the `integrity_warnings` badge names the exact `sheet/column`.

## Troubleshooting — PowerShell blocked (0x800704ec)

**Error:** `error 2147943660 (0x800704ec) when launching %SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe`
**Meaning:** AppLocker / Software Restriction Policy blocks `powershell.exe` on this machine (`This program is blocked by group policy`). All tools that spawn PowerShell 5.1 (including this agent's `bash` tool) will show `spawn UNKNOWN`.

**Fix:**
- Use **CMD**, not PowerShell: double-click `restart_server.bat` / `push_and_merge.bat` in `kb-manager/` (they use `cmd.exe`/`netstat`/`taskkill`/`python`), or run `cmd` → `cd D:\Code\KB\kb-manager` → `netstat -ano | findstr :8000` → `taskkill /F /PID <pid>` → `python start_server_detached.py`.
- To unblock: `gpedit.msc` → Computer/User Configuration → Windows Settings → Security Settings → Software Restriction Policies / AppLocker → allow `%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe` (requires admin).

**Push without PowerShell:** use `push_and_merge.bat` (CMD) — see below.

## QA-Aware Retrieval Experiment (Isolated, no prod change)

Planned under `components/retrieval_experiments/` (see `docs/QA_RETRIEVAL_EXPERIMENT.md` after Phase 1) — four signals `question_sim` + `answer_sim` + `keyword_overlap` + `category_match`; modes **A** prod BM25+semantic, **B** QA-only, **C** hybrid RRF/weighted. Evaluation on `TestQuestions_IVA` (15 Q) + `data/test_questions.json` reports `Recall@1/5, MRR, NDCG, groundedness, hallucination, latency` + `failure_analysis.md` (missed/wrong/hallucinated/guardrail FP). Merge only if `Recall@5 ↑` and `hallucination ¬↑` and `p95` acceptable.

## Roadmap & Remediation — Updated for v7 + Transparency

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
