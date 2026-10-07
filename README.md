# Work RAG KB

Persian knowledge-base lifecycle, hybrid retrieval (RAM-first, Rust tantivy, GPU), and KB web UI for the Work Credit RAG platform.

> Status: **implemented, operational, RAM-first.** The retrieval pipeline and Web UI run from `kb-manager/` (see [kb-manager/README.md](kb-manager/README.md) for current benchmark numbers and quick start). Current release **v13_1405-07-06** — 18 docs / 1,123 chunks, full rebuild 2026-10-07 (`pgvector` HNSW 384d, vectors + HNSW fully in RAM `shared_buffers=16GB`, Rust tantivy BM25 50 MB RAM, GPU dense/rerank on H200). Previous **v11_1405-06-23** — 35 docs / 2,227 chunks, Hit@5 0.9306 (see [kb-manager/README.md#version-history](kb-manager/README.md#version-history) for full history). Top-level docs ([`KB_ARCHITECTURE.md`](KB_ARCHITECTURE.md), [`retrieval-evaluation-research.md`](retrieval-evaluation-research.md)) hold the architecture plan and IR theory; new latency plots below live in `kb-manager/data/plots/`.

## 1. Summary

The Knowledgebase component owns **source documents, ingestion, indexing, retrieval, reranking, and KB evaluation**. It does **not** own model lifecycle, safety policy, conversation orchestration, or the frontend.

Core capabilities (RAM-first, v13):

- Ingest Persian `XLSX` / `PDF` / `DOCX` into a versioned KB (pgvector HNSW 384d, **all vectors + HNSW in RAM** `shared_buffers=16GB` `effective_cache_size=800GB` + `pg_prewarm`; SQLite fallback).
- Semantic chunking tuned to ICS: **QA pairs**, **reason codes**, **articles**, section headings, parent chunks, QA dedup.
- Embed with `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, **GPU** `KB_EMBED_DEVICE=cuda`, H200 3 GB HBM) and cache to `.npz`.
- **Hybrid retrieval**: **Rust tantivy BM25** (RAM, 50 MB heap, whitespace over Persian 3-gram stream) + dense (GPU pgvector HNSW) → RRF fusion k=60 → cross-encoder rerank (`BAAI/bge-reranker-v2-m3`, **GPU**) — **55 ms warm** (see [Latency](#latency--ram-first--rust-tantivy) with plots).
- Web UI (dashboard, documents, chunks, pipeline, search, benchmarks, versions, cleanup, monitoring) on port `8000`.
- Benchmark + evaluation harness (120 frozen queries, 6 formats; v11 Hit@5 0.9306, v13 pending re-benchmark on 1405-07-06).

The full architecture & processing plan (data taxonomy, pgvector schema, preprocessing, chunking, embedding, hybrid retrieval, monitoring, versioning, CI/CD, roadmap) is in [`KB_ARCHITECTURE.md`](KB_ARCHITECTURE.md).

---

## 2. Repository layout

```text
components/knowledgebase/
├── KB_ARCHITECTURE.md            # 2k-line architecture + processing plan
├── retrieval-evaluation-research.md  # IR metrics / frameworks / evaluation
└── kb-manager/                   # the main package
    ├── README.md                 # quick start + current retrieval metrics
    ├── pyproject.toml
    ├── run_server.py / start_server.py
    ├── Dockerfile / docker-compose.yml
    ├── kb_manager/
    │   ├── cli.py                # ingest, status, search, serve, inspect, eval-*
    │   ├── config.py             # env-var config (DB, embedding, chunking, parser)
    │   ├── dense.py / reranker.py
    │   ├── query_reform.py       # HyDE, multi-query, RRF fusion
    │   ├── preprocessor/ chunker/ embedder/ pipeline/
    │   └── web/
    │       ├── app.py            # FastAPI app + index prewarm
    │       └── routes/           # documents, chunks, pipeline, versions,
    │                             #   monitoring, search, benchmarks, cleanup
    ├── tests/                    # 17 tests
    ├── evaluation/               # benchmark, generators, metrics, datasets, famteb
    └── synthetic_generation/     # synthetic Persian QA/conv generation
```

---

## 3. Retrieval architecture (RAM-first, v13)

### 3.1 Four-stage hybrid pipeline — all stages on RAM/GPU (55 ms)

```mermaid
flowchart LR
    Q[Persian query] --> N[Persian normalization<br/>Arabic→Persian chars, ZWNJ, digits]
    N --> TOK[tokenize: words + char 3-grams<br/>~50 tokens + 100 3-grams]
    TOK --> BM[tantivy BM25 (Rust, RAM)<br/>heap 50 MB, k1≈1.2 b=0.75<br/>~4 ms, 8%]
    N --> DE[Dense semantic (GPU)<br/>MiniLM-L12-v2 384-dim<br/>pgvector HNSW 4.8 MB RAM<br/>~24 ms, 44%]
    BM --> F[RRF fusion<br/>k=60, ~0.1 ms]
    DE --> F
    F --> CE[Cross-encoder rerank (GPU)<br/>BGE-v2-m3, top-50<br/>~24 ms, 44%]
    CE --> TOP[Top-K final_results<br/>55 ms total]
    Q -.->|Redis cache TTL 600s<br/>2 ms hit, 150×| RC
    PG[(Postgres<br/>16GB shared_buffers<br/>800GB effective<br/>pg_prewarm 750 blks)] -.-> DE
```

> Host: 1.0 TiB RAM (867 Gi free) · 2× H200 143 GB HBM · `tantivy==0.26.2` · `KB_FAST_CACHE=true` eliminates 2.7 s fingerprint scan · `KB_EMBED_DEVICE=cuda` `KB_RERANKER_DEVICE=cuda`.

![Latency breakdown before vs after (log scale)](kb-manager/data/plots/latency_breakdown_ram_tantivy.png)

![Stage share after (55 ms)](kb-manager/data/plots/latency_breakdown_pie_after.png)

### 3.2 Ingestion pipeline

```mermaid
flowchart TD
    SRC[source files: XLSX / PDF / DOCX] --> PARSE[parsers: reason_codes / crm_qa / articles]
    PARSE --> HASH[content_hash]
    HASH --> CHUNK[semantic chunks:<br/>QA-pairs, reason-codes, articles, body]
    CHUNK --> DEDUP[skip incomplete QA + dedup questions]
    DEDUP --> EMBED[embed batch MiniLM-L12 384-d]
    EMBED --> PARENT[parent chunks per sheet/document]
    PARENT --> STORE[store chunks + DocumentVersion]
    STORE --> GATE[QualityGate validation]
```

### 3.3 Key parameters (v13 RAM-first)

| Stage | Param | Value | Where |
|---|---|---|---|
| Postgres | `shared_buffers` / `effective_cache_size` | **16 GB** / **800 GB** + `pg_prewarm` (chunks 152 + HNSW 598 blks RAM) | `docker exec rag-postgres psql -c "ALTER SYSTEM SET …"` + restart |
| BM25 | engine | **Rust tantivy** 0.26.2, RAMDirectory heap 50 MB, `whitespace` over Persian 3-gram stream | `kb_manager/tantivy_bm25.py` |
| BM25 | k1 / b / keyword boost / n-grams | 1.5 / 0.75 / ×3 / 3 | `search.py:268` / `tantivy_bm25.py` |
| BM25 | beam | 5 (set `KB_SYNONYM_BEAM=1` to cut 5×) | `query_expansion.py:238` |
| Dense | model / dim / batch / device | `paraphrase-multilingual-MiniLM-L12-v2` / 384 / 64 / **cuda** (H200 3 GB) | `KB_EMBED_DEVICE=cuda` |
| Dense | index | pgvector HNSW `vector_cosine_ops` m=16 ef=64, **RAM** (4.8 MB) + npz cache | `models/database.py` |
| RRF | k | 60 | `search.py:754` |
| Rerank | model / pool / device | `BAAI/bge-reranker-v2-m3` / top-50 (100 in bench) / **cuda** (H200) | `KB_RERANKER_DEVICE=cuda` `reranker.py:268` |
| Cache | `KB_FAST_CACHE` | **true** (skip 2.7 s fingerprint scan, invalidation via restart) | `search.py:523` |
| Redis | TTL | 600 s, 2–7 ms hits, 150× | `KB_REDIS_URL` |
| Chunk | strategy / max tokens | semantic / 512 | |
| Chunk | parent max / scope | 1536 / sheet | |

### 3.4 Latency — RAM-first + Rust tantivy (1405-07-06, 1110 chunks, measured 2026-10-07)

**Before (Python BM25, CPU, slow fingerprint, warm):** BM25 ~3000 ms (97%), dense ~24 ms, rerank ~24 ms (CPU would be 5400 ms cold), **total ~3100 ms** warm, **12 s cold** (dense 4246 + rerank 5396 first query). **After (tantivy, GPU, fast cache):** BM25 3.8–5.9 ms, dense 22–27 ms, rerank 21–25 ms, **total 51–66 ms avg 55 ms** (`POST /search/api` `stage_ms`, `curl --noproxy "*" http://127.0.0.1:8000/search/api -d '{"query":"اعتبارسنجی چیست","top_k":5}'`).

| Stage | Before | After | Share after |
|-------|--------|-------|-------------|
| BM25 lexical (content+kw, beam5) | ~3000 ms | **4.5 ms** | 8% |
| Dense cosine (pgvector HNSW, GPU) | 24 ms (CPU 3000) | **24.5 ms** | 44% |
| RRF fusion k=60 | 0.2 ms | 0.15 ms | <1% |
| Cross-encoder rerank (BGE-v2-m3, GPU) | 24 ms (CPU 5400) | **24.5 ms** | 44% |
| **Total warm** | **3100 ms** | **55 ms** | **100%** |
| Redis hit | 2–7 ms | 2 ms | 150× vs cold |
| Cold first query | 12 s | 12 s one-time (model load) | — |

![Improvement factor (log scale)](kb-manager/data/plots/improvement_factor.png)

![Postgres vectors fit entirely in RAM](kb-manager/data/plots/postgres_ram_fit.png)

**Does Postgres save vectors on RAM and leverage GPU?** No GPU — `pgvector` HNSW (`chunks_embedding_hnsw 4.8 MB`, `chunks 1.2 MB`, `pg_total 203 MB` for 1110 chunks) lives on disk but is **fully cached in RAM** via `shared_buffers=16GB` + `effective_cache_size=800GB` (host 1.0 TiB, 867 Gi free) + `pg_prewarm(152+598 blocks)`. HNSW distance (`vector_cosine_ops`) is **CPU-only**. Dense and reranker are **GPU** (H200, 3 GB HBM, `torch.cuda.is_available()`).

**Is cross-encoder on GPU?** Yes after `KB_RERANKER_DEVICE=cuda` (`reranker.py:274` `model.to(actual_device)` `dtype=float16` on cuda, `nvidia-smi` 3006 MiB for KB PID). Before: 5.4 s CPU → 24 ms GPU. Same for dense (`KB_EMBED_DEVICE=cuda`).

### 3.5 Design choices & improvements (why RAM-first)

| Choice | Alternative considered | Why chosen | Cost if not chosen |
|--------|------------------------|------------|-------------------|
| **Rust tantivy** `tantivy_bm25.py` heap 50 MB RAM | Python BM25 (`BM25` class `search.py:280`) | 500× (3000→5 ms), Persian 3-gram stream preserved via whitespace tokenizer over pre-tokenized text | BM25 dominates 97% latency |
| **`KB_FAST_CACHE=true`** skip fingerprint | Full `SELECT *` + `DenseSemanticIndex.fingerprint` per search (2.7 s) | Removes per-search DB scan; invalidation via restart after ingestion | Every query pays 2.7 s |
| **Postgres `shared_buffers=16GB` + `pg_prewarm`** | 128 MB default (5.8% of 203 MB) | Whole DB in shared buffers, no disk miss; `effective_cache_size=800GB` guides planner | Cold HNSW miss → disk 10 ms |
| **GPU dense + rerank** `cuda` | CPU `MiniLM` + `BGE` (5.4 s rerank) | 200× rerank, 100× dense; H200 143 GB HBM holds both (3 GB) | Total 12 s cold |
| **pgvector HNSW `m=16 ef=64`** | IVFFlat or brute numpy | RAM, <25 ms, 4.8 MB, no external Qdrant (Qdrant hybrid `qdrant_hybrid.py` kept as future option) | Qdrant extra service |
| **Redis TTL600 + in-memory** | No cache | 2 ms hits, 150× for repeat queries (same `sha256(query|top_k|boost)`) | Repeat pays 55 ms |

---

## 4. API endpoints (KB Manager)

### Web UI pages

| Path | Page |
|---|---|
| `/` | Dashboard |
| `/documents` | Documents (with **Inspect → Transparency** per row) |
| `/chunks` | Chunks |
| `/pipeline` | Pipeline (run ingestion) |
| `/search` | Search page |
| `/transparency` | **Transparency — Excel → Chunks (NEW)** — raw table + schema debug + Persian RTL (Vazirmatn) + live `.xlsx` upload + `GET /transparency/api/raw/{id}` JSON |
| `/benchmarks` | Benchmarks |
| `/benchmarks/comparison` | Version comparison (v2→v7 data-driven) |
| `/versions` | Versions |
| `/cleanup/qa` | QA cleanup |
| `/monitoring` | Monitoring |

### JSON APIs

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/search/api` | **Hybrid search** — returns step-by-step `SearchSteps` + `final_results` |
| `GET` | `/documents` | List docs (filters + pagination) |
| `GET` | `/documents/{doc_id}` | Doc detail with chunks |
| `POST` | `/documents/upload` | Upload file (creates draft doc) |
| `POST` | `/documents/{doc_id}/delete` | Soft-delete (archive) |
| `GET` | `/chunks` / `/chunks/{chunk_id}` | List / detail chunks |
| `POST` | `/chunks/{id}/verify` / `/edit` | Verify / edit chunk |
| `GET` | `/pipeline` / `POST /pipeline/run` / `GET /pipeline/status/{job_id}` | Pipeline control |
| `GET` | `/versions` / `POST /versions/{doc_id}/rollback/{version_id}` | Version history + rollback |
| `GET` | `/monitoring/staleness`, `/monitoring/metrics` | Staleness + metrics |
| `POST` | `/benchmarks/run`, `GET /benchmarks/status/{job_id}`, `GET /benchmarks/result` | Benchmark runner |
| `GET` | `/benchmarks/comparison/data`, `/benchmarks/snapshots*` | Comparison + snapshots |

### Search response (`/search/api`)

```jsonc
{
  "final_results": [
    {
      "chunk_id": "chunk-123",
      "doc_id": "doc-1",
      "doc_title": "گزارش اعتباری",
      "heading_path": "فصل ۲ / گزارش",
      "content_preview": "… متن کوتاه …",
      "rerank_score": 0.87,          // or hybrid_score
      "hybrid_score": 0.81
    }
  ]
}
```

The orchestrator maps these KB-native field names to its internal `KBRetrievalResult`.

---

## 5. CLI commands

| Command | Purpose |
|---|---|
| `kb-manager ingest [-s DIR] [--full] [-m MODEL] [--parent-scope sheet\|document]` | Ingest (full or incremental) |
| `kb-manager status` | Doc/chunk counts (rich table) |
| `kb-manager search -q "..." -k 5` | Vector search (cosine, `<=>`) |
| `kb-manager serve` | Start uvicorn web server (reload) |
| `kb-manager inspect -f file.xlsx` | Inspect file structure |
| `kb-manager eval-generate [-n N]` | Generate synthetic eval dataset |
| `kb-manager eval-run -i dataset -k N` | Run retrieval evaluation |
| `kb-manager status-chunks` | Chunk statistics (types, tokens, QA completeness) |

---

## 6. Configuration (environment variables)

| Env var | Default | Purpose |
|---|---|---|
| `KB_DB_URL` | `sqlite+aiosqlite:///./data/kb_test.db` | DB URL; prod `postgresql+asyncpg://postgres:postgres@127.0.0.1:5433/kb_manager` (RAM: `shared_buffers=16GB` `effective_cache_size=800GB` + `pg_prewarm`) |
| `KB_DB_HOST/PORT/NAME/USER/PASSWORD` | `localhost/5432/kb_manager/postgres/postgres` | pgvector target (prod `5433`) |
| `KB_EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | embedding model (GPU `KB_EMBED_DEVICE=cuda` 24 ms) |
| `KB_EMBED_DIM / BATCH` | `384 / 64` | embedding params |
| `KB_EMBED_DEVICE` / `KB_RERANKER_DEVICE` | `cpu` | `cuda` on H200 (3 GB HBM) |
| `KB_USE_TANTIVY` | `true` | Rust tantivy RAM BM25 (50 MB heap) vs Python fallback |
| `KB_FAST_CACHE` | `true` | skip 2.7 s fingerprint scan per search |
| `KB_SYNONYM_BEAM` | `5` | 1 to cut 5× |
| `KB_CHUNK_STRATEGY / MAX` | `semantic / 512` | chunking |
| `KB_CHUNK_PARENT_MAX / SCOPE` | `1536 / sheet` | parent chunks |
| `KB_SOURCE_DIR` | `./kb-source` | version folder `1405-07-06` (18 docs) |
| `KB_WEB_HOST / PORT` | `0.0.0.0 / 8000` | web server |
| `KB_LLM_BACKEND` | `mock` | `mock`/`openai`/`ollama`/`vllm` (HyDE) |

---

## 7. Database

- **Runtime:** SQLite (`sqlite+aiosqlite`, ~2.7 GB corpus in dev) — used by the web server and tests.
- **Production target:** PostgreSQL + pgvector (`postgresql+asyncpg`), provisioned by `docker-compose.yml` (`scripts/init_db.sql` enables `vector`, `pg_trgm`, `uuid-ossp`).
- ORM models (SQLAlchemy 2.0 async): `Document`, `Chunk` (self-FK `parent_id`, `embedding_model`, `quality_score`, `is_verified`), `DocumentVersion`, `IngestionJob`, `RetrievalLog`.

```bash
cd kb-manager
docker compose up -d   # pgvector on :5432 + kb-manager on :8000
```

---

## 8. Tests

| File | Covers |
|---|---|
| `test_chunker.py` | semantic chunking, incomplete-QA skip, parent scope, QA Persian field formatting, FixedChunker |
| `test_parsers.py` | XLSX (reason_codes, crm_qa), DOCX, parser registry |
| `test_preprocessor.py` | Persian normalization, HTML/URL/whitespace, pipeline quality |
| `test_pipeline.py` | orchestrator scan (full vs incremental) |
| `test_embedder.py` | dims, query embed, content-hash cache |
| `test_evaluation.py` | Ranx vs pure-Python fallback, Ragas availability |
| `test_cli.py` | version, inspect |

```bash
cd kb-manager
pytest tests/ -v
```

---

## 9. Run

```bash
cd components/knowledgebase/kb-manager
pip install -e ".[dev]"
python run_server.py          # port 8000 (prewarms BM25 + dense + reranker, ~30-60s)
python -m kb_manager.cli ingest --full   # (once) build the KB from kb-source
# or: docker build -t work-rag-kb . && docker run -p 8000:8000 work-rag-kb
```

---

## 10. Persian resources

[`kb-manager/PERSIAN_RESOURCES.md`](kb-manager/PERSIAN_RESOURCES.md) catalogs Persian NLP options: FaMTEB benchmark suite, embedding models (current MiniLM + candidates like BGE-M3, ParsBERT, FaBERT), cross-encoders, libraries (Hazm, Parsivar, Persian-tools), ZWNJ handling, and evaluation metrics.

---

## 11. Planning & progress checklist

### Done (MVP)

- [x] Persian normalization + extraction (XLSX / PDF / DOCX)
- [x] Semantic chunking (QA pairs, reason codes, articles) + parent chunks
- [x] Skip incomplete QA + dedup by normalized question
- [x] Dense embedder (MiniLM-L12v2, 384-d) with content-hash cache
- [x] BM25 (char 3-grams, keyword boost) + RRF(k=60) + cross-encoder rerank
- [x] `/search/api` normalized response + Web UI (port 8000)
- [x] Pipeline orchestration (full rebuild vs incremental) + versioning + rollback
- [x] Benchmark harness (v5: 120 frozen queries, Hit@5 84.2%, MRR 0.751)
- [x] Dockerfile + docker-compose (SQLite runtime / pgvector target)
- [x] Test suite (17 tests) passing

### Done (v13 RAM-first, 2026-10-07)

- [x] pgvector production migration + HNSW `vector_cosine_ops` + `pg_prewarm` (RAM: 16GB shared_buffers, 203 MB total relation)
- [x] Rust tantivy BM25 RAM (50 MB heap, 500×) + `KB_FAST_CACHE=true` (skip 2.7 s fingerprint)
- [x] GPU dense + rerank on H200 (24 ms each, 3 GB HBM, `KB_EMBED_DEVICE=cuda` `KB_RERANKER_DEVICE=cuda`)
- [x] Latency 3100 ms → 55 ms (plots `kb-manager/data/plots/latency_breakdown_ram_tantivy.png` etc.)
- [x] Version `1405-07-06` ingestion (18 docs / 1123 chunks, `5741927`)

### Next / open

- [ ] Wire HyDE / multi-query reformulation end-to-end (default off)
- [ ] Contextual retrieval live toggle wiring
- [ ] Re-generate synthetic eval dataset with current KB (regen script) + re-benchmark Hit@5 on 1405-07-06
- [ ] Staleness → auto-reingest cron/pipeline integration
- [ ] CI/CD for KB evaluation runs on every ingest
  - [x] Persist benchmark comparison plots as immutable snapshots (`latency_breakdown_ram_tantivy.png`, `improvement_factor.png`, `postgres_ram_fit.png`)

---

## License

See parent repository `LICENSE` and the submodule's own obligations (`kb-manager` is private — ICS Credit Scoring).
## Metadata Filtering
The pipeline extracts the folder hierarchy from `kb-source` and attaches it as `folder_hierarchy` metadata to all chunks. Use the `filter_path` argument in the search API to restrict retrieval to specific directories (e.g. `filter_path='اشخاص حقوقی'`).

## Supplementary Data
See `kb-manager/data/supplementary_architecture.md` for how `ضمیمه پایگاه دانش` files are handled via semantic injection instead of dense embedding.