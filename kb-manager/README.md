# KB Manager — Persian RAG Knowledge Base

> **ICS Credit Scoring Knowledge Base** — ingest, chunk, version, and serve a Persian-language corpus (credit reports, cheque scoring, CRM Q&A) for RAG agents, with hybrid retrieval (BM25 + dense + RRF + cross-encoder rerank), stage-level benchmarks, and a FastAPI operations UI.

- **Current KB:** `v13_1405-07-06` — 18 docs / 1,123 chunks, ingested 2026-10-07 (full rebuild, `pgvector` HNSW 384d, all vectors in RAM, tantivy RAM BM25, GPU rerank), Hit@5 pending re-benchmark (v11 was 0.9306 on 35 docs / 2,227 chunks)
- **Stack:** FastAPI · SQLAlchemy (async) · SQLite / pgvector (HNSW, RAM-pinned) · Rust tantivy BM25 (RAM) · sentence-transformers (GPU) · cross-encoder `BAAI/bge-reranker-v2-m3` (GPU) · hazm/shekar (Persian) · Click + Rich CLI · Redis result cache (TTL 600 s, 150× hit speedup)
- **Validated:** `compileall` clean · `pytest` **116 passed**, 6 skipped, 1 pre-existing env-dependent failure (2026-10-01; `test_qa_massive_count` — newest `kb-source/1405-*` folder exposes 4 QA files < 5)
- **RAM-first:** Postgres `shared_buffers=16GB` `effective_cache_size=800GB` `pg_prewarm` (chunks + HNSW in shared buffers), tantivy 50 MB heap RAM, dense `.npz` + reranker weights on H200 GPU (1 TiB host RAM, 2×143 GB HBM) — see [Latency](#latency)

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [API & UI](#api--ui)
- [Version history](#version-history)
- [Architecture evolution](#architecture-evolution)
- [Benchmarks & metrics](#benchmarks--metrics)
- [Latency](#latency)
- [LLM-as-judge feedback & root causes](#llm-as-judge-feedback--root-causes)
- [Evaluation reports](#evaluation-reports)
- [Testing on Open WebUI — end to end](#testing-on-open-webui--end-to-end-rag--filter_path--clarifying)
- [Development](#development)

## Features

- **Multi-format ingest** — `.xlsx` (openpyxl/calamine), `.docx`, `.pdf` (PyMuPDF) via a parser registry (`kb_manager/parsers/`), orchestrated full/incremental rebuilds (`kb_manager/pipeline/orchestrator.py`) with quality gates (`pipeline/quality.py`) and versioning (`pipeline/versioning.py`).
- **Semantic chunking** — `semantic` / `fixed` strategies (`kb_manager/chunker/`); QA-pair, article, kv-pair, reason-detail, single-col-list and staff-profile schemas with parent chunks, question dedup, and overlap control.
- **Hybrid retrieval** — Rust tantivy BM25 (RAM, 50 MB heap, whitespace tokenizer over Persian word+3-gram stream) + dense MiniLM-L12 cosine index (numpy `.npz` cache + pgvector HNSW, both RAM-pinned) → RRF fusion k=60 → cross-encoder rerank (`kb_manager/tantivy_bm25.py`, `kb_manager/dense.py`, `kb_manager/reranker.py`, `kb_manager/web/routes/search.py`), with Redis + in-memory result cache (`KB_REDIS_URL`, TTL 600 s, 2–7 ms hits).
- **RAM-first** — Postgres vectors + HNSW fully in `shared_buffers` (16 GB) + `effective_cache_size` (800 GB) + `pg_prewarm`; tantivy in RAMDirectory; dense + reranker on H200 GPU (`KB_EMBED_DEVICE=cuda` `KB_RERANKER_DEVICE=cuda`, `nvidia-smi` 3 GB); host 1.0 TiB RAM, 2×143 GB HBM.
- **Persian query path** — normalization (`preprocessor/persian.py`, `regex_persian.py`), synonym/beam expansion (`query_expansion.py`, 93-entry map incl. colloquial→formal + v12 typo/exclusive-term gates), reform/enhance stages, optional HyDE (`hyde.py`, off by default).
- **Operations UI** — dashboard, transparency views (RTL Persian), 5-step ingestion wizard, benchmarks, chunk/document browsers, zip browser, monitoring (`kb_manager/web/routes/`, 12 routers).
- **Stage-level evaluation** — every gold row classified A / A′ / B / D with per-stage top-k lists dumped to JSONL, plus a live TUI dashboard and LLM-judge corpus tooling (`artifacts/retrieval_training/`).

## Architecture

### Ingest pipeline

```mermaid
graph LR
    SRC[kb-source version folder xlsx / docx / pdf] --> SCAN[orchestrator._scan_files TestQuestion* excluded]
    SCAN --> PARSE[parsers registry xlsx / docx / pdf]
    PARSE --> PRE[preprocessor pipeline clean + persian + validators]
    PRE --> CHUNK[semantic chunker qa / article / kv / reason-detail parents + question dedup]
    CHUNK --> QUAL[quality gates]
    QUAL --> DEDUP[dedup.py question + MinHash LSH]
    DEDUP --> EMB[embedder + dense index MiniLM-L12 384d, npz cache]
    EMB --> DB[(SQLite aiosqlite or pgvector HNSW)]
```

### Retrieval pipeline (per query) — RAM/GPU

```mermaid
graph LR
    Q[Persian query] --> QX[query enhance / expansion beam5 / reform HyDE optional, off by default]
    QX --> BM25[tantivy BM25 (Rust, RAM) keyword x3 + char 3-grams ~4 ms, 50 MB heap]
    QX --> DENSE[Dense cosine (GPU) MiniLM-L12 384d in-RAM cosine ~14 ms, batched fp16]
    BM25 --> RRF[RRF fusion k=60 merged top-100 ~0.1 ms]
    DENSE --> RRF
    RRF --> RERANK[cross-encoder rerank (GPU) BGE-v2-m3, 100-pool ~24 ms, memoized]
    RERANK --> TOP[final top-5 + scores ~40 ms total]
    Q -.->|cache lookup / store TTL 600 s, 2-7 ms hits errors never cached| RC[(Redis + in-memory KB_REDIS_URL)]
```

### Query Classification & Ambiguity Resolution (v13.1)

The pipeline now includes a **3-way LLM classifier** (`nodes/classify.py`) that routes every query into one of three categories before retrieval, plus an **Mizan System-1 + Gemma hybrid ambiguity resolver** (`mizan.py` + `mizan.py:234-280`) for subspace routing:

```mermaid
flowchart TD
    Q[Persian query] --> CLASSIFY[3-way LLM classifier greeting / specification / question]
    CLASSIFY -->|greeting| GREET[return greeting no retrieval]
    CLASSIFY -->|specification| SPEC[extract subspace cheque/facilities_person/legal/general]
    CLASSIFY -->|question| T5[is_scattered_ambiguous?]
    T5 -->|no| RETRIEVE[retrieve + rerank]
    T5 -->|yes| AMBIG[Mizan System-1 subspace routing + ambiguous flag]
    AMBIG -->|confident| RETRIEVE
    AMBIG -->|ambiguous| CLARIFY[ask clarifying question چک / تسهیلات / شرکت / عمومی]
    CLARIFY --> SPEC
    SPEC --> RETRIEVE
```

**Classification categories:**
- **greeting**: سلام، تشکر، خداحافظی — no KB lookup
- **specification**: user names a subspace (چک، تسهیلات، شرکت، عمومی) — locks subspace
- **question**: real query needing retrieval — passes to ambiguity resolver

**Ambiguity resolver (Mizan System-1 + Gemma hybrid):**
- **Mizan System-1** (`mizan.py:234-280`): calibrated subspace routing with choice probabilities + ambiguous flag
- **Gemma LLM** (`infer_specification.py`): scattered detection + clarifying question generation
- **Fallback**: heuristic keyword-based subspace detection when LLMs unavailable

```mermaid
flowchart LR
    Q[Question] --> MIZAN[Mizan System-1 subspace routing + ambiguous flag]
    MIZAN -->|confident| ROUTE[route to subspace]
    MIZAN -->|ambiguous| GEMMA[Gemma LLM scattered detection]
    GEMMA -->|scattered| CLARIFY[ask clarification]
    GEMMA -->|not scattered| ROUTE
```

### Knowledge tree — `folder_hierarchy` + supplements (`v13_1405-07-06` → 18 docs / 1,123 chunks)

Source ZIP hierarchy → `Document.doc_metadata.folder_hierarchy: list[str]` (`database.py:81`) → per-chunk `cd[7]` at `search.py:438`. Leaf `allowed_ids` powers `POST /search/api` `filter_path` (`search.py:634`; tester at `search.html:127`). Chunk types in this KB: `qa_pair 776 · reason_detail 198 · body 71 · article 46 · kv_pair 19` (+ 13 parent chunks).

```text
پایگاه دانش
├── اشخاص حقیقی
│   ├── اعتبارسنجی تسهیلات
│   │   ├── بانک سوالات (Q&A)
│   │   │   ├── بانک سوالات افراد در زمینه گزارش اعتبارسنجی تسهیلات  (337 chunks, q/a schema)
│   │   │   ├── بانک سوالات افراد — نسخۀ اصلی  (244, q/a)
│   │   │   ├── بانک سوالات — جایگزین  (0 chunks, skeleton)
│   │   │   └── — also credit-side alt docs above carry their own QA slots
│   │   ├── توضیحات تشریحی — بخش‌های گزارش اعتبارسنجی افراد  (37, article)
│   │   └── توضیحات تشریحی — دلایل کاهش امتیاز
│   │       ├── مدل اصلی و روش بهبود  (80, reason_detail)
│   │       └── مدل جایگزین و روش بهبود  (11, reason_detail)
│   └── گزارش اعتبارسنجی چک
│       ├── بانک سوالات مباحث مختلف چک  (118, q/a)
│       ├── توضیحات تشریحی بخش‌های مختلف گزارش چک  (12, article)
│       ├── نکات قابل توجه چک  (10, article/kv)
│       └── توضیحات تشریحی دلایل کاهش امتیاز چک و روش بهبود  (87, reason_detail)
├── اشخاص حقوقی
│   └── اعتبارسنجی تسهیلات
│       ├── رسا — بخش‌های مختلف گزارش اعتباری شرکتها  (24, article)
│       ├── توضیحات بخش‌های مختلف گزارش اعتباری شرکتها  (16, article)
│       └── توضیحات تشریحی دلایل کاهش امتیاز اعتباری شرکتها  (24, reason_detail)
└── پایگاه دانش عمومی  (no 3rd-level leaf — filter by leaf title)
    ├── معرفی شرکت اعتبارسنجی ایران  (4, article/body)
    ├── هیات مدیره  (5, article)
    ├── 16 کاربرد اعتبارسنجی در اقتصاد  (17, article)
    ├── سوالات عمومی  (81, q/a + body)
    └── لینک‌ها  (16, article/kv_pair)

ضمیمه / Supplements — `EntityList` supplementary tables (`database.py:249`, NOT `Chunk`s)
  (on-disk XLSX list sheets → entity_lists · search-time injected at `search.py:960` step 8:
   query→MiniLM-L12 cosine vs EntityList.description_embedding, threshold 0.6
   → SearchSteps.injected_entities[{list_name, description, content: content_json, similarity}]
   POST /search/api returns them; orchestrator wires this next as rag.supplemental)
  ├── استفاده کنندگان گزارش  — {"Sheet1": [[شرکت راهکارهای مالی هوشم...
  ├── تامین‌کنندگان دیتا  — {"Sheet1": [[صندوق کارآفرینی امید, صندو...
  ├── سهامداران  — {"Sheet1 (2)": [[بانک رفاه کارگران, بانک‌…
  ├── لیست و عناوین وام ها  — {"وام‌ها": [[اسنپ پی, خرید اقساطی], …
  └── Word Document  — {} (empty placeholder)
```

*How to use a branch:* KB tester `filter_path: "شاخه/چک"` (last version, `search.html:127`) now yields only that subtree (`search.py:634` `allowed_ids` = `{chunk_id | filter_path in "/".join(hierarchy)}`). The Open WebUI scope picker + clarifying question in the next milestone will reuse the same `filter_path` key on `POST /v1/chat/completions` (`schemas.py` → `knowledgebase.py` → `retrieve.py:141`), with each chunk rendered as `from which part (which tree leaf) + which content` plus a `rag.supplemental: EntityList[]` envelope for the list rows.

#### Diagrams — detail tree (UTF-8 RTL friendly, mermaid)

<details>
<summary>1) Folder hierarchy — 18 docs / 1,123 chunks (v13, <code>folder_hierarchy</code>)</summary>

```mermaid
flowchart TB
    Root["پایگاه دانش (18 docs, 1123 chunks)"]

    subgraph T1["اشخاص حقیقی — اعتبارسنجی تسهیلات"]
        direction TB
        QA["بانک سوالات Q&A 337 + نسخۀ اصلی 244 + جایگزین 0"]
        DESC_A["توضیح بخش‌های گزارش افراد 37 (article)"]
        REASON_A["دلایل کاهش امتیاز مدل اصلی 80 + جایگزین 11 (reason_detail)"]
    end

    subgraph T2["اشخاص حقیقی — گزارش چک"]
        direction TB
        QA_C["بانک سوالات مباحث چک 118 (q/a)"]
        DESC_C1["بخش‌های مختلف گزارش چک 12 (article)"]
        NOTE_C["نکات قابل توجه چک 10 (article/kv)"]
        REASON_C["دلایل کاهش امتیاز چک 87 (reason_detail)"]
    end

    subgraph T3["اشخاص حقوقی — اعتبارسنجی تسهیلات"]
        direction TB
        CORP_A["رسا — بخش‌های گزارش اعتباری شرکتها 24 (article)"]
        CORP_B["بخش‌های مختلف شرکتها 16 (article)"]
        CORP_REASON["دلایل کاهش امتیاز شرکتها 24 (reason_detail)"]
    end

    subgraph T4["پایگاه دانش عمومی — بدون برگه سوم"]
        direction TB
        CORP_INTRO["معرفی شرکت 4"]
        BOARD["هیات مدیره 5"]
        P16["16 کاربرد اعتبارسنجی 17"]
        PUB_QA["سوالات عمومی 81 (q/a + body)"]
        LINKS["لینک‌ها 16 (kv_pair)"]
    end

    Root --> T1
    Root --> T2
    Root --> T3
    Root --> T4
```

</details>

<details>
<summary>2) Supplements — 5 EntityList (ضمیمه، NOT chunks)</summary>

```mermaid
flowchart LR
    XLSX["XLSX sheet لیست بانک‌ها / عناوین وام"] --> PARSER["Importer: EntityList database.py:249"]
    PARSER --> TBL["entity_lists list_name + description + description_embedding content_json"]
    TBL --> STEP8["Search step 8 search.py:960 query×MiniLM-L12 cosine vs description_embedding threshold 0.6"]
    STEP8 --> INJ["SearchSteps.injected_entities search.py:989 [{list_name, description, content: content_json, similarity}]"]
    INJ --> API["POST /search/api returned with final_results orchestrator next: rag.supplemental: EntityList[]"]
```

</details>

<details>
<summary>3) Conditional subspace retrieval — <code>filter_path</code> → <code>allowed_ids</code></summary>

```mermaid
flowchart LR
    UI["KB tester search.html:127 filter_path: ش.چک / شاخه گزارش"]
    Q["Query: مثلا چک برگشتی"]
    HIER["Document.doc_metadata.folder_hierarchy database.py:81 → per-chunk cd[7] search.py:438"]
    ALLOWED["allowed_ids = {chunk_id | filter_path in /join(hierarchy)} search.py:634"]
    BM25["tantivy BM25 (content+kw) search(..., allowed_ids) search.py:651"]
    DENSE["Dense (pgvector HNSW) search(..., allowed_ids)"]
    RRF["RRF k=60 + cross-encoder rerank — only leaves under filter_path survive"]

    UI --> ALLOWED
    Q --> BM25
    Q --> DENSE
    HIER --> ALLOWED
    ALLOWED --> BM25
    ALLOWED --> DENSE
    BM25 --> RRF
    DENSE --> RRF
```

</details>

<details>
<summary>4) Idea — clarifying question when retrieval spreads across leaves</summary>

```mermaid
flowchart TB
    Q2["User: مبهم (مثلا تسهیلات و چک هر دو می‌خورند)"]
    RET1["POST /search/api → stripped leaves پس از retrieve.py:141 sources = {شاخه تسهیلات, شاخه چک}"]
    GATE{"چند برگ مختلف؟"}
    CLAR["Orchestrator → Guardrails LLM guarded_generate.py:18 «کدام برگ را ترجیح می‌دهید؟ available_subtrees: [شاخه تسهیلات, شاخه چک]» finish_reason=clarify"]
    NEXT["User specifies: چک → fix filter_path = ش.چک → دومین retrieve"]
    ANSWER["Build context + generate build_context.py:18 → format_response.py:38 rag.supplemental + citation badge"]

    Q2 --> RET1 --> GATE
    GATE -- "یک برگ غالب →" --> ANSWER
    GATE -- "پراکنده →" --> CLAR --> NEXT --> ANSWER
```

</details>

<details>
<summary>5) Idea — Open WebUI scope picker + per-chunk provenance</summary>

```mermaid
flowchart LR
    SEL["WebUI picker # Cheque / گزارش / تسهیلات / ضمیمه (GET /v1/models → folder_hierarchy tree)"]
    BODY["POST /v1/chat/completions schemas.py:9 {query, filter_path: ش.چک}"]
    FILTER["retrieve.py:141 → KnowledgebaseClient knowledgebase.py:74 json={query, top_k, filter_path: body.filter_path}"]
    LEAF["search.py:634 allowed_ids — only ش.چک chunks survive"]
    RENDER["Dashboard / Response renderChunkCard: from which part: ش.چک / Nex name + content 600-char + score bar + citation badge + supplemental envelope rag.supplemental: EntityList[]"]

    SEL --> BODY --> FILTER --> LEAF --> RENDER
```

</details>

> All labels are UTF-8 Persian; keep them quoted in mermaid (`"…"`) and avoid mixing inline `»` with the `mermaid` plugin's RTL quirks by using `پایگاه دانش` as a plain `subgraph` title. Tested on GitHub mermaid rendering (Chrome 120, `v13_1405-07-06`).

### Testing on Open WebUI — end to end (RAG + filter_path + clarifying)

#### Prerequisites

- WebUI at `http://127.0.0.1:13000` (Open WebUI — ICS Helper, NOT Arena — Arena is disabled)
- Tracing fallback `:3000` and KB `:8000` running
- `OPENAI_API_BASE_URL=http://host.docker.internal:8100/v1` + `rag_trace_capture` filter installed (global, is_global True)

#### Smoke (no scope)

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ics-helper-agent","messages":[{"role":"user","content":"امتیاز اعتباری چیست؟"}]}' | jq '.audit.stage_timing_ms, .rag.citations[0] // .audit.retrieved_chunks[0].folder_hierarchy'
curl -s http://127.0.0.1:3000/api/observability/pipeline/{request_id} | jq '.timeline' # same id as trace
```

#### Conditional scope — KB tester

`http://127.0.0.1:8000/search` tester: fill `filter_path: گزارش اعتباری چک` (or `اشخاص حقیقی/اعتبارسنجی تسهیلات`) → `POST /search/api {"query":"...","filter_path":"...","top_k":5}` → only that subtree's `allowed_ids` survive.

#### Conditional scope — Open WebUI

Two equivalent ways (both hit `filter_path`):

*Method A — body extra field (explicit):*

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ics-helper-agent","filter_path":"گزارش اعتباری چک","messages":[{"role":"user","content":"وضعیت من چطور است"}]}' | jq .rag
```

→ `audit.applied_subspace: "cheque"` + leaf-pure `retrieved_chunks[].folder_hierarchy`.

*Method B — scope picker (no API change):*

Open WebUI → chat with **ICS Helper** → Integrations (bottom bar) → `rag trace` chip → chip's `UserValves.scope` dropdown (cheque / facilities person / facilities legal / general / auto) → the `rag_trace_capture` filter's `_inject_scope` writes `body["filter_path"]` (`rag_trace_capture.py:108`) before the request hits the orchestrator — fail-silent if already set.

#### Clarifying flow — scattered + sticky

1. First scattered query (e.g. `وضعیت اعتباری چطور است` mixed across 709 تسهیلات + 227 چک): history insufficient → LLM judge `is_specification` (`nodes/infer_specification.py`) returns `question` → no sticky → `generate_clarification` asks one Persian pill `چک / تسهیلات / شرکت`.
2. Next message `چک` → `is_specification` returns `true` → overwrites `applied_subspace = cheque` (sticky in `MemorySaver` keyed by `chat_id → thread_id`), filtered re-retrieve (`KnowledgebaseClient.retrieve(..., filter_path="گزارش اعتباری چک")`) → generate.
3. Third generic `بدهکار شدم چیکار کنم` → `is_specification: false`, keeps sticky `cheque`, still filtered. Re-specify `شرکت` flips to `facilities_legal`.

Test:

```bash
# 1. scattered -> clarify
curl -s http://127.0.0.1:8100/v1/chat/completions -d '{"model":"ics-helper-agent","messages":[{"role":"user","content":"وضعیت اعتباری چطور است"}]}' | jq '.choices[0].finish_reason' # expect "clarify"
# 2. specify -> filtered
curl -s http://127.0.0.1:8100/v1/chat/completions -d '{"model":"ics-helper-agent","messages":[{"role":"user","content":"وضعیت اعتباری چطور است"},{"role":"assistant","content":"clarifying?"},{"role":"user","content":"چک"}]}' | jq '.audit.applied_subspace'
```

#### Verifying persistence

- Every RAG turn writes `traces` + `evaluations` with `created_at`, `applied_subspace`, `available_subspaces`, `filter_path` → visible in `http://127.0.0.1:3000/dashboard/observability` as subspace badge + clarifying state and exportable as JSONL.

### Web service

```mermaid
graph TD
    APP[kb_manager.web.app FastAPI 0.1.0 + lifespan pre-warm] --> R1[/documents /chunks /pipeline/]
    APP --> R2[/search /benchmarks /monitoring/]
    APP --> R3[/transparency + zip_browser /ingestion + kb_history /versions /cleanup/]
    APP --> H[/health /api/health /ready dashboard /]
    APP --> DB[(SQLite or pgvector)]
```

Lifespan pre-warm builds the tantivy RAM index (1110 docs, ~5 ms) + MiniLM embeddings + cross-encoder on GPU (~60 s first boot includes HuggingFace cache load) so the first query is fast (`kb_manager/web/app.py:14-57`, `kb_manager/tantivy_bm25.py`). Subsequent queries skip the 2.7 s fingerprint DB scan via `KB_FAST_CACHE=true` (`search.py:_get_index`), yielding ~55 ms total.

## Project structure

```
kb-manager/
├── kb_manager/
│   ├── cli.py                 # CLI: ingest / status / search / serve / inspect / eval-* / dedup
│   ├── config.py              # dataclass config, KB_* env resolution (0.0.0.0:8000, sqlite default)
│   ├── dense.py               # numpy cosine index, npz cache + fingerprint, use_context=False
│   ├── reranker.py            # cross-encoder registry (BGE default, MiniLM fallback), pool override
│   ├── chunker/               # base / fixed / semantic / registry (strategies: semantic, fixed)
│   ├── parsers/               # base / xlsx / docx / pdf / registry
│   ├── preprocessor/          # clean / persian / regex_persian / validators / pipeline
│   ├── embedder/              # base / sentence_transformer / registry
│   ├── pipeline/              # orchestrator / quality / versioning
│   ├── models/                # database / schemas / queries
│   ├── web/                   # app.py + routes/ (12 routers) + templates/ + static/
│   │   └── routes/            # benchmarks, chunks, cleanup, documents, ingestion_suite,
│   │                          # kb_history, monitoring, pipeline, search, transparency,
│   │                          # versions, zip_browser
│   ├── evaluation/            # synthetic generator + retrieval metrics
│   ├── retrieval_training/    # mining / stage analysis packages
│   ├── versioning/            # KB snapshot/version helpers
│   ├── dedup.py               # question + MinHash LSH dedup pipeline
│   ├── query_expansion.py     # synonym beam5 + colloquial→formal map
│   ├── query_reform.py query_enhance.py hyde.py llm.py famteb.py
│   └── synonym_*.py cleanup/
├── tests/                     # 18 test modules: 13 top-level (chunker, parsers, pipeline, embedder,
│                              # evaluation, ingestion_suite, kb_history, qa_massive, …) + 5 under retrieval_training/
├── configs/                   # default.yaml + chunking/
├── versions/                  # v1, v2, v4_retrieval, v6, v7_iva_1405-05-31,
│                              # v10_1405-06-23, v11_1405-06-23 (manifest + export + report)
├── artifacts/retrieval_training/  # agent2 benchmark, JSONL rows, live_dashboard.py,
│                              # rca/ + rca_judge/ corpora, report_plots/, HTML+PDF reports
├── data/                      # kb_*.db, dense_embeddings.npz, v10_source/, plots/
└── pyproject.toml             # deps, kb-manager entry point, pytest/ruff/mypy config
```

Corpus sources live in the **`kb-source` submodule** (`Work_RAG-KB-SourceFiles`): version folders `1405-05-31`, `1405-06-23`, `1405-07-06` (current ingestion, `5741927` — 18 docs / 1,123 chunks, full rebuild 2026-10-07). Previous versions kept for history; see `kb-source/README.md`.

## Tech stack

| Layer | Libraries / models (pinned in `pyproject.toml`) |
|-------|-----------------------------------------------|
| API / server | `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `jinja2`, `python-multipart`, `pyyaml` |
| Data | `sqlalchemy[asyncio]>=2.0` (`aiosqlite` / `asyncpg`), `pgvector>=0.3`, `psycopg2-binary` |
| Retrieval | `sentence-transformers>=3.0` (MiniLM-L12 384d, GPU), `torch>=2.0` (cuda 13.0, H200), cross-encoders: default `BAAI/bge-reranker-v2-m3` (GPU, 3 GB HBM), benchmark runs `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `tantivy==0.26.2` (Rust BM25, RAM), `numpy>=1.24` |
| Persian NLP | `hazm>=0.10`, `shekar>=0.5` + local `regex_persian.py` maps |
| Parsing | `openpyxl` (+calamine engine option), `python-docx`, `PyMuPDF`, `pandas>=2.0` |
| CLI | `click>=8.1`, `rich>=13` (`kb-manager = kb_manager.cli:main`) |
| Cache (optional) | `redis` async client (lazy import, `redis-test:16379` → TTL 600 s, 2–7 ms hits) + in-memory dict fallback; `search.py:62-` `tantivy_bm25.py` |
| Dev / QA | `pytest` + `pytest-asyncio` + `pytest-cov`, `httpx`, `ruff`, `mypy --strict` |
| Reporting | `matplotlib` (plots), `fpdf2` (PDF export) |

## Getting started

```bash
pip install -e ".[dev]"        # or: pip install -e .
export KB_DB_MODE=sqlite       # sqlite (default) | pgvector
export KB_SQLITE_PATH=./data/kb_test.db
export KB_SOURCE_DIR=/path/to/kb-source/1405-06-23

python -m kb_manager.cli ingest --full -s "$KB_SOURCE_DIR"   # full rebuild
python -m kb_manager.cli status                              # doc/chunk counts
python -m kb_manager.cli search -q "…" -k 5                  # CLI search
python -m kb_manager.cli serve                               # web on 0.0.0.0:8000
```

Useful extras: `inspect <file>` (parser dump), `status-chunks`, `dedup --dry-run`, `eval-generate` / `eval-run`.

## Configuration

Defaults from `kb_manager/config.py` (all overridable via `KB_*` env):

| Key | Default | Notes |
|-----|---------|-------|
| `KB_DB_MODE` | `sqlite` | `sqlite` (aiosqlite) or `pgvector` (asyncpg); `KB_DB_URL` overrides; production `pgvector` at `127.0.0.1:5433` with `shared_buffers=16GB` `effective_cache_size=800GB` + `pg_prewarm` (all 1216 kB + 4.8 MB HNSW in RAM) |
| `KB_SQLITE_PATH` | `./data/kb_test.db` | pgvector runs ignore this |
| `KB_SOURCE_DIR` | `<root>/kb-source` | pass `-s` to ingest for explicit version folder; current `1405-07-06` |
| `KB_WEB_HOST` / `KB_WEB_PORT` | `0.0.0.0` / `8000` | |
| `KB_EMBED_DEVICE` / `KB_RERANKER_DEVICE` | `cpu` | `cuda` on H200 (dense 24 ms, rerank 24 ms vs 5 s CPU) |
| `KB_USE_TANTIVY` | `true` | Rust tantivy RAM BM25 (50 MB heap) vs Python fallback |
| `KB_FAST_CACHE` | `true` | skip 2.7 s per-search fingerprint DB scan; invalidation via restart |
| `KB_SYNONYM_BEAM` | `5` | set `1` to cut BM25 5× if needed |
| Chunking | `semantic`, max 512 / min 100 / overlap 50, parent 1536, scope `sheet`, `dedup_questions=True` | v11: overlap skipped for row-wise atomic chunks |
| Dense | MiniLM-L12, 384d, batch 64, L2-normalised, `use_context=False`, GPU | npz fingerprint includes flag → auto-rebuild; pgvector HNSW `vector_cosine_ops` |
| Reranker | `BAAI/bge-reranker-v2-m3` (GPU, `reranker.py:29`) | MiniLM fallback for benchmarks |
| Fusion | RRF k=60 (`search.py:866`), pool = legacy `min(50, 3×top_k)` unless `KB_RERANK_POOL` overrides (benchmarks use 100) | reranker adds +180 rows into top-5 over RRF (v11 final) |
| Cache | `KB_REDIS_URL` (`redis://127.0.0.1:16379/0`), `KB_REDIS_TTL` (600 s) | 2–7 ms hits (150×), miss → in-memory fallback; errors never cached |

## API & UI

Routers mounted in `kb_manager/web/app.py:90-101` — `/documents`, `/chunks`, `/pipeline`, `/versions`, `/monitoring`, `/search`, `/benchmarks`, `/cleanup`, `/transparency` (+ zip browser), `/ingestion` (+ KB history) — plus `/health`, `/api/health`, `/ready` and `/` dashboard. The UI ships Persian RTL views (`Vazirmatn`) and a live massive-benchmark page.

## Version history

| Ver | Corpus | Pipeline / retrieval | Benchmark (measured) |
|-----|--------|---------------------|----------------------|
| v1 | ~160 files (31Tir1405) | baseline ingest + BM25 | — |
| v2 | 355 docs / 6,208 chunks | BM25 + TF-IDF (RRF k=60) | Hit@5 90%, MRR 0.736, 2.8 s |
| v3 | 355 docs / 6,208 chunks | + dense MiniLM-L12 + RRF | Hit@5 89.2%, MRR 0.787, 1.9 s |
| v4 | 355 docs / 6,208 chunks | + char 3-grams + cross-encoder reranker | Hit@5 90%, MRR 0.775, 15.8 s (rerank cost) |
| v5 | 355 docs / 6,208 chunks | frozen dataset + BM25×3 | Hit@5 84.2%, MRR 0.751, 4.2 s |
| v6 | 69 docs / 3,626 chunks | P0–P8 remediation, Persian central, synonym beam5, MinHash dedup | 10-q smoke 100% Hit@5 |
| v7 | 34 docs / 2,074 chunks | fresh 1405-05-31 KB, TestQuestion* excluded, colloquial expansion, IVA-15 | doc-Hit@5 73.3% (11/15), MRR 0.466 |
| v8 | 103 docs / 6,593 chunks | pgvector HNSW-384 + tunable keyword×3.0 | HNSW 23.1 s → GPU 18.4 s |
| v9 | 21+32 docs / ~1,084 chunks | KB_9.7.2026 type-aware, transparency/zip, massive-live | IVA 11/15 73.3% MRR 0.474 (contemporary record; current re-run file `data/iva_results.json` shows 13/15 doc-hit, MRR 0.458) |
| v10 | **35 docs / 2,282 chunks** | 1405-06-23, kv/single-col schemas, stage-level bench (old A/B/C/D taxonomy) | **Hit@5 0.9244**, MRR 0.8369 — A=46 B=6 C=2 D=660, 0 errors (`c3193ad`, `v10_benchmark.log` DONE line; A/A′ not yet split) |
| v11 ⭐ | **35 docs / 2,227 chunks** | chunkfix rebuild + A/A′ diagnostics + dense `use_context=False` | **Hit@5 0.9306** (617/663), Hit@1 0.7511, MRR 0.8281 — A=25 A′=8 B=13 D=617, 0 errors (final re-run; intermediate 714-row run hit 55 Cheque OOMs, kept as `.pre_cheque_rebuild.jsonl`) |
| v12 | code only (same v11 KB) | typo corrections + exclusive-term beam preservation + adaptive beam gate + `fingerprint()` chunk_types param (`fa1f851`); Redis result cache (`1f48847`) | SYNONYM_MAP 74 (v7) → **93 entries** |
| v13 | **18 docs / 1,123 chunks** | `1405-07-06` corpus (`5741927`), full rebuild 2026-10-07, all vectors in RAM (`shared_buffers 16GB`), tantivy RAM BM25, GPU dense/rerank | ingest: 18/24 docs, 1,123 chunks, 0 failed; retrieval Hit@5 pending re-benchmark |
| v13-RAM | code + infra | **RAM-first:** `shared_buffers=16GB` `effective_cache_size=800GB` `pg_prewarm` (4.8 MB HNSW), `tantivy_bm25.py` (Rust, 50 MB heap, 500×), `KB_FAST_CACHE=true` (skip 2.7 s fingerprint), GPU `cuda` for dense/rerank | **Latency 55 ms** (see below) vs 3.1 s before |

v11 manifest: `versions/v11_1405-06-23/manifest.json`. v10 export: `versions/v10_1405-06-23/`.

## Architecture evolution

- **v1→v4 — hybrid core.** Baseline BM25 ingest grew a dense MiniLM leg with RRF-k60 fusion, char 3-grams for Persian typos, and a cross-encoder rerank stage. Latency moved 2–4 s → ~16 s on CPU: the reranker pool is the cost center (still true in v11).
- **v5→v6 — quality remediation.** Frozen checksummed datasets, BM25 keyword weighting, Persian regex central, question + MinHash-LSH dedup, fingerprint/invalidation, async fixes.
- **v7→v9 — fresh corpora + scale-out.** Isolated 1405-05-31 rebuild with test-dir exclusion and colloquial expansion; pgvector HNSW-384 backend; KB_9.7.2026 type-aware chunking with transparency/zip tooling.
- **v10 — measurement.** New `kv_pair` / `single_col_list` schemas, ingest-time variable fan-out, 5-step ingestion wizard, and the first full-corpus stage-level benchmark (714 gold rows, per-stage top-k JSONL).
- **v11 — chunkfix (current).** Three pollution fixes, all verified at `0` residual in the live DB across all QA files:
  1. overlap prefix removed from row-wise atomic chunks (`chunker/semantic.py:426`);
  2. `کلیدواژه‌ها:` label stripped from QA content, keywords kept only in the JSON column (`chunker/semantic.py:485-486,494`);
  3. dense `use_context` default `True→False` with fingerprint-gated auto-rebuild (`dense.py:124,257`) — ablation: recall@100 0.822→0.962.
  
  Plus A vs A′ fusion-loss diagnostics (`agent2_v10_1405_06_23.py`) and the live funnel TUI (`live_dashboard.py`).
- **v12 — query path + caching (current code, same v11 KB).** Typo corrections (`_TYPO_CORRECTIONS`), exclusive-term beam preservation (rare tokens never dropped), adaptive beam gate (`should_expand()`, beam=2 for long exclusive queries), and `fingerprint()` chunk_types param (`fa1f851`); SYNONYM_MAP grew 74 → 93 entries. Re-benchmark pending.
- **Search-result cache.** `POST /search/api` responses cached by `sha256(normalized_query|top_k|keyword_boost)`, TTL 600 s, errors never cached; lazy `redis.asyncio` client (`KB_REDIS_URL`, default `redis://127.0.0.1:16379/0`) with in-memory dict fallback (`1f48847`; measured 150× cold→warm, 2–7 ms hits on v13).
- **v13 — RAM-first + Rust tantivy (current).** Postgres vectors pinned in RAM (`shared_buffers 16GB` `effective_cache_size 800GB` `pg_prewarm` 750 blocks, `docker exec rag-postgres psql -c "ALTER SYSTEM SET shared_buffers='16GB'"` + restart), tantivy `kb_manager/tantivy_bm25.py` (Rust, RAMDirectory 50 MB heap, whitespace tokenizer over Persian 3-gram stream, 500× vs Python BM25: 3000 ms → 4 ms), `KB_FAST_CACHE=true` eliminates per-search fingerprint DB scan (2.7 s → 0), GPU for dense (`paraphrase-multilingual-MiniLM-L12-v2`) and reranker (`BAAI/bge-reranker-v2-m3` on H200, 3 GB HBM, 5400 ms → 24 ms). End-to-end: **3.1 s → 55 ms** (measured `curl --noproxy "*" -X POST :8000/search/api -d '{"query":"اعتبارسنجی چیست","top_k":5}'` `stage_ms` on 1110 chunks, 1405-07-06).

## Benchmarks & metrics

v11 final run (**663 gold rows, 0 errors**; reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`):

| Label | Count | Meaning |
|-------|-------|---------|
| D success | 617 | gold in final top-5 |
| A retriever miss | 25 | gold in neither BM25 nor dense top-100 (Company 14, Individual 7, Public 4, **Cheque 0**) |
| A′ fusion loss | 8 | one leg found gold, RRF dropped it |
| B reranker loss | 13 | merged top-100 → dropped from top-5 |

**Hit@5 0.9306** (617/663), Hit@1 0.7511 (498/663), MRR 0.8281 — recomputed from the committed JSONL (`report_metrics.json`).

![failure funnel](artifacts/retrieval_training/report_plots/p_funnel.png)

### Stage funnel (663 rows)

| Stage | Hit@5 | Recall@100 |
|-------|-------|------------|
| BM25 | 382 (0.576) | 576 (0.869) |
| Dense | 399 (0.602) | 552 (0.833) |
| Merged RRF | 437 (0.659) | 630 (0.950) |
| Final rerank | **617 (0.931)** | — |

![stage funnel](artifacts/retrieval_training/report_plots/p_stages.png)

### Per-file Hit@5 (663 gold rows, 0 errors)

| File | n | Hit@5 | Funnel |
|------|---|-------|--------|
| Company_CRM_Questions | 332 | 0.9157 | D=304 A=14 A′=6 B=8 |
| IndividualCRMQuestions | 179 | 0.9330 | D=167 A=7 A′=1 B=4 |
| PublicQuestions.xlsx | 55 | 0.9273 | D=51 A=4 |
| Individual_CRM_Questions_categorized | 10 | 0.9000 | D=9 A′=1 |
| DisputeQuestions / EtebaritoProblems / مباحثی | 21 | 1.0000 | D=21 |
| **ChequeQuestions** | 66 | **0.9848** | D=65 B=1 |

![per-file hit rate](artifacts/retrieval_training/report_plots/p_perfile.png)

### v10 vs v11

| Metric | v10 pre-fix | v11 intermediate (OOM episode) | v11 final (current) |
|--------|-------------|-------------------------------|---------------------|
| Gold rows | 714 | 714 | **663** |
| Hit@5 | 0.9244 | 0.8585 (0.9302 on executed) | **0.9306** |
| MRR | 0.8369 | 0.8271 | **0.8281** |
| Funnel | A=46 (lumped) B=6 C=2 D=660, err=0 | A=25 A′=8 B=13 D=613, err=55 | A=25 A′=8 B=13 **D=617, err=0** |
| Chunk pollution | overlap + keyword label + dense prefix present | fixed in code, DB rebuilding | 0 label leaks / 0 overlap prefixes (verified) |

The intermediate run (`resume_run.log` DONE line; backup `.pre_cheque_rebuild.jsonl`) counted 55 Cheque infra OOMs as misses (Cheque 61/117 = 0.5214). On re-run all 55 cleared — the gold set shrank by 51 Cheque rows while D grew by 4 (613→617) — leaving Cheque at 65/66 = 0.9848.

Title-removal ablation (measured, `title_removal_experiment.json`):

![title-removal ablation](artifacts/retrieval_training/report_plots/p_ablation.png)

Dense recall@100 **0.822 → 0.962**, recall@5 0.413 → 0.793 with content-only embeddings.

## Latency

### v11 baseline (Python BM25, CPU, slow fingerprint — 663 rows, 100-pool)

Measured per-query `elapsed_ms` from the v11 final benchmark JSONL (663 rows, 0 errors; CPU cross-encoder, 100-pool):

| Slice | n | mean | p50 | p95 | max |
|-------|---|------|-----|-----|-----|
| overall | 663 | 15.2 s | 13.7 s | 22.3 s | 282.1 s |
| ChequeQuestions | 66 | 15.4 s | 13.7 s | 20.7 s | 104.5 s |
| Company_CRM_Questions | 332 | 15.7 s | 13.9 s | 22.5 s | 282.1 s |
| IndividualCRMQuestions | 179 | 14.6 s | 13.2 s | 24.9 s | 41.2 s |
| PublicQuestions.xlsx | 55 | 14.0 s | 13.7 s | 17.6 s | 20.1 s |
| Dispute / Etebarito / categorized | 29 | 13.1–14.7 s | 12.7–14.6 s | 16.5–18.0 s | 16.5–18.0 s |

The rerank stage dominated (BM25/dense/RRF ms-scale on small corpora). Reference points: v5 end-to-end 4.2 s; v8 HNSW-GPU 18.4 s vs CPU 23.1 s.

### v13 RAM-first (tantivy + GPU + fast cache — 1405-07-06, 1110 chunks, top_k=5, measured 2026-10-07)

Host: 1.0 TiB RAM (867 Gi free) · 2× NVIDIA H200 143 GB HBM · Postgres `shared_buffers=16GB` `effective_cache_size=800GB` `pg_prewarm` (chunks 152 blocks + HNSW 598 blocks in RAM) · `KB_EMBED_DEVICE=cuda` `KB_RERANKER_DEVICE=cuda` (3 GB HBM) · `tantivy==0.26.2` RAM (50 MB heap) · `KB_FAST_CACHE=true` (skip 2.7 s fingerprint) · `KB_SYNONYM_BEAM=5`.

| Stage | Python BM25 (before) | Rust tantivy (after) | Share after |
|-------|----------------------|----------------------|-------------|
| BM25 lexical (content+kw, beam5) | **~3000 ms** | **3.8–5.9 ms** | **7%** |
| Dense cosine (pgvector HNSW, GPU) | 16–50 ms (CPU: 3000 ms) | **22–27 ms** | **44%** |
| RRF fusion k=60 | 0.1–0.2 ms | **0.1–0.2 ms** | <1% |
| Cross-encoder rerank (BGE-v2-m3, GPU) | 20–23 ms (CPU: 5400 ms) | **21–25 ms** | **44%** |
| **Total** | **~3100 ms** (warm) / 12 s cold | **51–66 ms** (warm, 55 ms avg) | **100%** |
| **Redis hit** | 2–7 ms | **2–7 ms** | **150×** vs cold |

*Warm = after first query loads GPU models; cold first query after restart: dense 4246 ms + rerank 5396 ms + BM25 5 ms → ~12 s one-time. Subsequent queries: 55 ms avg (measured `POST /search/api` `stage_ms` via `httpx` trust_env=False, `curl --noproxy "*" http://127.0.0.1:8000/search/api -d '{"query":"اعتبارسنجی چیست","top_k":5}'`).*

**Does Postgres save vectors on RAM and leverage GPU?** No GPU — `pgvector` HNSW (`chunks_embedding_hnsw 4.8 MB`, `chunks 1.2 MB`, `pg_total_relation_size 203 MB` for 1110 chunks) lives on disk but is fully cached in `shared_buffers` (16 GB) + OS `effective_cache_size` (800 GB) + `pg_prewarm(152+598 blocks)`. All queries served from RAM; HNSW distance is CPU-only (`vector_cosine_ops`). Dense embeddings and reranker are on **GPU** (H200, `torch.cuda.is_available()`, `nvidia-smi` 3 GB for KB, `KB_EMBED_DEVICE=cuda` `KB_RERANKER_DEVICE=cuda`).

**Is cross-encoder on GPU?** Yes after `KB_RERANKER_DEVICE=cuda` (`reranker.py:274` `model.to(actual_device)` `dtype=float16` on cuda). Before: CPU 5.4 s; after: 24 ms, GPU memory 3006 MiB (`nvidia-smi` PID 1768968). Same for dense (`KB_EMBED_DEVICE=cuda`).

Tuning to keep everything on RAM/GPU: `ALTER SYSTEM SET shared_buffers='16GB'` `effective_cache_size='800GB'` `work_mem='256MB'` `maintenance_work_mem='4GB'` + `CREATE EXTENSION pg_prewarm; SELECT pg_prewarm('chunks')`; `tantivy` RAMDirectory heap 50 MB; `KB_FAST_CACHE=true` eliminates per-search fingerprint scan; `KB_USE_TANTIVY=true` (fallback Python if `tantivy` missing).

### Restart runbook — never regress to CPU / offline reranker (2026-10-13)

**Incident:** after a 2026-10-12 restart (NVIDIA module reload + `restore_services.sh`), a bare relaunch (a) dropped `KB_EMBED_DEVICE`/`KB_RERANKER_DEVICE`, so the old `"cpu"` dataclass default turned 55 ms retrieval into ~2.6 s (dense leg on CPU), and (b) set `HF_HOME=/tmp/hf_clean` (nonexistent) + `HF_HUB_OFFLINE=1`, so transformers 4.57's `is_base_mistral()` network probe hard-failed and the cross-encoder reranker was **silently offline on every query** (`"reranker offline fallback"` log lines; RRF-only results; `stage_ms.rerank≈0.7` is the passthrough timer, not real scoring). Note `HF_HUB_CACHE` must point at the `…/huggingface/hub` subdirectory, not its parent (huggingface_hub 0.36 resolves `models--*` directly inside it).

**Fixes (all in code now — defaults are safe):**
- `config.py` `_default_torch_device()`: device defaults are **GPU-first** (`cuda` when `torch.cuda.is_available()`), explicit `KB_*_DEVICE` env still wins — a bare restart can no longer land on CPU.
- `kb_manager/model_cache.py` `resolve_local_model_path()`: reranker/dense/embedder loads resolve repo ids to local dirs first (offline-prep mirror `Work_RAG-Server-Setup/offline-prep/models/huggingface/<org_model>`, then HF cache snapshot) — local paths set `_is_local` in transformers and skip the offline-mode network probe entirely. Reranker weights are mirrored at `offline-prep/models/huggingface/BAAI_bge-reranker-v2-m3` (2.2 GB).
- Canonical launcher: `Work_Credit-RAG_Phase1/deploy/h200/start_kb.sh` + `kb_env.sh` (idempotent restart, health/GPU/reranker post-checks, warm-up query); `/tmp/reload_backup/restore_services.sh` sources the same env.

**Benchmark after fix (2026-10-13, 30 unique queries, warm):** p50 **44.0 ms** / p95 56.2 ms (normal load) — beats the 55 ms v13 baseline; under concurrent host load ≈60 (other GPU work), p50 55 ms / p95 76 ms. Stage split (normal load): BM25 3.7 · dense 17.8 · RRF 0.1 · rerank 19.3 (all p50, ms). RAM residency verified via `pg_buffercache`: `chunks` 153/153 blocks, `chunks_embedding_hnsw` 598/598 blocks resident; KB GPU footprint ~2.5–2.9 GB (`nvidia-smi`); Redis hits ~29 ms wall (curl overhead). Cold first query after restart ≈7–9 s one-time (lazy GPU model load) — subsequent queries warm.

**Verify a restart is healthy:** `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep <kb_pid>` (expect ~2.5 GB), `grep -c "reranker offline fallback" /tmp/kb-logs/kb_8000.log` (expect 0), then one uncached `POST /search/api` and check `stage_ms` dense/rerank ≈20 ms each.

### Optimization pass 2026-10-13 (dense leg)

After the v13-RAM restart fix above, the dense leg — now the largest share (41%) — was optimized from **20.5 ms → 13.5 ms** p50 without touching quality:

| Change | Dense p50 | Quality guard |
|---|---|---|
| In-process RAM cosine matrix replaces per-beam **pgvector SQL roundtrip** (1 encode + `(1110×384) @ (n_beam×384)`; HNSW now only at ingestion) | 20.5 → 15.1 ms | 100% top-5 overlap vs pgvector (60/60, 12 queries) |
| All beam variants encoded in **one batched GPU call** (was 1 GPU round-trip per variant) | + | identical |
| **fp16** encode via `model.half()` (corpus matrix stays float32; query cast back) | +2–3 ms (load) | fp16 query vs fp32 corpus, 100% overlap |
| **Reranker score memoization**: `(query,passage)`-keyed TTL cache (600 s); repeated candidate pairs skip the forward pass | rerank repeats → ~0 ms | order identical (reranker is deterministic) |
| `ORJSONResponse` on the search route (26 KB payloads) | ~−1 ms serialization | — |

![latency v13.1](artifacts/retrieval_training/report_plots/p_latency_v13_1.png)

Fresh-query p50 **33.1 ms / p95 36.0 ms** (load 61–66 from co-tenant session; v13 README baseline 55 ms). Cached path **~5 ms** server-side (Redis exact-match, 2–7 ms, 150× vs the 2.6 s pre-RAM state). Stage split (p50, warm): BM25 3.5 · dense 8.2 · RRF 0.1 · rerank 20.7. Cold first query after restart ≈7–9 s one-time (lifespan pre-warming the models); subsequent queries warm. RAM residency unchanged: `chunks` 153/153 + HNSW 598/598 blocks in Postgres `shared_buffers` (verified via `pg_buffercache`).

Not pursued (documented as future work): ONNX Runtime / TensorRT export of the BGE cross-encoder — the venv's onnxruntime has no CUDA EP, so a GPU build isn't available offline; swapping to the smaller `bge-reranker-base` or `mmarco-mMiniLM` cross-encoders would ~2× the reranker if validated against the 663-row gold funnel.

## LLM-as-judge feedback & root causes

Judge corpus: `artifacts/retrieval_training/rca_judge/judge_*.md` (gold full text vs retrieved-winner full text + token-Jaccard) with group RCAs in `rca/`. Note the judge files were built pre-fix, so their gold excerpts still show the old `...` prefix and `کلیدواژه‌ها:` label — itself evidence of the pollution v11 removed.

- **Company#141 (B)** — q↔gold Jaccard 0.292 (7 shared: اعضای، گذارد…) vs q↔winner 0.083. *Fusion casualty:* dense had gold at 15, RRF pushed it to merged-62, reranker never recovered (final 11). → weighted fusion / merged-top pinning.
- **Company#151 (C)** — q↔gold 0.084 vs q↔winner 0.075. *Lexical trap:* 9 near-identical Cheque_ReasonCode keyword-stuffed rows (≈249.17 BM25) buried gold at BM25-28. → reason-code sheet quarantine / keyword de-boost.
- **Company#202 (B)** — dense 80 → merged 94 → final 43. *Dense-invisible phrasing* («چه زمانی… اقدام کنم»). → query-rewrite / paraphrase augmentation.
- **Cheque#26 (B)** — BM25=7, merged=12, final=6. *Twin crowding, correct-but-second:* two distractors carry the identical short answer as the gold's family (scores 1.0/0.9999); 17 of 92 Cheque short answers are duplicated (current DB). → answer-level dedup before scoring.
- **Cheque OOM episode (resolved, pre-rebuild backup)** — 55 rows with empty rank lists (`MemoryError` / paging-1455, worst elapsed 108.8 s). *Not a retrieval judgment.* Re-run cleared all 55. → infra headroom (rerank batch/RAM/pagefile) + `--resume` re-runs.

![OOM before vs after](artifacts/retrieval_training/report_plots/p_errors.png)

Ranked root causes (final 663-row run): (1) true retriever misses A=25 in Company/Individual/Public; (2) reranker near-misses B=13, the largest rescue opportunity; (3) fusion losses A′=8 — RRF dropped single-leg finds; (4) twin crowding — real but small (1 B row, −1 slot); (5) the resolved Cheque OOM episode (see backup JSONL).

## Evaluation reports

- `artifacts/retrieval_training/cheque_eval_report.html` — full breakdown with Persian samples + plots (218 KB)
- `artifacts/retrieval_training/cheque_eval_report.pdf` — plots + tables, ASCII-safe (168 KB)
- `artifacts/retrieval_training/report_metrics.json` — machine-readable funnel/stage/latency summary
- `artifacts/retrieval_training/retrieval_failures_v10_1405-06-23.jsonl` — 663 rows with per-stage top-k (plus `.pre_cheque_rebuild.jsonl`, the 714-row OOM-episode backup)
- `artifacts/retrieval_training/live_dashboard.py` — live funnel TUI
- `docs/retrieval_training/RCA_v10_README.md` — consolidated RCA

## Dataset — ICS Credit Scoring Knowledge Base (v13)

This section documents the corpus, retrieval pipeline, model architecture, evaluation methodology, and all benchmark results.

### Corpus Overview (v13_1405-07-06)

| Metric | Value |
|---|---|
| **Version** | `v13_1405-07-06` (commit `5741927`) |
| **Documents** | 18 |
| **Chunks** | 1,123 |
| **Chunk Types** | `qa_pair` 776 · `reason_detail` 198 · `body` 71 · `article` 46 · `kv_pair` 19 (+ 13 parent chunks) |
| **Language** | Persian (Farsi) |
| **Domain** | Iranian credit scoring (check reports, facility reports, CRM Q&A) |
| **Source** | `kb-source` submodule (`1405-07-06` folder, full rebuild 2026-10-07) |
| **KB DB** | PostgreSQL + pgvector (HNSW 384-d) + `shared_buffers=16GB` + `pg_prewarm` |

### Knowledge Tree (Folder Hierarchy)

```
پایگاه دانش (18 docs / 1123 chunks)
├── اشخاص حقیقی — اعتبارسنجی تسهیلات
│   ├── بانک سوالات (Q&A) — 337 + 244 + 0 chunks
│   ├── توضیحات تشریحی بخش‌های گزارش افراد — 37
│   └── دلایل کاهش امتیاز — 80 + 11
├── اشخاص حقیقی — گزارش چک
│   ├── بانک سوالات چک — 118
│   ├── توضیحات بخش‌های گزارش چک — 12
│   ├── نکات قابل توجه چک — 10
│   └── دلایل کاهش امتیاز چک — 87
├── اشخاص حقوقی — اعتبارسنجی تسهیلات
│   ├── رسا — بخش‌های گزارش اعتباری شرکتها — 24
│   ├── توضیحات بخش‌های مختلف گزارش اعتباری شرکتها — 16
│   └── دلایل کاهش امتیاز اعتباری شرکتها — 24
└── پایگاه دانش عمومی
    ├── معرفی شرکت اعتبارسنجی ایران — 4
    ├── هیات مدیره — 5
    ├── 16 کاربرد اعتبارسنجی در اقتصاد — 17
    ├── سوالات عمومی — 81 (q/a + body)
    └── لینک‌ها — 16
```

**Supplements (EntityList tables, not chunks):** 5 lists (report users, data providers, shareholders, loan titles, Word docs) — injected at search step 8 via cosine similarity vs `description_embedding` (threshold 0.6).

### Retrieval Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Persian Query                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Query Normalize → Synonym/Beam Expansion (beam=5) → HyDE (opt)    │
│  colloquial→formal map (93 entries) + typo corrections              │
└─────────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│  BM25 (Rust Tantivy)    │       │  Dense (MiniLM-L12)     │
│  RAM 50MB heap          │       │  pgvector HNSW 384d     │
│  Keyword×3 + char 3-gram│       │  GPU (fp16, 18 ms)      │
│  ~4 ms                  │       │  ~14 ms (batched)       │
└───────────┬─────────────┘       └───────────┬─────────────┘
            │                                 │
            └───────────────┬─────────────────┘
                            ▼
               ┌─────────────────────────┐
               │  RRF Fusion (k=60)      │
               │  merged top-100         │
               └───────────┬─────────────┘
                           ▼
               ┌─────────────────────────┐
               │  Cross-Encoder Rerank   │
               │  BGE-v2-m3 (GPU fp16)   │
               │  pool=100 → top-5       │
               │  ~24 ms (memoized)      │
               └───────────┬─────────────┘
                           ▼
               ┌─────────────────────────┐
               │  Filter Path / Pin Top-3│
               │  Final top-5 + scores   │
               └─────────────────────────┘
```

### Model Details

| Component | Model | Device | Precision | Latency (p50) |
|---|---|---|---|---|
| **BM25** | Tantivy (Rust) | CPU (RAM) | — | 2.6 ms |
| **Dense Encoder** | `paraphrase-multilingual-MiniLM-L12-v2` (384-d) | GPU (cuda:0) | fp16 encode / fp32 matmul | 13.5 ms (batched beam-5) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` (568M) | GPU (cuda:1) | fp16 | 22.0 ms (memoized) |
| **HyDE** | Off by default | — | — | — |

**Key Optimizations (v13.1, 2026-10-13):**
- In-RAM cosine matrix replaces pgvector HNSW SQL roundtrip
- Single batched GPU encode for query + all beam variants (beam-5)
- fp16 encode (corpus fp32, query cast back) — identical quality
- Query embedding LRU cache (2048 entries, ~186× repeat speedup)
- Reranker score memoization (8192 entries, 600s TTL) — repeat pairs → 0 ms
- Single GPU sync per rerank call
- Model prewarm in lifespan — no cold first user (was 7–9 s)

### Evaluation Methodology

**Gold Set:** 663 Persian QA pairs (0 errors), classified A / A′ / B / D:
- **D success**: gold in final top-5 (617 = 93.06%)
- **A retriever miss**: gold in neither BM25 nor dense top-100 (25 = 3.77%)
- **A′ fusion loss**: one leg found gold, RRF dropped it (8 = 1.21%)
- **B reranker loss**: merged top-100 → dropped from top-5 (13 = 1.96%)

**Per-File Hit@5 (663 rows):**

| File | n | Hit@5 | Funnel |
|---|---|---|---|
| Company_CRM_Questions | 332 | 0.9157 | D=304 A=14 A′=6 B=8 |
| IndividualCRMQuestions | 179 | 0.9330 | D=167 A=7 A′=1 B=4 |
| PublicQuestions.xlsx | 55 | 0.9273 | D=51 A=4 |
| Individual_CRM_Questions_categorized | 10 | 0.9000 | D=9 A′=1 |
| DisputeQuestions / EtebaritoProblems / مباحثی | 21 | 1.0000 | D=21 |
| **ChequeQuestions** | 66 | **0.9848** | D=65 B=1 |

**Stage Funnel (663 rows):**

| Stage | Hit@5 | Recall@100 |
|---|---|---|
| BM25 | 382 (0.576) | 576 (0.869) |
| Dense | 399 (0.602) | 552 (0.833) |
| Merged RRF | 437 (0.659) | 630 (0.950) |
| **Final Rerank** | **617 (0.931)** | — |

### Latency Benchmarks (v13.1, 2026-10-13)

**Fresh Queries (30 queries, load 61–66):**

| Metric | p50 | Mean | p95 | Max |
|---|---|---|---|---|
| **Total** | **39.6 ms** | 41.1 ms | 55.0 ms | 86.7 ms |
| BM25 | 2.6 ms | 3.1 ms | 5.6 ms | 13.4 ms |
| **Dense** | **13.5 ms** | 14.8 ms | 24.2 ms | 41.7 ms |
| RRF | 0.1 ms | 0.1 ms | 0.2 ms | 0.2 ms |
| **Rerank** | **22.0 ms** | 25.5 ms | 33.6 ms | 44.7 ms |

**Cached Path (Redis exact-match):**
- Server-side: **2–7 ms** (150× vs cold 2.6 s)
- Wall time (curl): ~26 ms / query

**Quality Guard:** **100% top-5 overlap** (60/60 queries) between RAM-dense path and legacy pgvector path.

### Benchmark Plots

| Plot | Description | Path |
|---|---|---|
| Latency Comparison | Stage p50 bars: v13 README / before / after | `artifacts/retrieval_training/report_plots/p_latency_v13_1.png` |
| Failure Funnel | A/A′/B/D counts | `artifacts/retrieval_training/report_plots/p_funnel.png` |
| Stage Funnel | Per-stage Hit@5 / Recall@100 | `artifacts/retrieval_training/report_plots/p_stages.png` |
| Per-File Hit@5 | By source file | `artifacts/retrieval_training/report_plots/p_perfile.png` |
| Ablation (Title Removal) | Dense recall@100 0.822 → 0.962 | `artifacts/retrieval_training/report_plots/p_ablation.png` |
| Errors (OOM) | Before vs after Cheque fix | `artifacts/retrieval_training/report_plots/p_errors.png` |

**Data Files:**
- `artifacts/retrieval_training/latency_v13_1.json` — machine-readable latency data
- `artifacts/retrieval_training/report_metrics.json` — funnel/stage/latency summary
- `artifacts/retrieval_training/retrieval_failures_v10_1405-06-23.jsonl` — 663 rows with per-stage top-k
- `artifacts/retrieval_training/cheque_eval_report.html/pdf` — full breakdown with Persian samples

### Evaluation Reports

- `artifacts/retrieval_training/cheque_eval_report.html` — full breakdown with Persian samples + plots (218 KB)
- `artifacts/retrieval_training/cheque_eval_report.pdf` — plots + tables, ASCII-safe (168 KB)
- `artifacts/retrieval_training/report_metrics.json` — machine-readable funnel/stage/latency summary
- `artifacts/retrieval_training/retrieval_failures_v10_1405-06-23.jsonl` — 663 rows with per-stage top-k (plus `.pre_cheque_rebuild.jsonl`, 714-row OOM backup)
- `artifacts/retrieval_training/live_dashboard.py` — live funnel TUI
- `docs/retrieval_training/RCA_v10_README.md` — consolidated RCA

### Training / Build Process

The KB is **not a trained model** but a retrieval index built from source documents:

1. **Ingest** (`kb_manager.cli ingest --full`):
   - Parse `.xlsx` / `.docx` / `.pdf` via parser registry
   - Preprocess: clean + Persian normalization + validators
   - Semantic chunking (max 512 / min 100 tokens, overlap 50)
   - Quality gates + Question dedup + MinHash LSH dedup
   - Embed via `MiniLM-L12` (GPU fp16) → npz cache + pgvector HNSW

2. **Index Build** (lifespan pre-warm):
   - Tantivy BM25 RAM index (50 MB heap, 1110 docs, ~5 ms)
   - Dense npz fingerprint → load or rebuild
   - Cross-encoder reranker warmup (fp16 on cuda:1)
   - Total pre-warm: ~30–60 s first run

3. **Incremental / Rebuild**:
   - `--incremental` flag for new/changed files
   - Fingerprint-based invalidation (model + texts + chunk types)
   - Versioning via `versions/` snapshots

---

## Ambiguity Resolver & Classifier Evaluation (v13.1)

The ambiguity resolution system uses a **hybrid Mizan System-1 + Gemma LLM** approach with a trained SVM-RBF classifier on 7-dimensional subspace features.

### Architecture: Mizan System-1 + Gemma Hybrid Resolver

```mermaid
flowchart TD
    Q[User Query] --> CLASSIFY[3-way LLM Classifier greeting / specification / question]
    CLASSIFY -->|greeting| GREET[Return greeting no retrieval]
    CLASSIFY -->|specification| SPEC[Extract subspace label cheque/facilities_person/legal/general]
    CLASSIFY -->|question| T5[is_scattered_ambiguous?]
    T5 -->|no| RETRIEVE[Retrieve + rerank]
    T5 -->|yes| MIZAN[Mizan System-1 subspace routing + ambiguous flag]
    MIZAN -->|confident| ROUTE[Route to subspace]
    MIZAN -->|ambiguous| GEMMA[Gemma LLM scattered detection + clarify]
    GEMMA -->|scattered| CLARIFY[Ask clarification چک / تسهیلات / شرکت / عمومی]
    GEMMA -->|not scattered| ROUTE
    ROUTE --> RETRIEVE[Retrieve with subspace]
```

**Components:**
- **3-way LLM Classifier** (`nodes/classify.py`): greeting / specification / question
- **Mizan System-1** (`mizan.py:234-280`): calibrated subspace routing with choice probabilities + ambiguous flag via `/v1/systemone`
- **Gemma LLM** (`infer_specification.py`): scattered detection + clarifying question generation
- **Trained SVM-RBF Classifier** (`train_eval_ambiguity.py`): 7-dim subspace features → ambiguity binary decision
- **Fallback**: heuristic keyword-based when LLMs unavailable

### Classifier Training & Evaluation

**Dataset Construction** (`scripts/build_ambiguity_dataset.py`):
- 800 real KB questions extracted from `qa_pair` chunks with ground-truth subspace labels
- 200 ambiguous questions generated via cross-subspace mixing, negation patterns, aliases
- Total 1000 samples (800 per-subspace × 4 + 200 ambiguous) → 80/10/10 train/test/eval split
- Judge: Gemma 4-31B validates each sample for subspace membership / ambiguity

**Feature Engineering** (`build_subspace_protos.py`):
- 7-dimensional subspace features: 4 cosine similarities to subspace prototypes + margin + top1 + entropy
- Extracted via MiniLM-L12 (384-d) → cosine similarity to 4 subspace prototypes
- Interpretable, computationally efficient (no raw embeddings)

### Trained Classifier Performance (svm_rbf — Best Model)

| Metric | Value (Eval Set) |
|---|---|
| **Multiclass Balanced Accuracy** | **0.7006** |
| **Binary Balanced Accuracy (Ambiguous vs Non)** | **0.7804** |
| **Binary F1 (Ambiguous)** | 0.6190 |
| **Binary ROC-AUC** | 0.4211 |

**Model Comparison (Eval Set):**

| Model | Multiclass Bal Acc | Binary Bal Acc | Binary F1 |
|---|---|---|---|
| **svm_rbf** | **0.7006** | **0.7804** | **0.6190** |
| lr | 0.6835 | 0.7680 | 0.5909 |
| svm_linear | 0.6585 | 0.7804 | 0.6190 |
| mlp | 0.6560 | 0.7417 | 0.5581 |
| rf | 0.6292 | 0.7277 | 0.5500 |

### Classifier Evaluation Plots

| Plot | Description | Path |
|---|---|---|
| Confusion Matrix | 5-class confusion matrix (eval set) | `components/orchestrator/data/model_eval/plots/confusion_matrices.png` |
| Model Comparison | Eval set balanced accuracy comparison | `components/orchestrator/data/model_eval/plots/model_comparison_eval.png` |
| Learning Curves | Test balanced accuracy vs train size | `components/orchestrator/data/model_eval/plots/learning_curves.png` |
| Learning Curves (Balanced) | Balanced accuracy learning curves | `components/orchestrator/data/model_eval/plots/learning_curves_test_bal.png` |
| CV vs Eval | Cross-validation vs eval balanced accuracy | `components/orchestrator/data/model_eval/plots/cv_vs_eval.png` |
| Per-Class F1 | Per-class F1 scores by model | `components/orchestrator/data/model_eval/plots/per_class_f1.png` |

**Confusion Matrix (Best Model: svm_rbf, Eval Set):**

| | ambiguous | cheque | facilities_legal | facilities_person | general |
|---|---|---|---|---|---|
| **ambiguous** | 13 | 1 | 0 | 1 | 4 |
| **cheque** | 1 | 22 | 0 | 0 | 0 |
| **facilities_legal** | 1 | 2 | 5 | 1 | 2 |
| **facilities_person** | 5 | 1 | 0 | 15 | 3 |
| **general** | 3 | 0 | 1 | 1 | 18 |

**Per-Class F1 (svm_rbf):**
- cheque: 0.8980 (support 23)
- facilities_person: 0.7143 (support 24)
- general: 0.7200 (support 23)
- ambiguous: 0.6190 (support 19)
- facilities_legal: 0.5882 (support 11)

### Classifier Integration in Pipeline

The trained `svm_rbf` model is serialized to `data/model_eval/best_ambiguity_model.joblib` and loaded at runtime. It operates on the 7-dim subspace features extracted from query embeddings, providing a fast (~1 ms) ambiguity decision that complements the Mizan+Gemma resolver:

```python
# Inference path (nodes/classify.py → mizan.py → train_eval_ambiguity.py)
query → MiniLM-L12 embed → 7-dim subspace features → SVM-RBF → ambiguous? → Mizan/Gemma resolver
```

### Classifier Training Artifacts

- **Best model:** `components/orchestrator/data/model_eval/best_ambiguity_model.joblib` (svm_rbf, eval bal acc 0.7006)
- **Evaluation report:** `components/orchestrator/data/model_eval/evaluation_report.md`
- **Results JSON:** `components/orchestrator/data/model_eval/evaluation_results.json`
- **Plots:** `components/orchestrator/data/model_eval/plots/*.png`

### Key Findings (from `evaluation_report.md`)

1. **Best model:** `svm_rbf` (eval balanced accuracy 0.7006)
2. **Binary classification:** `svm_linear` / `svm_rbf` tied at 0.7804 balanced accuracy
3. **Overfitting check:** All models show minimal overfitting (CV vs Eval diff < 0.05)
4. **Feature efficiency:** 7-dim subspace features achieve 70% balanced accuracy with only 7 interpretable features
5. **Data efficiency:** Learning curves show diminishing returns after ~400 samples

---

## Mizan System-1 Model & Hybrid Architecture (v13.1)

### Mizan Model Overview

| Property | Value |
|---|---|
| **Model** | `mizan-v3-3b` (System-1) |
| **Format** | GGUF Q4_K_M (977 MB) |
| **Path** | `offline-prep/models/huggingface/Nimruz_Mizan-v3-3B-GGUF/mizan-v3-3b-Q4_K_M.gguf` |
| **API Endpoint** | `/v1/systemone` (via Manager :9000 or direct :8083) |
| **Status** | **Model available, not currently deployed** (see Deployment Status below) |
| **Capabilities** | System-1 calibrated decisions: choice/noul/score with probabilities |

### Mizan Deployment Status (2026-10-14)

| Component | Status | Details |
|---|---|---|
| **Model File** | ✅ Available | `offline-prep/models/huggingface/Nimruz_Mizan-v3-3B-GGUF/mizan-v3-3b-Q4_K_M.gguf` (977 MB, Q4_K_M) |
| **Manager (:9000)** | ⚠️ Not loaded | 3 models loaded (gemma-4-31b, gemma-3-27b, qwen3.8-27b); Mizan not in registry |
| **Direct Port (:8083)** | ❌ Not running | No process listening on 8083 |
| **TRT-LLM (:8010)** | ✅ Running | Gemma-4-31B on 8010 (used for generation) |
| **LLM Manager (:9000)** | ✅ Running | 3 models loaded; no `/v1/systemone` endpoint |

**Current Integration:** The codebase has full Mizan client (`clients/mizan.py`), guardrails integration (`judge.py`), retrieval judge (`mizan_retrieval_judge.py`), and subspace routing (`mizan_subspace_route`), but **Mizan model is not currently deployed**. The hybrid resolver falls back to Gemma LLM when Mizan is unavailable.

### Mizan Hybrid Architecture (Production Design)

The hybrid resolver combines **Mizan System-1** (fast, calibrated) with **Gemma LLM** (flexible, fallback):

```mermaid
flowchart TD
    Q[User Query] --> CLASSIFY[3-way Classifier greeting / specification / question]
    CLASSIFY -->|greeting| GREET[Return greeting]
    CLASSIFY -->|specification| SPEC[Lock subspace]
    CLASSIFY -->|question| T5[is_scattered_ambiguous?]
    T5 -->|no| RETRIEVE[Retrieve + rerank]
    T5 -->|yes| CLASSIFIER[SVM-RBF 7-dim features → ambiguous?]
    CLASSIFIER -->|not ambiguous| RETRIEVE[Retrieve + rerank]
    CLASSIFIER -->|ambiguous| MIZAN[Mizan System-1 /v1/systemone subspace routing + ambiguous flag]
    MIZAN -->|confident p > 0.7| ROUTE[Route to subspace]
    MIZAN -->|ambiguous| GEMMA[Gemma LLM scattered detection + clarify]
    GEMMA -->|scattered| CLARIFY[Ask clarification چک / تسهیلات / شرکت / عمومی]
    GEMMA -->|not scattered| ROUTE[Route to subspace]
    ROUTE --> RETRIEVE[Retrieve with subspace]
```

### Mizan vs. Classifier: Side-by-Side Comparison

| Aspect | **SVM-RBF Classifier** | **Mizan System-1** |
|---|---|---|
| **Primary Role** | Fast ambiguity gate (binary) | Full System-1 decision engine |
| **Output** | Binary: ambiguous / not | Choice probs + noul + score |
| **Subspace Routing** | ❌ | ✅ 4-way choice + probs |
| **Guardrails** | ❌ | ✅ Safety/Scope/Severity |
| **Retrieval Judge** | ❌ | ✅ Relevance/Subspace/Sufficiency |
| **Subspace Change** | ❌ | ✅ Multi-turn tracking |
| **Latency** | **~1 ms** (local) | **~50-100 ms** (HTTP + GPU) |
| **Calibration** | SVM decision function | Native System-1 calibration |
| **Training** | 1000 samples (7-dim features) | Pre-trained System-1 (Mizan-v3-3B) |
| **Fallback** | Heuristic keywords | Gemma LLM |
| **Deployment** | Local joblib (~2 MB) | HTTP service (Manager :9000 → Mizan :8083) |

### Mizan Integration Points in Pipeline

| Pipeline Stage | Mizan Function | Endpoint |
|---|---|---|
| **Guardrails** | Safety/Scope/Severity judge | `mizan_guardrails_judge()` |
| **Subspace Routing** | Subspace choice + ambiguous flag | `mizan_subspace_route()` |
| **Retrieval Judge** | Relevance/Subspace/Sufficiency/Quality | `mizan_retrieval_judge()` |
| **Subspace Change** | Conversation-level change decision | `mizan_subspace_change_decision()` |

### Mizan API Specification

**Endpoint:** `POST {MIZAN_BASE_URL}/systemone`

**Request:**
```json
{
  "model": "mizan-v3-3b",
  "record": {"text": "...", "stage": "input"},
  "questions": [
    {"id": "safety", "question": "Is this input safe?", "type": "choice", "options": ["allow","block","clarify"]},
    {"id": "scope", "question": "Is this in-domain?", "type": "noul"},
    {"id": "severity", "question": "Rate severity 0-10", "type": "score", "min_value": 0, "max_value": 10}
  ],
  "temperature": 0.0
}
```

**Response:**
```json
{
  "answers": [
    {"question_id": "safety", "type": "choice", "choice": "allow", "choice_probs": {"allow": 0.95}},
    {"question_id": "scope", "type": "noul", "noul": true, "noul_probs": {"true": 0.98}},
    {"question_id": "severity", "type": "score", "score": 1.2, "score_distribution": {...}}
  ]
}
```

### Mizan Deployment (To Activate)

```bash
# 1. Launch Mizan via llama.cpp (GGUF) on port 8083
cd /splunk-data/v1/Work_RAG-Server-Setup
nohup /tmp/llama-venv/bin/python scripts/services/llama_chat_server.py \
  --model offline-prep/models/huggingface/Nimruz_Mizan-v3-3B-GGUF/mizan-v3-3b-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8083 --model-id mizan-v3-3b --n-ctx 4096 > /tmp/mizan.log 2>&1 &

# 2. Verify
curl -s http://127.0.0.1:8083/health
curl -s -X POST http://127.0.0.1:8083/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model":"mizan-v3-3b","record":{"text":"test"},"questions":[{"id":"test","question":"test","type":"choice","options":["a","b"]}]}'

# 3. Update guardrails/orchestrator env
export MIZAN_BASE_URL=http://127.0.0.1:8083/v1
export GUARD_JUDGE_BACKEND=mizan
```

### Mizan vs. Classifier: When Each Is Used

| Pipeline Stage | Component | Reason |
|---|---|---|
| **Classify** | 3-way LLM (Gemma) | Fast 3-way: greeting/specification/question |
| **Scattered Check** | `_is_scattered_ambiguous` (heuristic) | Fast T5 check |
| **Ambiguity Gate** | **SVM-RBF** (7-dim) | **Fast binary gate** (~1 ms) |
| **Subspace Routing** | **Mizan System-1** | Calibrated 4-way choice + probs |
| **Retrieval Judge** | **Mizan System-1** | Multi-dim: relevance/scope/sufficiency |
| **Guardrails** | **Mizan System-1** | Safety/Scope/Severity |
| **Subspace Change** | **Mizan System-1** | Conversation-level decision |

### Mizan Benchmarks (Planned)

| Metric | Target | Status |
|---|---|---|
| Subspace routing accuracy | > 85% | Pending deployment |
| Ambiguous detection F1 | > 0.75 | Pending deployment |
| Latency (p50) | < 100 ms | Pending deployment |
| Calibration error (ECE) | < 0.05 | Pending deployment |

---

### Restart Runbook (Never Regress to CPU)

```bash
# Stop
fuser -k 8000/tcp; sleep 3

# Start (with optimized env)
KB_DB_MODE=pgvector KB_DB_URL=postgresql+asyncpg://postgres:postgres@127.0.0.1:5433/kb_manager \
KB_EMBED_DEVICE=cuda KB_RERANKER_DEVICE=cuda KB_DENSE_BACKEND=ram \
KB_USE_TANTIVY=true KB_FAST_CACHE=true KB_SYNONYM_BEAM=5 \
KB_REDIS_URL=redis://127.0.0.1:16379/0 KB_REDIS_TTL=600 \
HF_HOME=/root/.cache/huggingface HF_HUB_CACHE=/root/.cache/huggingface/hub HF_HUB_OFFLINE=1 \
nohup /tmp/kb-venv/bin/python -m uvicorn kb_manager.web.app:app --host 0.0.0.0 --port 8000 > /tmp/kb.log 2>&1 &

# Verify
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep <pid>
grep -c "reranker offline fallback" /tmp/kb-logs/kb_8000.log  # expect 0
curl --noproxy "*" -X POST :8000/search/api -d '{"query":"اعتبارسنجی چیست","top_k":5}' | jq .stage_ms
```

---

## Development

```bash
python -m pytest tests/ -q        # suite (116 passed / 6 skipped; 1 pre-existing
                                  # env-dependent failure: test_qa_massive_count finds
                                  # 4 QA files in kb-source/1405-06-28/extracted, needs >=5)
python -m compileall -q kb_manager
ruff check kb_manager tests
mypy kb_manager                   # strict
```

Docs live under `docs/` (retrieval training, architecture notes, Persian resources). KB snapshots under `versions/` with manifest + export + report per version.
