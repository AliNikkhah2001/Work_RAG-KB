# Work_RAG-KB — Detailed Deliverable Plan

> **Source**: Shared opencode session `Lf4DNf9H` + user requirements for webserver as full KB management system
> **Repo**: https://github.com/AliNikkhah2001/Work_RAG-KB
> **Date**: 2026-08-19

---

## Executive Summary

Transform the existing KB Manager (FastAPI + SQLAlchemy + sentence-transformers) from a **pipeline + search demo** into a **production-grade, multi-role KB Management System** with:

| Role | Core Capability |
|------|-----------------|
| **Developer** | Upload ZIP of Excel files → auto-extract → ingest → retrieval API (REST + OpenAPI docs) |
| **Editor** | Excel-like table editor for chunks/documents, inline edit, versioning, merge/split, export |
| **Monitoring** | Real-time dashboards: latency trends, top queries, retrieval quality (MRR/NDCG/Recall), staleness, costs, errors |
| **Admin** | User management, roles (developer/editor/admin), auth (JWT/session), audit logs |

**Persian-first**: All UI RTL, Persian labels, ZWNJ-aware search, Hazm+Shekar preprocessing integrated.

**Advanced Retrieval** (beyond current BM25+Dense+RRF): HyDE, Multi-query rewriting, Contextual retrieval, Cross-encoder reranking, Query-type adaptive weighting — all exposed via unified `/api/search` with strategy params.

---

## Milestone 1: Detailed Settings & Configs Section (Week 1)

### 1.1 Expand Config System (`config.py` + `configs/*.yaml`)

| Config File | Purpose | Key Fields |
|-------------|---------|------------|
| `configs/default.yaml` | Base config | db, embedding, chunking, parser, ragas, **retrieval**, **reranker**, **hyde**, **multi_query**, **auth**, **monitoring**, **web** |
| `configs/retrieval.yaml` | Retrieval strategies | bm25_weight, dense_weight, rrf_k, top_k_candidates, rerank_top_k, hyde_enabled, multi_query_enabled, query_type_adaptive |
| `configs/reranker.yaml` | Cross-encoder settings | model_name, batch_size, max_length, device, threshold |
| `configs/hyde.yaml` | HyDE generation | prompt_template, max_length, temperature, cache_ttl |
| `configs/multi_query.yaml` | Multi-query rewriting | num_queries, prompt_template, beam_size, temperature, query_types |
| `configs/auth.yaml` | Authentication | secret_key, algorithm, access_token_expire_min, roles, default_role |
| `configs/monitoring.yaml` | Monitoring thresholds | latency_p95_ms, recall_threshold, mrr_threshold, staleness_days, cost_threshold_usd |
| `configs/web.yaml` | Web UI settings | language, rtl, theme, page_size, editor_settings, dev_api_prefix |

### 1.2 Settings Web Page (`/settings`)

- **Tabs**: Database, Embedding, Chunking, Retrieval, Reranker, HyDE, Multi-Query, Auth, Monitoring, Web UI
- **Persian RTL** form with live validation
- **Save → reload** (hot-reload via `config_watcher` or app restart signal)
- **Export/Import** config as YAML/JSON
- **Environment override** display (shows which values come from env vars)

### 1.3 Deliverables

- [ ] `config.py` — add dataclasses for RetrievalConfig, RerankerConfig, HyDEConfig, MultiQueryConfig, AuthConfig, MonitoringConfig, WebConfig
- [ ] `load_config()` — merge YAML files + env overrides with precedence
- [ ] `configs/*.yaml` — 8 config files with Persian comments
- [ ] `kb_manager/web/routes/settings.py` — GET/POST /settings, /settings/export, /settings/import
- [ ] `templates/settings.html` — RTL Persian tabbed form
- [ ] Tests: `tests/test_config.py` — load, merge, validation, hot-reload signal

---

## Milestone 2: Advanced Multi-Leg Retrieval API (Week 1-2)

### 2.1 Wire Existing Components into Search Route

Currently in `kb_manager/web/routes/search.py`:
- BM25 (lexical) ✓
- Dense (MiniLM-L12-v2, contextual) ✓
- RRF fusion (k=60) ✓
- Cross-encoder rerank (mmarco-mMiniLMv2) ✓
- Step-by-step transparency (`SearchSteps` model) ✓

**Missing integration from `query_reform.py`:**
- HyDE (hypothetical document embedding) — generate pseudo-doc → embed → search
- Multi-query rewriting — generate 6 variants → search each → RRF fuse
- Query type detection — adaptive weights per type (keyword_only → BM25 heavy; conversational → dense heavy; typo → dense+rerank heavy)

### 2.2 New Unified Search API

```python
# POST /api/search
{
  "query": "امتیاز اعتباری چیست؟",
  "top_k": 5,
  "strategy": "auto",           # "auto" | "bm25" | "dense" | "hybrid" | "hyde" | "multi_query" | "full"
  "retrieval_options": {
    "hyde": {"enabled": true, "top_k": 20},
    "multi_query": {"enabled": true, "num_queries": 6},
    "rerank": {"enabled": true, "top_k": 50},
    "query_type_adaptive": true,
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "rrf_k": 60
  },
  "filters": {"domain": "general", "category": "article"}
}
```

**Response:**
```json
{
  "query": "original",
  "detected_type": "verbatim",
  "strategy_used": "full",
  "sub_queries": [...],           // if multi_query
  "hyde_doc": "...",              // if hyde
  "steps": { ... },               // SearchSteps with all intermediate results
  "results": [...],
  "elapsed_ms": 245,
  "strategy_breakdown": {
    "bm25_ms": 12,
    "dense_ms": 78,
    "hyde_ms": 145,
    "multi_query_ms": 310,
    "rerank_ms": 95
  }
}
```

### 2.3 Query-Type Adaptive Weights (from research)

| Query Type | BM25 Weight | Dense Weight | HyDE | Multi-Query | Rerank |
|------------|-------------|--------------|------|-------------|--------|
| verbatim | 0.5 | 0.5 | ✗ | ✗ | ✓ |
| paraphrase | 0.3 | 0.7 | ✓ | ✓ | ✓ |
| conversational | 0.2 | 0.8 | ✓ | ✓ | ✓ |
| typo | 0.2 | 0.8 | ✗ | ✓ | ✓ |
| keyword_only | 0.8 | 0.2 | ✗ | ✗ | ✓ |
| reworded | 0.3 | 0.7 | ✓ | ✓ | ✓ |
| auto (detect) | adaptive | adaptive | per-type | per-type | ✓ |

### 2.4 Deliverables

- [ ] `kb_manager/retrieval/orchestrator.py` — new `HybridRetriever` class combining all strategies
- [ ] `kb_manager/retrieval/adaptive.py` — query type detection + weight selection
- [ ] Update `search.py` — new `/api/search` endpoint with strategy param; deprecate old `/search/api`
- [ ] `templates/search.html` — Persian UI with strategy selector, filters, step-by-step expandable panels
- [ ] OpenAPI docs (`/docs`) — full schema with examples
- [ ] Tests: `tests/test_retrieval.py` — each strategy, adaptive weights, combined, latency

---

## Milestone 3: Developer Workflow — ZIP Upload → Retrieval API (Week 2)

### 3.1 Requirements

- **POST /api/dev/upload-zip** — accept `multipart/form-data` with `.zip` file
- Extract → validate (only .xlsx/.pdf/.docx) → save to temp dir
- Trigger **async ingestion job** (reuse `PipelineOrchestrator.run_full_rebuild`)
- Return `job_id` + status polling endpoint
- **GET /api/dev/jobs/{job_id}** — progress, errors, stats
- **POST /api/dev/search** — simplified retrieval for external apps (API key auth)
- OpenAPI docs page at `/dev/docs`

### 3.2 Auth for Developer API

- API Key model: `DeveloperKey(id, name, key_hash, rate_limit, created_at, expires_at, is_active)`
- Header: `X-API-Key: <key>`
- Rate limit: configurable (default 100 req/min)
- Scopes: `ingest`, `search`, `admin`

### 3.3 Deliverables

- [ ] `kb_manager/models/database.py` — add `DeveloperKey` model + migration
- [ ] `kb_manager/auth/api_keys.py` — generate, validate, hash (bcrypt), rate limit (token bucket)
- [ ] `kb_manager/web/routes/dev.py` — dev API routes (upload-zip, job status, search)
- [ ] `kb_manager/services/zip_ingest.py` — ZIP extraction, validation, temp dir cleanup, orchestrator integration
- [ ] `templates/dev.html` — Persian developer portal: upload form, job list, API key management, search test
- [ ] Tests: `tests/test_dev_api.py` — upload, ingest, search, auth, rate limit

---

## Milestone 4: Excel-Like Table Editor + Versioning (Week 2-3)

### 4.1 Requirements

- **Grid view** of chunks (or documents) with inline editing (ag-Grid or Tabulator or custom)
- Columns: ID, Type, Content (truncated, click to expand), Heading Path, Keywords, Token Count, Quality Score, Verified, Actions
- **Inline edit**: double-click cell → edit → Enter to save → auto-create version snapshot
- **Row actions**: verify/unverify, delete (soft), merge with next, split at cursor, duplicate
- **Bulk actions**: select multiple → verify, export, delete, re-index
- **Versioning**: every edit → `DocumentVersion` snapshot with diff preview
- **Sidebar**: version history panel, click to view diff, rollback button
- **Filters**: by document, type, verified status, quality score range, search in content
- **Export**: CSV, Excel, JSON (selected or all filtered)

### 4.2 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/editor/chunks` | Paginated, filtered chunk list (for grid) |
| PATCH | `/api/editor/chunks/{id}` | Update single field (inline edit) |
| POST | `/api/editor/chunks/{id}/version` | Manual version snapshot |
| GET | `/api/editor/chunks/{id}/versions` | Version history for chunk |
| POST | `/api/editor/chunks/{id}/rollback/{version_id}` | Rollback chunk to version |
| POST | `/api/editor/chunks/merge` | Merge multiple chunks |
| POST | `/api/editor/chunks/split` | Split chunk at position |
| POST | `/api/editor/chunks/bulk` | Bulk actions |
| GET | `/api/editor/export` | Export filtered chunks (csv/xlsx/json) |

### 4.3 Deliverables

- [ ] `kb_manager/web/routes/editor.py` — editor API routes (extends chunks.py with versioning/bulk)
- [ ] `kb_manager/services/editor.py` — merge, split, version snapshot, diff logic
- [ ] `templates/editor.html` — RTL Persian ag-Grid integration (CDN or npm build)
- [ ] `static/js/editor.js` — grid config, inline edit handlers, version panel, bulk actions
- [ ] `static/css/editor.css` — RTL grid styles, Persian font, compact density
- [ ] Tests: `tests/test_editor.py` — CRUD, merge, split, version diff, rollback, export

---

## Milestone 5: Monitoring Dashboards (Week 3)

### 5.1 Requirements

**Real-time metrics** (WebSocket or polling):

| Dashboard | Metrics | Visualization |
|-----------|---------|---------------|
| **Overview** | Total docs/chunks, active jobs, retrieval QPS, avg latency p50/p95/p99 | KPI cards + sparklines |
| **Retrieval Quality** | MRR, NDCG@5/10, Recall@5/10, Hit@5, by query type | Time series + bar by type |
| **Latency** | BM25 ms, Dense ms, Rerank ms, Total ms | Stacked area + percentile lines |
| **Staleness** | Docs by age, stale %, by category/domain | Heatmap + trend |
| **Costs** | Embedding tokens, LLM tokens (RAGAS), $/1k queries | Bar + projection |
| **Errors** | Failed jobs, failed queries, parsing errors | Table + count by type |

**Data sources:**
- `retrieval_logs` table (enrich: add `strategy`, `query_type`, `bm25_ms`, `dense_ms`, `rerank_ms`, `faithfulness`, `relevancy`)
- `ingestion_jobs` table
- `documents` / `chunks` for staleness
- `cost_log` table (from KB_ARCHITECTURE.md)

### 5.2 Deliverables

- [ ] `kb_manager/models/database.py` — extend `RetrievalLog` with strategy, timings, quality scores
- [ ] `kb_manager/web/routes/monitoring.py` — WebSocket `/ws/monitoring` + REST `/api/monitoring/*` endpoints
- [ ] `kb_manager/services/monitoring.py` — aggregation queries, time-bucketed metrics
- [ ] `templates/monitoring_dashboard.html` — RTL Persian dashboard with Chart.js (or ApexCharts)
- [ ] `static/js/monitoring.js` — WS connection, chart updates, time range selector
- [ ] Tests: `tests/test_monitoring.py` — aggregation, WS, endpoints

---

## Milestone 6: Login/Auth + Roles (Week 3)

### 6.1 Requirements

- **User model**: id, username, email, password_hash, role (admin/editor/developer/viewer), is_active, created_at, last_login
- **Session**: JWT (access + refresh) or secure cookie session
- **Routes**: `/login`, `/logout`, `/register` (admin only), `/profile`
- **Middleware**: `require_auth`, `require_role(*roles)`
- **Protect**: all `/editor/*`, `/monitoring/*`, `/settings/*`, `/dev/*`, `/api/*` (except public search)
- **Persian RTL** login page with remember-me, password reset (email stub)

### 6.2 Deliverables

- [ ] `kb_manager/models/database.py` — add `User` model
- [ ] `kb_manager/auth/jwt.py` — create/verify tokens, refresh, password hashing (argon2/bcrypt)
- [ ] `kb_manager/auth/dependencies.py` — FastAPI deps: `get_current_user`, `require_role`
- [ ] `kb_manager/web/routes/auth.py` — login, logout, register, profile
- [ ] `templates/login.html`, `templates/profile.html` — Persian RTL
- [ ] Update `app.py` — add auth middleware, protect routes
- [ ] Tests: `tests/test_auth.py` — login, token, role protection, logout

---

## Milestone 7: Persian RTL UI Unification (Week 3-4, parallel)

### 7.1 Requirements

- **Base template** (`base.html`): RTL dir, Vazirmatn font (CDN), RTL-aware CSS variables, Persian date formatting
- **All pages**: Dashboard, Documents, Chunks, Editor, Versions, Pipeline, Monitoring, Benchmarks, Settings, Dev Portal, Search, Login
- **Components**: Navbar (role-aware), Sidebar, Tables, Forms, Modals, Toasts, Pagination, Filters
- **Icons**: Heroicons or Tabler (SVG, RTL-friendly)
- **Colors**: Persian-themed (saffron, turquoise, deep blue), dark/light mode

### 7.2 Deliverables

- [ ] `templates/base.html` — complete RTL rewrite with Alpine.js or vanilla JS for interactivity
- [ ] `static/css/main.css` — RTL utility classes, Persian font, component styles
- [ ] `static/js/main.js` — global handlers (toast, modal, confirm, date picker)
- [ ] Update all existing templates to extend new base
- [ ] Visual QA checklist (all pages, mobile responsive)

---

## Milestone 8: Tests, Lint, CI/CD, Deploy (Week 4)

### 8.1 Testing

| Layer | Target | Tools |
|-------|--------|-------|
| Unit | config, auth, retrieval strategies, editor ops, monitoring agg | pytest, pytest-asyncio |
| Integration | full ingestion pipeline, search API, dev API, auth flow | httpx + test DB (sqlite) |
| E2E | editor grid, versioning, monitoring WS, dev upload | Playwright (optional) |

### 8.2 CI/CD (`.github/workflows/`)

- `lint.yml` — ruff, mypy strict, pytest (unit)
- `test.yml` — integration tests on PR (PostgreSQL service)
- `deploy.yml` — build Docker, push to GHCR, deploy to server (manual approval)

### 8.3 Docker

- Multi-stage: builder → runtime (python 3.11 slim)
- `docker-compose.yml` — app + postgres + pgvector + redis (for Celery later)
- Health checks, resource limits

### 8.4 Deliverables

- [ ] `tests/` — add tests for all new modules (config, auth, retrieval, editor, monitoring, dev)
- [ ] `.github/workflows/*.yml` — 3 workflows
- [ ] `Dockerfile` + `docker-compose.yml` (production-ready)
- [ ] `README.md` — update with new features, API docs link, deployment guide
- [ ] Run full test suite: `pytest -xvs`, `ruff check .`, `mypy kb_manager/`

---

## Cross-Cutting Concerns

| Concern | Approach |
|---------|----------|
| **Observability** | Structured JSON logs (structlog), OpenTelemetry traces (optional) |
| **Security** | Rate limiting (slowapi), CORS, CSP headers, SQL injection prevention (ORM) |
| **Performance** | Connection pooling (asyncpg), query optimization, index hints, caching (Redis for auth/sessions) |
| **i18n** | Persian (fa) primary; English fallback via `gettext` or simple dict |
| **Accessibility** | Semantic HTML, ARIA labels, keyboard nav, color contrast |

---

## Branch Strategy

| Feature | Branch |
|---------|--------|
| Settings & Configs | `feat/settings-configs` |
| Advanced Retrieval | `feat/advanced-retrieval` |
| Developer Workflow | `feat/dev-workflow` |
| Excel Editor | `feat/editor-grid` |
| Monitoring | `feat/monitoring-dashboards` |
| Auth | `feat/auth-roles` |
| Persian UI | `feat/persian-ui` |
| Tests/CI/CD | `feat/tests-ci` |

**Merge order**: settings-configs → advanced-retrieval → dev-workflow → editor-grid → monitoring-dashboards → auth-roles → persian-ui → tests-ci → `main`

---

## Effort Estimate

| Milestone | Dev Days | Risk |
|-----------|----------|------|
| 1. Settings & Configs | 2-3 | Low |
| 2. Advanced Retrieval | 3-4 | Medium (strategy composition) |
| 3. Developer Workflow | 2-3 | Low-Medium (async jobs) |
| 4. Excel Editor | 4-5 | Medium (grid complexity) |
| 5. Monitoring | 2-3 | Low |
| 6. Auth | 2 | Low |
| 7. Persian UI | 2-3 | Medium (template migration) |
| 8. Tests/CI/CD | 2-3 | Low |
| **Total** | **19-26 days** | |

---

## Acceptance Criteria (Definition of Done)

1. **Settings**: All configs in YAML, editable via `/settings` UI, hot-reload works, Persian RTL
2. **Retrieval**: `/api/search?strategy=full` returns results with breakdown; adaptive weights per query type; latency < 500ms p95
3. **Developer**: Upload ZIP → job completes → `/api/dev/search` returns results with API key auth; rate limit enforced
4. **Editor**: Grid loads 1000+ chunks, inline edit saves + creates version, merge/split/rollback work, export CSV/XLSX
5. **Monitoring**: Dashboard shows real-time charts, websocket updates, filters by date/type
6. **Auth**: Login/logout works, roles protect routes, JWT refresh works, Persian RTL
7. **UI**: All pages RTL Persian, Vazirmatn font, consistent components, dark mode
8. **Quality**: `pytest` passes, `ruff` clean, `mypy` strict passes, Docker builds, CI green

---

## Next Actions (Immediate)

1. Create `DELIVERABLES_PLAN.md` in repo root ✓
2. Create feature branch `feat/settings-configs`
3. Implement Milestone 1 (Settings & Configs) — start with `config.py` expansion
4. Daily: push WIP commits, open PR for review
5. After Milestone 1 PR merged → branch for Milestone 2