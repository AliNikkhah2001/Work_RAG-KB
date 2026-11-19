# KB Manager — API Endpoints

Base URL (local): `http://127.0.0.1:8000`
Base URL (compose): `http://kb:8000`

Source of truth: the running app's own OpenAPI document.

```bash
curl -s http://127.0.0.1:8000/openapi.json | python3 -m json.tool | less
```

At the time of writing that document contains **92 paths**. This page lists them all,
grouped by what they are for, with the request/response shapes that matter for
integrating.

---

## Authentication — opt-in API key

The KB manager ships with the key gate **disabled**. Set `KB_API_KEY` and every
endpoint requires it; leave it unset and everything stays open, which is how the
current deployment runs.

### When `KB_API_KEY` is set

Any one of these is accepted:

| Where | Form |
|---|---|
| Header | `X-API-Key: <key>` |
| Header | `Authorization: Bearer <key>` |
| Query | `?api_key=<key>` |

A wrong or missing key returns `401` with `WWW-Authenticate: Bearer`:

```json
{"detail": "Invalid or missing API key"}
```

These paths stay public so container and orchestrator probes never fail on auth:
`/health`, `/api/health`, `/ready`, `/metrics`, `/metrics/healthz`, and everything
under `/static/`.

```bash
export KB_API_KEY=...          # on the KB service
curl -s -H "X-API-Key: $KB_API_KEY" http://127.0.0.1:8000/registry/api/active
curl -s -H "Authorization: Bearer $KB_API_KEY" \
  -X POST http://127.0.0.1:8000/search/api \
  -H 'Content-Type: application/json' -d '{"query":"چک","top_k":1}'
```

Key comparison is constant-time (`hmac.compare_digest`). The gate lives in
`kb_manager/web/auth.py` and is a no-op when the variable is empty.

### When it is unset (current state)

Nothing is checked — not even a bogus header:

```bash
curl -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/registry/api/active   # 200
curl -o /dev/null -w '%{http_code}\n' \
  -H 'X-API-Key: not-a-real-key' http://127.0.0.1:8000/registry/api/active        # 200
```

Consequences to design around:

- Any process that can reach the port can read the corpus, write packs
  (`PUT /registry/api/packs/{kind}`), change the active manifest, and start ingest
  jobs.
- Keep port 8000 on loopback or an internal network. If you front it with a gateway,
  the gateway is the only auth boundary unless you also set `KB_API_KEY`.
- This is a shared secret: no per-caller identity, scopes, rotation or expiry.

Callers that need to work in both modes should send the header only when a key is
configured. The orchestrator currently sends none, so enabling the gate requires
setting `KB_API_KEY` on the KB service *and* adding the header in
`components/orchestrator/src/work_rag_orchestrator/clients/knowledgebase.py`.

There are separate **outbound** keys, which are not inbound auth:

| Env var | Purpose |
|---|---|
| `OPENAI_API_KEY` / `KB_HYDE_API_KEY` | HyDE generation (optional, off unless `KB_HYDE_ENABLED=true`) |
| `KB_RAGAS_API_KEY` | RAGAS benchmark judge |
| `KB_OPENROUTER_API_KEY` / `OPENROUTER_API_KEY` | OpenRouter embedder backend |

---

## Routers and their mounts

From `kb_manager/web/app.py`:

| Router | Prefix | Purpose |
|---|---|---|
| `metrics` | — | Prometheus-style metrics |
| `settings` | `/settings` | Runtime settings page |
| `documents` | `/documents` | Document CRUD |
| `chunks` | `/chunks` | Chunk read / verify / edit |
| `pipeline` | `/pipeline` | Source-dir ingest pipeline |
| `versions` | `/versions` | Document version history |
| `monitoring` | `/monitoring` | Dashboards, staleness |
| `search` | `/search` | Retrieval (the important one) |
| `benchmarks` | `/benchmarks` | Eval runs, snapshots, RAGAS/IVA/massive |
| `cleanup` | `/cleanup` | QA cleanup actions |
| `transparency` | `/transparency` | Trace + question inspection |
| `zip_browser` | `/transparency` | Zip preview / ingest |
| `ingestion` | `/ingestion` | Ingestion suite + KB snapshots + zip ingest |
| `registry` | `/registry` | Packs, manifests, DVC pin |

Plus top-level `GET /health`, `GET /api/health`, `GET /ready`, `GET /`.

Many endpoints return **HTML** (Jinja templates) because the same app serves the
admin UI. Those are marked `HTML` below and are not meant for programmatic use.

---

## 1. Health

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness. `{"status":"ok"}` |
| GET | `/api/health` | Same, JSON |
| GET | `/ready` | Readiness, includes index/reranker state |

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/ready
```

`/ready` is the one to poll after a restart: startup pre-warms the BM25 index,
MiniLM embeddings and the cross-encoder reranker, which takes roughly 30s on first
run.

---

## 2. Search — `/search`

### `POST /search/api` — the main retrieval call

This is what the orchestrator uses.

**Request**

```json
{
  "query": "اعتبارسنجی چیست",
  "top_k": 10,
  "keyword_boost": 3.0,
  "filter_path": null,
  "subspace": "cheque"
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | — | Required; empty query returns `{"error":"Empty query"}` |
| `top_k` | int | 10 | Clamped to max 50 |
| `keyword_boost` | float | env `KB_KEYWORD_BOOST` (3.0) | BM25 keyword field weight |
| `subspace` | string | null | L2 subspace id; resolved to `filter_path` via the taxonomy |
| `filter_path` | string | null | Raw folder-hierarchy filter; wins over `subspace` if both set |

**Response** — `SearchSteps`, every pipeline stage exposed:

```json
{
  "query": "...",
  "normalized_query": "...",
  "tokens": ["..."],
  "total_chunks_indexed": 12345,
  "bm25_results":   [SearchResult],
  "semantic_results":[SearchResult],
  "dense_results":  [SearchResult],
  "merged_candidates":[SearchResult],
  "final_results":  [SearchResult],
  "injected_entities": [],
  "elapsed_ms": 42.1,
  "rerank_ms": 11.3,
  "stage_ms": {"bm25": 3.1, "dense": 22.0, "rerank": 11.3},
  "config": {...},
  "keyword_boost_used": 3.0
}
```

`SearchResult`:

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "doc_title": "...",
  "heading_path": "...",
  "folder_hierarchy": ["..."],
  "content_preview": "...",
  "bm25_score": 0.0,
  "semantic_score": 0.0,
  "dense_score": 0.0,
  "hybrid_score": 0.0,
  "rerank_score": 0.0,
  "ordinal": 0
}
```

**Errors.** Retrieval failures do not raise a bare 500; they return a JSON body and
pick the status from the failure kind:

- `503` when the message mentions `reranker`, `cross-encoder`, `dense`, `encode`,
  `bm25`, or `index not built`
- `500` otherwise

```json
{"error": "...", "retrieval_failed": true, "reranker_failed": true, "traceback": "..."}
```

**Caching.** Results are cached in Redis for 600s keyed on
`hash(normalized_query + top_k + keyword_boost + filter_path)`. Errors are never
cached. If Redis is down it computes normally (and falls back to an in-process dict).

```bash
curl -s http://127.0.0.1:8000/search/api \
  -H 'Content-Type: application/json' \
  -d '{"query":"اعتبارسنجی چیست","top_k":3}' \
  | jq -c '.final_results[0] | {doc_title, heading_path, score: .rerank_score}'
```

### `POST /search/supplement` — entity/supplement lookup

Tool-callable search over `entity_lists` (reason codes, score tables, catalogs),
embedding against `list_name` + `description`. Default threshold 0.4 — deliberately
looser than the 0.6 used by the hidden inject step.

**Request**

```json
{
  "query": "کد ۱۲",
  "threshold": 0.4,
  "subspace": "cheque",
  "filter_path": null,
  "reason_code_model": "person_main"
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | `""` | Empty returns an empty result set, not an error |
| `threshold` | float | 0.4 | Clamped to 0.0–1.0 |
| `subspace` | string | null | Keeps score tables on the matching model |
| `filter_path` | string | null | Used to infer `subspace` when that is unset |
| `reason_code_model` | string | null | `person_main`, `person_substitute`, `legal`, `cheque` — narrows reason-code catalogs after a «کدام مدل؟» clarify turn |

**Response**

```json
{"results": [ ... ], "query": "...", "threshold": 0.4, "count": 2, "subspace": "cheque"}
```

### `GET /search/supplement`

Query-param variant of the same call, for direct tool calling:

```
GET /search/supplement?query=...&threshold=0.4&subspace=cheque&reason_code_model=person_main
```

### `GET /search/config`

Current retrieval config for tuning: embed backend, reranker backend, dense model,
keyword boost and related knobs.

### `POST /search/enhance`

Query enhancement/rewriting preview.

---

## 3. Ingestion — `/ingestion`

Sessions let you build or edit a KB from a zip or a server path, review the chunking,
then ingest into an isolated DB.

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/ingestion` | HTML | The RTL ingestion UI |
| POST | `/ingestion/sessions` | JSON | Create session from an uploaded `.zip` (multipart `file`) |
| POST | `/ingestion/sessions/from-path` | JSON | Create session from a server-local dir/file/zip |
| GET | `/ingestion/sessions` | JSON | List sessions |
| GET | `/ingestion/sessions/{sid}/tree` | JSON | File tree |
| POST | `/ingestion/sessions/{sid}/tree/{op}` | JSON | Tree op (rename/mkdir/delete) |
| GET | `/ingestion/sessions/{sid}/file` | JSON/text | Read a file in the session |
| PUT | `/ingestion/sessions/{sid}/file` | JSON | Write a file in the session |
| GET | `/ingestion/sessions/{sid}/file/chunking` | JSON | Chunking preview for one file |
| GET | `/ingestion/sessions/{sid}/config` | JSON | Session config |
| PUT | `/ingestion/sessions/{sid}/config` | JSON | Update session config |
| GET | `/ingestion/sessions/{sid}/review` | JSON | Review bundle before ingest |
| POST | `/ingestion/sessions/{sid}/review` | JSON | Save review decisions |
| POST | `/ingestion/sessions/{sid}/save-zip` | JSON | Pack the session back to a zip |
| POST | `/ingestion/sessions/{sid}/export-path` | JSON | Write the session tree to a server dir |
| POST | `/ingestion/sessions/{sid}/ingest` | JSON `202` | Start an ingest job |
| GET | `/ingestion/jobs/{job_id}` | JSON | Job status |

**Create from path**

```bash
curl -s http://127.0.0.1:8000/ingestion/sessions/from-path \
  -H 'Content-Type: application/json' \
  -d '{"path":"./kb-source/KB_9.7.2026","name":"kb-9.7"}'
# {"session_id":"ab12cd34ef56","name":"kb-9.7","files":412,"source_path":"..."}
```

Relative paths resolve against the project root. Missing path → `404`, empty tree →
`400`.

**Upload a zip**

```bash
curl -s http://127.0.0.1:8000/ingestion/sessions -F 'file=@KB.zip'
```

Only `.zip`, max 500MB, and the archive is integrity-checked (`testzip`).

**Start ingest**

```bash
curl -s http://127.0.0.1:8000/ingestion/sessions/ab12cd34ef56/ingest \
  -H 'Content-Type: application/json' \
  -d '{"db_name":"kb_9_7","job_type":"full_rebuild","parent_scope":"sheet","dataset_mode":"default"}'
# 202 {"job_id":"...","db_path":".../data/kb_9_7"}
```

`job_type` is `full_rebuild` or `incremental` (anything else falls back to
`full_rebuild`). `parent_scope` is `sheet` or `document`. Poll
`GET /ingestion/jobs/{job_id}` for `status`, `started_at`, `completed_at`, `error`.

**Save back to zip**

```bash
curl -s -X POST http://127.0.0.1:8000/ingestion/sessions/ab12cd34ef56/save-zip
# {"zip_path":".../kb_zips/kb-9.7-ab12cd34ef56.zip","size":12345678,"files":412}
```

### KB snapshots — `/ingestion/kbs`

Versioned snapshots of a processed DB, with a repack step to produce the
distribution zip (this is the "supplements zip" flow).

| Method | Path | Body | Notes |
|---|---|---|---|
| GET | `/ingestion/kbs` | — | Snapshots + `kb_*.db` files + `kb_zips/*.zip` |
| POST | `/ingestion/kbs` | `{"label","notes","db_path","zip_path"}` | Create snapshot; `409` if label exists |
| GET | `/ingestion/kbs/{label}` | — | Snapshot detail (`404` if absent) |
| PUT | `/ingestion/kbs/{label}` | `{"notes"}` | Update notes |
| POST | `/ingestion/kbs/{label}/repack` | — | Rebuild the zip, returns `zip_path` + `size` |

```bash
curl -s -X POST http://127.0.0.1:8000/ingestion/kbs \
  -H 'Content-Type: application/json' \
  -d '{"label":"kb-9.7-2026-08-28","notes":"reason codes v1","db_path":"data/kb_9_7"}'

curl -s -X POST http://127.0.0.1:8000/ingestion/kbs/kb-9.7-2026-08-28/repack
```

Labels are sanitized; an unsafe label is a `400`. Note that `GET /ingestion/kbs`
returns `[]` on this deployment right now: no snapshots have been created yet, so
the list is empty rather than an error.

---

## 4. Zip browser — `/transparency/zip`

Preview an archive's contents before importing (separate from ingestion sessions).

| Method | Path | Returns |
|---|---|---|
| GET | `/transparency/zip` | HTML upload page |
| POST | `/transparency/zip/preview` | HTML preview; multipart `file`, max 200MB |
| POST | `/transparency/zip/ingest` | HTML result; form fields `zip_id`, `selected[]`, `job_type`, `parent_scope` |

The staged zip expires; a stale `zip_id` gives `400` with "Zip expired or not found".

---

## 5. Registry — packs, manifests, DVC pin

This is the versioning surface for taxonomy, canned text, prompts and retrieval
config, plus the processed-corpus pin.

| Method | Path | Notes |
|---|---|---|
| GET | `/registry` | HTML dashboard |
| GET | `/registry/api/packs` | Active pins + available versions per pack |
| GET | `/registry/api/packs/{kind}` | One pack's content |
| PUT | `/registry/api/packs/{kind}` | Save/bump a pack (`{"content": ..., "version": ..., "activate": ...}`) |
| POST | `/registry/api/reload` | Re-read packs from disk |
| GET | `/registry/api/taxonomy` | Active taxonomy pack |
| GET | `/registry/api/canned` | Active canned pack |
| GET | `/registry/api/prompts/{scope}` | `scope` = `orchestrator` or `guardrails` |
| GET | `/registry/api/manifests` | All index manifests |
| POST | `/registry/api/manifests/build?activate=false` | Build from current DB counts + pins |
| POST | `/registry/api/manifests/activate` | `{"manifest_id":"..."}`; `404` if unknown |
| GET | `/registry/api/active` | **Bootstrap payload** for the orchestrator |
| GET | `/registry/api/dvc` | Active processed snapshot pin; `404` if none |

`kind` accepts `schemas`, `taxonomy`, `canned`, `retrieval`, `prompts_orchestrator`,
`prompts_guardrails` (the last two map to `prompts` with a scope).

### `GET /registry/api/active` — start here

One call that returns everything a client needs to mirror the KB's current state:

```json
{
  "pins": {"schemas":"...","taxonomy":"...","canned":"...","prompts_orchestrator":"...","prompts_guardrails":"...","retrieval":"..."},
  "manifest": {...},
  "taxonomy": {
    "version": "...",
    "subspaces": ["cheque","facilities_person","..."],
    "subspace_map": {"cheque": "..."},
    "keyword_boost": {"cheque": 3.0},
    "ui_filter_groups": [...]
  },
  "canned": {...},
  "prompts": {...},
  "prompts_guardrails": {...},
  "dvc": {...}
}
```

### `GET /registry/api/dvc` — processed corpus pin

Returns the DVC-tracked snapshot pin for the processed KB, or `404` with
`"no DVC processed snapshot pin"`. This is how you tie a running KB back to an exact
corpus revision:

```bash
curl -s http://127.0.0.1:8000/registry/api/dvc | jq .
```

Live example (this deployment):

```json
{
  "corpus_date": "1405-08-20",
  "manifest_id": "kb-20261117T132201Z",
  "documents": 18,
  "chunks": 1204,
  "entity_lists": 12,
  "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dim": 384,
  "vectors": {"sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2-384": 1204}
}
```

Current active pins:

```
schemas 1.0.2 · taxonomy 1.0.3 · canned 1.0.2
prompts_orchestrator 1.0.7 · prompts_guardrails 1.0.1 · retrieval 1.0.3
```

Bump a pack:

```bash
curl -s -X PUT http://127.0.0.1:8000/registry/api/packs/taxonomy \
  -H 'Content-Type: application/json' \
  -d '{"content": {...}, "activate": true}'
```

`PUT` accepts a dict or a YAML/JSON string; invalid YAML is a `400`.

---

## 6. Documents, chunks, versions

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/documents` | HTML | Document list |
| GET | `/documents/{doc_id}` | HTML | Document detail |
| POST | `/documents/upload` | `302` redirect | Multipart `file`, form `domain`, `category` |
| POST | `/documents/{doc_id}/delete` | `302` redirect | Soft delete (`status=archived`) |
| GET | `/chunks` | HTML | Chunk list |
| GET | `/chunks/{chunk_id}` | HTML | Chunk detail |
| POST | `/chunks/{chunk_id}/verify` | HTML/JSON | Mark verified |
| POST | `/chunks/{chunk_id}/edit` | HTML/JSON | Edit chunk text |
| GET | `/versions` | HTML | Documents that have version history |
| GET | `/versions/{doc_id}` | HTML | Version history |
| POST | `/versions/{doc_id}/rollback/{version_id}` | `302` | Restore a snapshot |

Note the upload/delete/rollback endpoints redirect rather than return JSON — they
were built for the browser UI. Use `-i` or `-L` when scripting them.

---

## 7. Pipeline — `/pipeline`

Ingest straight from the configured source directory (no session).

| Method | Path | Returns | Notes |
|---|---|---|---|
| GET | `/pipeline` | HTML | Status page + last 50 jobs |
| POST | `/pipeline/run` | `302` | **Form-encoded**, not JSON: `job_type`, `parent_scope`, `source_dir` |
| GET | `/pipeline/status/{job_id}` | HTML | Job detail |

```bash
curl -s -X POST http://127.0.0.1:8000/pipeline/run \
  -d 'job_type=full_rebuild' -d 'parent_scope=sheet' -d 'source_dir=./kb-source/KB_9.7.2026'
```

---

## 8. Benchmarks — `/benchmarks`

| Method | Path | Notes |
|---|---|---|
| GET | `/benchmarks` | HTML dashboard |
| POST | `/benchmarks/run` | Single-turn run |
| POST | `/benchmarks/multiturn/run` | Multi-turn run |
| GET | `/benchmarks/status/{job_id}` | Job status |
| GET | `/benchmarks/result` | Latest result |
| GET | `/benchmarks/plot/{name}` | Plot image |
| POST | `/benchmarks/snapshot/create` | Snapshot the current results |
| GET | `/benchmarks/comparison` | HTML comparison |
| GET | `/benchmarks/comparison/data` | Comparison JSON |
| GET | `/benchmarks/snapshots` | List snapshots |
| GET | `/benchmarks/snapshots/{label}` | Snapshot detail |
| GET | `/benchmarks/snapshots/{label}/file/{name}` | File inside a snapshot |
| POST | `/benchmarks/ragas` | RAGAS run (needs `KB_RAGAS_API_KEY`) |
| GET | `/benchmarks/ragas/result` | RAGAS result |
| GET | `/benchmarks/massive` | HTML view |
| POST | `/benchmarks/massive/run` | Massive retrieval run |
| GET | `/benchmarks/massive/result` | Result |
| POST | `/benchmarks/iva/run` | IVA run |
| GET | `/benchmarks/iva/result` | Result |

---

## 9. Monitoring, metrics, transparency, cleanup

| Method | Path | Returns |
|---|---|---|
| GET | `/monitoring` | HTML dashboard |
| GET | `/monitoring/staleness` | HTML staleness report |
| GET | `/monitoring/metrics` | HTML metrics view |
| GET | `/metrics` | Prometheus-style metrics |
| GET | `/metrics/healthz` | Metrics health |
| GET | `/transparency` | HTML trace view |
| GET | `/transparency/{doc_id}` | HTML document trace |
| GET | `/transparency/api/raw/{doc_id}` | Raw parsed document |
| POST | `/transparency/parse-upload` | HTML parse preview |
| GET | `/transparency/questions` | HTML question set |
| GET | `/transparency/questions/json` | JSON question set |
| GET | `/transparency/questions/retrieval-check` | Retrieval check for the question set |
| GET | `/transparency/benchmarks/by-section` | Per-section benchmarks |
| GET | `/cleanup/qa` | HTML QA list |
| GET | `/cleanup/qa/{doc_id}/preview` | HTML preview |
| POST | `/cleanup/qa/{doc_id}/execute` | Execute cleanup for one doc |
| POST | `/cleanup/qa/execute-all` | Execute all |

## 10. Settings

| Method | Path | Returns |
|---|---|---|
| GET | `/settings`, `/settings/` | HTML settings form |
| POST | `/settings`, `/settings/` | Form save, HTML |

---

## Integration recipes

### Minimum viable retrieval client

```python
import httpx

async def search(query: str, top_k: int = 10, subspace: str | None = None):
    async with httpx.AsyncClient(base_url="http://kb:8000", timeout=60) as c:
        r = await c.post("/search/api", json={"query": query, "top_k": top_k, "subspace": subspace})
        if r.status_code >= 400:
            body = r.json()
            raise RuntimeError(f"{r.status_code}: {body.get('error')}")
        return r.json()["final_results"]
```

No auth header. If you add one later, this is the single place to insert it.

### Health gate before traffic

```bash
for p in 18000 8000 8200 8100; do
  printf '%s: ' "$p"; curl -s --max-time 8 "http://127.0.0.1:$p/health" | head -c 80; echo
done
curl -s --max-time 8 http://127.0.0.1:8000/ready
```

### Full corpus refresh (session → ingest → snapshot → repack)

```bash
SID=$(curl -s http://127.0.0.1:8000/ingestion/sessions/from-path \
  -H 'Content-Type: application/json' \
  -d '{"path":"./kb-source/KB_9.7.2026","name":"kb-9.7"}' | jq -r .session_id)

JOB=$(curl -s http://127.0.0.1:8000/ingestion/sessions/$SID/ingest \
  -H 'Content-Type: application/json' \
  -d '{"db_name":"kb_9_7","job_type":"full_rebuild"}' | jq -r .job_id)

until [ "$(curl -s http://127.0.0.1:8000/ingestion/jobs/$JOB | jq -r .status)" = "completed" ]; do sleep 10; done

curl -s -X POST http://127.0.0.1:8000/ingestion/kbs \
  -H 'Content-Type: application/json' \
  -d '{"label":"kb-9.7-2026-08-28","notes":"refresh","db_path":"data/kb_9_7"}'
curl -s -X POST http://127.0.0.1:8000/ingestion/kbs/kb-9.7-2026-08-28/repack
```

### Pin check

```bash
curl -s http://127.0.0.1:8000/registry/api/dvc | jq -c .
curl -s http://127.0.0.1:8000/registry/api/active | jq -c '.pins'
```

---

## Gaps worth knowing

- **Auth is opt-in.** Set `KB_API_KEY` to enable the gate; unset, the port is open.
  See the Authentication section.
- **HTML endpoints** — `/documents`, `/chunks`, `/versions`, `/pipeline`,
  `/monitoring`, `/transparency`, `/cleanup`, `/settings`, `/registry` are built for
  the browser UI. They work over HTTP but return templates, not JSON.
- **`/pipeline/run` takes form data**, unlike `/ingestion/.../ingest` which takes
  JSON. Easy to get wrong.
- **`/documents/upload` and `/documents/{id}/delete` redirect (302)** instead of
  returning JSON.
- **Redis is optional.** Search still works when it is down, just uncached and
  slower.
- **`top_k` is capped at 50** and `threshold` clamped to 0.0–1.0, silently.
