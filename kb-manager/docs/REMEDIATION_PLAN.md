# KB Manager — Evidence-Based Remediation Plan

> **Branch:** `feat/hyde-ragas` `0d29fb1` · **Baseline:** `master` `aa5c576` · **Merge-base:** `aa5c576` · **Python:** 3.14.4 · **Generated:** 2026-08-26 · **Status:** Planning only — no code modified

This plan verifies every reported finding, maps system flows, and provides a phased implementation roadmap an engineer can execute safely. Untracked working files (`.mcp.json`, `_check_db.py`, `create_small_dataset.py`, etc.) were **not overwritten**.

---

## 1. Executive Assessment

**Health: Yellow — functional but evaluation untrustworthy.** Retrieval works (reported Hit@5 84.2%, MRR 0.751) but 7× P1 paths make A/B comparisons meaningless. The severest risks are not model quality but benchmark integrity (fabricated `avg_rank 1.5`, typo no-ops, `Mock` silent) and concurrency (`asyncio.run` inside FastAPI, global `_index_cache`/`_sync_loop`).

**Top 5 confirmed risks:**

1. **Dense cache stale (F4/F5)** — `dense.py:116 fingerprint(texts)` ignores headings/context/model; `search.py:310 count-only` invalidation → incremental ingest keeps old embeddings; all post-ingest Hit numbers invalid.
2. **Lost ordinal + missing content (F1/F23)** — `search.py:250` stores `chunk_type` at index 5, `SearchResult.ordinal` always 0; `chunks.py:28 load_only` without `content` → N+1 lazy loads.
3. **`asyncio.run` in loop (F2/F29)** — `llm.py:103 OpenAIClient.generate` crashes inside FastAPI (`RuntimeError: asyncio.run() cannot be called from a running event loop`).
4. **Embeddings discarded (F6)** — `orchestrator.py:340` computes `embedder.embed_texts` but never persists; metric `chunks_embedded` fictional.
5. **Fabricated benchmark fallback (F24)** — `benchmarks.py:178` invents `avg_rank 1.5` when results missing → dashboard green on failure.

**Should `feat/hyde-ragas` merge now? `HOLD` and `SPLIT`.** Branch is only 2 commits ahead of master (9bde195 + 0d29fb1) but introduces duplicate HyDE and depends on unfixed cache invalidation. Split: (a) foundations Phases 0-2, (b) one canonical HyDE (`hyde.py`), (c) RAGAS route. No merge until a frozen-dataset A/B exists.

**Stabilization:** Freeze dataset → characterize baseline → land P1 fixes (Phases 1-2) → consolidate retrieval (Phase 3) → repair eval (Phase 4) → re-introduce HyDE behind flag with mocked tests (Phase 5).

---

## 2. System Map

**Entry points:** `web/app.py:lifespan → _get_index`, `web/routes/{search,benchmarks,pipeline,chunks}`, `pipeline/orchestrator.run_full_rebuild|incremental`, `cli.py`, `synthetic_generation/run_generation.py`.

**State:** `web/deps.py:12` global `db` singleton; `search.py:285 _index_cache/_index_cache_count/_sync_loop`; `semantic.py:83 _seen_questions set`; `benchmarks.py:52 _JOBS`; `hyde.py:71 _cache`.

```mermaid
graph TD
  Src[kb-source/*.xlsx 78 files] --> Parser[XlsxParser openpyxl/calamine]
  Parser --> Prep[PreprocessingPipeline\nclean→persian normalize→keywords]
  Prep --> Chunker[SemanticChunker\n_excel_rows sheet|document parent\n dedup set]
  Chunker --> DB[(SQLite documents/chunks\n355 docs 6208 chunks)]
  Chunker -.->|embed_texts discarded| Orchestrator[PipelineOrchestrator]
  DB --> Dense[load_or_build\nfingerprint(texts)→ data/dense_embeddings.npz\nDenseSemanticIndex MiniLM 384d]
  Query --> Tok[_tokenize ZWNJ→space + char 3-grams]
  Tok --> BM25[BM25 k1=1.5 b=0.75]
  Tok --> Dense
  Query -->|if KB_HYDE_ENABLED| HyDE[HyDEGenerator httpx 15s → embed]
  BM25 & Dense & HyDE --> RRF[RRF k=60]
  RRF --> Rerank[CrossEncoder mmarco 50->rerank pool]
  Rerank --> Resp[SearchSteps final_results]
  Bench[BenchmarkRunner 6 formats] --> Metrics[Ranx / RAGEvaluator / RagasEvaluator]
  Metrics --> Dashboard[/benchmarks, /comparison\nfallback invents 1.5 if missing/]
  Web[FastAPI to_thread + _sync_loop.run_until_complete] --> BM25 & Dense & Rerank & HyDE
```

**Caches & boundaries:** Disk `.npz` keyed only by `fingerprint(texts)`; memory `_index_cache` keyed by count; sync embed/reranker vs async routes via `asyncio.to_thread` + `_sync_loop` (illegal nesting). Source→chunk: file → `compute_content_hash` → skip if equal → `chunker.chunk(sheets, doc_type)` → dedup set → `parent_id_map` → commit. Query→response: `normalize → BM25(top_k*3) + Dense(top_k*3) + [HyDE top_k*3] → RRF → rerank(top_k)` → `SearchSteps`. Benchmark: `test_questions.json → apply_format → search_knowledge_base_sync → rank → aggregate → benchmark_results.json + ir_metrics.json → plots`.

---

## 3. Findings Matrix

*Verified on HEAD 0d29fb1; line numbers match that checkout.*

|ID|Disposition|Sev|Evidence|Affected|Runtime path|Resolution|Tests|Phase|Blocker|
|--|--|--|--|--|--|--|--|--|--|
|A1 ordinal lost|**Confirmed**|P1|`search.py:238-249` tuple index 5=`chunk_type` str; `SearchResult.ordinal` `cd[5] if int else 0` always 0|HEAD+master|every search/benchmark|dataclass `ChunkData` propagating `ordinal`|test_ordinal_propagated|1|YES|
|A2 asyncio.run|**Confirmed**|P1|`llm.py:103 OpenAIClient.generate→asyncio.run` + `evaluation/benchmark.py:173 AsyncBenchmarkRunner→asyncio.run` called from `benchmarks.py:263 create_task` on running loop → RuntimeError|both|web benchmark, synthetic gen|Async clients `async def generate`, routes `await`|test_llm_async_from_fastapi|1|YES|
|A3 reranker truncates|**Confirmed**|P1|`search.py:437 candidates[:50]` → `reranker.py:116 [:min(top_k*3=30)]` drops 20 when top_k=10|both|warm search|Single `RERANKER_POOL=50`, remove inner slice|test_pool_semantics|1|YES|
|A4 fingerprint ignores context|**Confirmed**|P1|`dense.py:116 hash(len+text)` ignores titles/headings/chunk_types/model/use_context|both|incremental ingest|Hash `model_name+use_context+contextual_texts` + version bump|test_cache_heading_change|2|YES|
|A5 invalidation count|**Confirmed**|P1|`search.py:310 if count==cache_count return` — swap delete+add same count stale|both|incremental|Fingerprint or `max(updated_at)` check|test_same_count_invalidation|2|YES|
|A6 embeddings discarded|**Confirmed**|P1|`orchestrator.py:340 embs=embed_texts` never stored; DB no vector col; `search._build_index` rebuilds |both|every rebuild|Delete embed block or persist `.npz` at ingest; fix metric|test_ingest_metric|2|YES|
|A7 query_reform HyDE halluc|**Confirmed dead**|P2|`query_reform.py:144 dense_index.embed_query` nonexistent (`dense` has `_encode/search(str)`), no caller|both|none if wired|Delete dead HyDE, keep `hyde.py`|grep dead import|5|NO|
|A8 drop by value|**Confirmed**|P2|`query_formats.py:126 drop=set(sample(words[1:-1]))` removes all duplicates|both|benchmark gen|Sample positions, drop by index|test_drop_position|4|NO|
|A9 typo identity|**Confirmed**|P2|`query_formats.py:57 7 self-maps` ("باید"→"باید")|both|benchmark typo|Provide real variants|test_typo_changes|4|NO|
|A10 BM25 dup length|**Partially**|P2|`search.py:248 content+kw*3` raises `dl`, weight length-dependent not 3×|both|keyword_only|Field weight `bm25(content)+w*bm25(keywords)`|test_keyword_weight|3|NO|
|A11 dtype None→float16|**Confirmed**|P1|`reranker.py:66 float16 if _device!="cpu"` true when None → float16 on CPU|both|cold start CPU host|Check `cuda.is_available()`|test_dtype_cpu|1|YES|
|A12 ZWNJ dead regex|**Partially**|P3|`search.py:55 \u200c→" "` then regex class includes `\u200c` never matches|both|Persian queries|Remove from class, unify norm|—|3|NO|
|B13 dual HyDE|**Confirmed**|P2|`hyde.py:35 active` vs `query_reform.py:80 dead`; `search.py:22` imports former|both|maintenance|Keep `hyde.py`, delete other|test_single_hyde|5|YES|
|B14 Gemma tokens|**Confirmed**|P2|`query_reform.py:18 <start_of_turn>` with default gpt-4o-mini; numbering duplicate, `{{` escaped|both|MultiQuery unused|Backend-specific templates|—|5|NO|
|B15 FaMTEB fallback IDs|**Confirmed**|P2|`famteb.py:161 doc_id fallback f"d_{i}"` masks schema mismatch|both|famteb load|Validate schema, raise on missing field|test_famteb_schema|4|NO|
|B16 identity typos|**Confirmed**|P3|`persian.py:173 4 identity maps`|both|normalize|Delete|—|7|NO|
|B17 stop-word space|**Confirmed dead**|P3|`pipeline.py:102 " manner"` never tokenized|both|keyword extraction|Delete|—|7|NO|
|B18 heuristic name|**Confirmed**|P2|`metrics.py:225 RAGEvaluator` = overlap not LLM, vs `ragas_metrics.py:29` real RAGAS|both|/ragas UI|Rename `HeuristicOverlapEvaluator`|—|4|NO|
|C19 unused template|**Confirmed dead**|P3|`dense.py:24 _CONTEXT_TEMPLATE` not read|both|—|Delete or use|—|7|NO|
|C20 unused IDF|**Confirmed dead**|P3|`search.py:319 _compute_idf` no caller|both|—|Delete|—|7|NO|
|C21 beam placeholder|**Confirmed dead**|P3|`query_reform.py:242 generate_beam→generate` ignores beam_size|both|—|Delete param|—|7|NO|
|C22 greedy JSON|**Confirmed**|P2|`query_reform.py:219 re.search(r'\[.*\]',DOTALL)` greedy|both|synthetic gen|Balanced parse|test_json_extract|5|NO|
|C23 load_only missing content|**Confirmed**|P1|`chunks.py:28 load_only without content` → N+1 lazy loads|both|/chunks page|Include content or `substr`|test_chunks_page|1|YES|
|C24 fabricated metrics|**Confirmed**|P1|`benchmarks.py:178 avg_rank 1.5` hallucinated|both|/benchmarks empty|Return null + banner|test_missing_null|4|YES|
|C25 mock silent|**Confirmed**|P2|`llm.py:287 default backend mock` canned Persian|both|synthetic gen|Require `KB_ALLOW_MOCK` flag|test_mock_gate|5|NO|
|D26 global DB|**Confirmed**|P2|`web/deps.py:11 config=load_config(); db=Database` frozen at import|both|tests & ingest|FastAPI Depends `get_db`|test_db_override|6|YES|
|D27 config drift|**Confirmed**|P2|`config dedup false` vs `SemanticChunker true` vs docs 6208 vs `source_dir data vs kb-source`|both|repro|Align defaults to true/kb-source|test_config_agree|3|YES|
|D28 parent O(n²) fragile|**Confirmed**|P1|`orchestrator.py:395-421` double loop `(ordinal,chunk_type)` matching|both|rebuild multi-sheet|keyed `parent_key→id` map before add|test_parent_key|2|YES|
|D29 global cache unsafe|**Confirmed**|P1|`search.py:285,465 _index_cache/_sync_loop` no lock, shared across threads|both|concurrent search|Lock + `anyio`/`to_thread` safe pattern|test_concurrent_build|3|YES|
|D30 jobs leak|**Confirmed**|P2|`benchmarks.py:52 _JOBS` never evicted|both|many benchmarks|TTL LRU 100/24h|test_job_ttl|6|NO|
|D31 error_log misuse|**Confirmed**|P2|`pipeline.py:77 error_log=f"parent_scope={parent_scope}"`|both|pipeline history|Add `job_config JSON` column|test_error_log|4|NO|
|D32 manual route copy|**Confirmed**|P2|`web/app.py:79 copies APIRoute` instead of `include_router`|both|openapi|Use include_router 0.141.1|—|6|NO|
|D33 print in FaMTEB|**Confirmed**|P3|`famteb.py:199 print`|both|famteb|logger|—|6|NO|
|D34 duplicated maps|**Confirmed**|P2|Persian map in 3 modules, tokenizers duplicated|both|norm|Centralize `persian_norm`|—|3|NO|
|D35 broad except|**Confirmed**|P2|`except Exception: return {}` swallows stack|both|benchmarks/hyde|Narrow exceptions, propagate 500/503|test_exception|4|NO|
|D36 mutable dedup|**Confirmed**|P2|`semantic.py:83 _seen_questions` persists across docs/jobs|both|full rebuild|Reset per job, make stateless|test_dedup_isolation|2|NO|
|E drift 90→84%|Requires runtime|P2|v2 90% vs HEAD 84.2% on regenerated `test_questions.json:72f6bdf` — no frozen checksum|—|benchmark history|Freeze dataset sha256|0|—|
|E HyDE no A/B|Confirmed|P2|No `tests/test_hyde*`, http not mocked|HEAD|—|Add mocked A/B protocol|5|YES|
|E 15 branches|Confirmed|P3|`git branch --merged master` 17 branches, only `feat/hyde-ragas` unmerged|—|—|Tag & delete|7|NO|
|E FaMTEB no run|Confirmed|P2|No `results/famteb*.json`|—|—|10-sample smoke|4|NO|
|E RAGAS done but silent|Confirmed|P2|PLAN marks ✅ but no stored result|—|—|Require stored result|4|NO|

## 4. Dependency Analysis

**Must precede HyDE:** F4/F5 cache → otherwise `hyde_vec dot dense._matrix` compares stale; F2 async → HyDE 15s blocks loop; F3 pool → HyDE gain conflated.

**Group:** Async `F2+F29` (touch `_sync_loop/_get_index`) — one PR. Index `F4+F5+F6+F28+F36+F27` — all require rebuild; splitting doubles rebuilds. Eval `F8,F9,F24,F35,F15,F18`.

**Separate:** Retrieval norm (`F10,F12,F34`) from HyDE (`B13,B14`) — first changes all BM25 scores, second adds 3rd leg; mixing hides gain. Ops (`D26,D30,D32`) from branch cleanup (Phase 7) — different risk.

**Critical path:** 0 baseline → 1 crashes → 2 async → 3 index → 4 retrieval → 5 eval → 6 HyDE → 7 ops → 8 perf.

## 5. Phased Roadmap

### Phase 0 — Reproducibility & Safety Net **M, 1w**
*Goal:* Immutable ground truth. *Includes:* E drift, F27 baseline.
*Files:* `tests/conftest.py`, `scripts/record_baseline.py`, `data/test_questions.sha256`, `pyproject.toml` markers.
*Tasks:* Tag `baseline-0d29fb1`; `sha256sum test_questions.json`; `pip freeze`; `tests/test_characterization.py` with mocked MiniLM fixture vectors; split `unit` vs `models` markers; doc schema.
*Migration:* none. *Invalidates benchmarks:* no. *Acceptance:* `pytest -m "not models"` green without network. *Rollback:* `git checkout baseline-0d29fb1`. *Parallel:* yes.

### Phase 1 — Immediate Correctness & Crash **S, 3d**
*Goal:* Wrong numbers/crashes fixed. *Findings:* F1 ordinal, F11 dtype, F23 content, F3 slice prep.
*Files:* `search.py`, `reranker.py`, `web/routes/chunks.py`.
*Tasks:* 7-tuple dataclass propagating ordinal; dtype `float16 if cuda else float32`; `chunks` include content; `RERANKER_POOL=50`.
*Migration:* none (benchmark invalidated). *Acceptance:* `test_ordinal_propagated` + SQL log shows single SELECT.

### Phase 2 — Async Boundary Repair **M, 1w**
*Findings:* F2, F29. *Files:* `llm.py`, `hyde.py`, `search.py:465`, `evaluation/benchmark.py`.
*Tasks:* `async def generate`, routes `await`, `asyncio.Lock` for `_index_cache`.
*Migration:* rolling. *Tests:* concurrent 20× search, benchmark from async route mocked.

### Phase 3 — Index & Ingestion Correctness **L, 2w (major rebuild)**
*Findings:* F4,F5,F6,F28,F36,F27,F31. *Files:* `dense.py`, `search.py`, `orchestrator.py`, `chunker/semantic.py`, `config.py`, `models/database.py`.
*Tasks:* Fingerprint `model+use_context+contextual_texts`; invalidation via fingerprint/`max(updated_at)`; delete or gate embed block; parent key map; dedup defaults true/kb-source; `job_config JSON` column.
*Migration:* **corpus rebuild required, dense cache bump**. Auto-detect old `.npz` without fingerprint → rebuild. *Acceptance:* same-count content update reflected.

### Phase 4 — Retrieval Consolidation **M, 1w**
*Findings:* F10,F12,F34,F27. *Files:* `persian.py` canonical, `search.py`.
*Tasks:* Centralize `normalize_persian`; BM25 field weight `w=3.0` via score param.
*Invalidates benchmarks:* yes.

### Phase 5 — Evaluation Integrity **M**
*Findings:* F8,F9,F24,F35,F15,F18,F31. *Files:* `query_formats.py`, `benchmarks.py`, `famteb.py`, `metrics.py`.
*Tasks:* Fix drop indices, real typo map, null not 1.5, validate FaMTEB schema, rename `RAGEvaluator`.
*Bumps dataset version.*

### Phase 6 — HyDE Consolidation **M**
*Findings:* B13,B14,F22,F25,F7. *Files:* `hyde.py` keep, delete `query_reform.HyDE`, strict JSON via decoder, httpx retry, cache key model.
*Tests:* mocked `httpx.MockTransport` success/malformed/timeout/empty, disabled, dim mismatch.
*Gate:* Disabled by default, manifest records `hyde_enabled`.

### Phase 7 — Operational Hardening **M**
*Findings:* D26,D30,D32,D33. *Files:* `web/deps.py`, `web/app.py`, `benchmarks.py`.
*Tasks:* `get_db` Depends, job TTL 24h, correct router include, structured logging.
*Tests:* `TestClient` override.

### Phase 8 — Dead Code & Branch Cleanup **S**
*Findings:* C19-21, B16,B17, D33 etc. Delete `rg` verified dead.

### Phase 9 — Performance **XL (after 0-8)**
*Goal:* Measure only after correctness. *Tasks:* Profile, pools 50→30→20, batch 64→128, quantization conditional; measure warm-up/memory; controlled report.

## 6. Test Strategy

|Category|What|Markers|
|--|--|--|
|Unit|ordinal, drop by position, typo guarantee, ZWNJ, fingerprint context, dtype, pool, JSON strict, config precedence, dedup reset|not models|
|Async|LLM from async route, concurrent build, cancellation, concurrent ingest, job TTL|not models|
|Database|chunk page content, parent key, same-count invalidation, rollback, job_config|not models|
|HyDE|headers without secret leak, malformed, timeout, disabled, cache hit/miss, 2 hypotheses dim check|not models (mocked)|
|Benchmark|no 1.5, fixed-seed determinism, checksum, per-format, error propagation, RAGAS missing|not models|
|Integration (models)|Ingest 5-file fixture → search 6 formats → update same count → HyDE disabled vs mocked|models (requires DL, nightly)|

CI default `pytest -m "not models"` — no download, no network, <2m.

## 7. Benchmark Protocol

**Frozen:** commit baseline, `kb_test.db` hash, corpus hash, `KB_CHUNK_PARENT_SCOPE=sheet`, `KB_CHUNK_DEDUP=true`, MiniLM 5.7.0 revision, reranker revision, `test_questions.json` sha256, `random.seed 42`, concurrency 1, warm-up 10 excluded, top_k=5/10, LLM model `gpt-4o-mini` temp 0.3 max 150 timeout 15s retry 1.

**Candidates:** master w/o HyDE (stabilized), w/ HyDE mocked, pools 50/30/20.

**Report:** hit@1, hit@5, MRR, avg_rank, p50/p95/p99 latency, per-format 6 splits, HyDE request count/failure/cache-hit, cost, variance over 3 runs.

**Success thresholds (pre-declared):** keyword-only hit@5 ≥+8pp 95% CI, reworded MRR ≥-2pp, p95 latency <6.0s, failure <1%. Overall hit@5 alone insufficient.

## 8. Migration & Compatibility

|Change|DB|Re-ingest|Cache|Benchmark|Rollout|
|--|--|--|--|--|--|
|PR1 ordinal/dtype/chunk page|no|no|no|yes|restart|
|PR3 fingerprint/invalidation/parent/dedup|add `job_config JSON`| **yes** full rebuild ~30m|**yes** delete `.npz`|no but old results incompatible|detect old `.npz` missing fingerprint → rebuild; backup `.db`|
|PR4 retrieval weight|no|no|yes|yes|restart|
|PR5 eval dataset|no|no|no|**yes** version bump|archive `versions/`|
|PR6 HyDE|no|no|no|extra `hyde_enabled` in schema v2|flag|

**Rollback:** `git revert <tag>`; restore `dense_embeddings.npz` backup; snapshot `.db` pre-rebuild.

## 9. Pull-Request Sequence

1. **PR0 characterization+frozen dataset** — tests + sha256, no behavior.
2. **PR1 ordinal+dtype+chunk page** — crash fixes, invalidates benchmarks.
3. **PR2 async boundary** — `asyncio.run` → `await`.
4. **PR3 index/invalidation/parent/dedup** — major rebuild, version bump.
5. **PR4 retrieval normalization** — separate from PR3 to isolate BM25 effect.
6. **PR5 eval integrity** — dataset version bump.
7. **PR6 HyDE canonical** — one impl, mocked tests, disabled default.
8. **PR7 operational** — DI, TTL, router.
9. **PR8 dead code + branch hygiene** — `rg` verified.
10. **PR9 perf 50→30** — measured protocol.

## 10. Branch Strategy

- `feat/hyde-ragas` (2 ahead): rebase into three PRs (PR6). Cherry-pick `9bde195:hyde.py` → PR6. Tag `hyde-ragas-backup-0d29fb1` before recreate. Create `feat/hyde-clean` from `master` with PR6 commits only.
- `git branch --merged master` 17 branches contained — safe. Verify `git merge-base --is-ancestor <b> master`. Tag `archive/<branch>-<short>` push tags then `git branch -d; git push origin --delete`.
- Keep roadmap/docs branches until PR5.

## 11. Definition of Done

- [ ] All 36 IDs dispositioned.
- [ ] P1 fixed/waived.
- [ ] `pytest -m "not models"` green air-gapped.
- [ ] `pytest -m models` documented nightly.
- [ ] No 1.5 fallback; schema v2 includes checksum/fingerprint/revisions.
- [ ] Frozen dataset + sha256 committed.
- [ ] Same-count invalidation passes.
- [ ] Async verified under FastAPI, no `asyncio.run` in route path.
- [ ] One HyDE impl; failures observable via health/log.
- [ ] Controlled A/B report reproduced twice.
- [ ] Config defaults agree across code/tests/docs.
- [ ] Corpus rebuild doc & rollback tested.
- [ ] 17 branches archived, deleted.
- [ ] Final `docs/BENCHMARK_REPORT.md` (before/after, cost, rollback).

## 12. Open Questions

1. **PGVector vs SQLite vectors?** Blocks PR3 persist-vs-rebuild choice. Default keep rebuild-at-search until ADR.
2. **Quantization hardware?** Blocks PR9 claims. Default measure on current 2×CPU, no merge until `hardware.md`.
3. **FaMTEB CI vs ad-hoc?** Blocks gate. Default manual + 10-sample smoke only.
4. **RAGAS credentials policy?** Blocks reproducibility. Default local manual, CI mocked.
5. **Corpus license for `versions/kb_export.json` public Pages?** Blocks snapshot retention. Default local/LFS.

*Do not implement code/tags/deletions until PR0 reviewed.*
