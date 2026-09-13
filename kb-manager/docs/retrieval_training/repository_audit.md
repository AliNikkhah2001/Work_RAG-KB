# Repository Audit (from code inspection, 2026-09-12)

Source of truth is the code cited per row; the brief's "verified facts" were
re-checked and confirmed unless noted.

## 1. Schema trace

- ORM: `kb_manager/models/database.py` — `Chunk` / `Document` / `DocumentVersion` /
  `IngestionJob` / `RetrievalLog`; `Base.metadata` is what `Database.create_tables()` builds.
- `Chunk` columns: `id` String(36) uuid (`_uuid`), `document_id` FK documents.id
  CASCADE, `parent_id` FK chunks.id SET NULL, `ordinal` int, `chunk_type` str(32)
  default `"body"`, `content` Text, `heading_path` str(1024), `keywords` JSON list,
  `token_count`, `embedding_model` nullable, `embedding` Vector(384) on postgres /
  JSON fallback on sqlite (`_HAS_PGVECTOR`), `quality_score` nullable,
  `is_verified` bool, `doc_metadata` mapped to column `"metadata"` JSON,
  `created_at`.
- `Document.metadata` likewise lives in attribute `doc_metadata`, column name
  `"metadata"`.
- Chunk production: `kb_manager/chunker/semantic.py::SemanticChunker._chunk_excel_rows`
  — headers lowercased into `metadata["fields"]` (`fields[header.lower()]`);
  children get `metadata["parent_key"]` (sheet name, or `"document"` when
  `parent_scope == "document"`); parents are `chunk_type=f"{doc_type}_parent"`
  with `is_parent=True`, `parent_scope`, `child_count`.
- QA gate (`doc_type == "qa_pair"`): row is skipped unless it has a question key
  (`question`/`پرسش`/`سوال`/`متن سوال`/`متن_سوال`) AND an answer key (`answer`/
  `briefanswer`/`پاسخ`/`متن پاسخ`/`متن_پاسخ`/`پاسخ کوتاه`/`پاسخ کامل`);
  counter `skipped_incomplete` surfaces as `chunks_skipped_incomplete` in the
  orchestrator result and `kb-manager ingest` output (`kb_manager/cli.py`).
- Keywords: `fields["keyword"]` (else `fields["keywords"]`) split on `،` (U+060C).
- Dedup normalizer `SemanticChunker._normalize_question` (static, confirmed):
  strip → Arabic yeh `ي`→`ی`, kaf `ك`→`ک`, alif-wasla `ٱ`→`ا`, ZWNJ→space,
  strip `؟`/`?`, collapse `\s+`. NO lowercasing. `dedup_questions=True` default
  (`ChunkingConfig`, env `KB_CHUNK_DEDUP_QUESTIONS`).
- Search index excludes parents: every chunk query filters
  `~Chunk.chunk_type.like("%_parent")` (`kb_manager/web/routes/search.py`).

## 2. Pipeline + pool sizes

- Entry: `search_knowledge_base(query, top_k, keyword_boost)` (async) +
  `search_knowledge_base_sync` wrapper (thread-pool safe); `SearchSteps` carries
  `bm25_results / semantic_results / dense_results / merged_candidates / final_results`.
- Leg 1 BM25: two indexes (content + keywords), single-query score =
  `content + keyword_boost * kw`; multi-beam (`KB_SYNONYM_BEAM=5`) max-pools per doc.
  Pool per index: `top_k * 3`.
- Leg 2 dense: pgvector (`embedding <=> query`, needs >100 non-null embeddings)
  else file-backed `DenseSemanticIndex`; beam max-pool; pool `top_k * 3`.
- Leg 3 HyDE: only when `KB_HYDE_ENABLED=true` AND an API key is set.
- Merge: RRF `k=60` over 2–3 legs → `hybrid_score`.
- Rerank: full pool `KB_RERANKER_TOP_K=100` scored by
  `CrossEncoderReranker.rerank(query, [dict], top_k)` (`kb_manager/reranker.py`,
  factory `get_reranker(model_name, batch_size, device)`), then
  `alpha * norm(rerank) + (1-alpha) * norm(rrf)` ordering (`alpha=0.7`),
  then pin-guard (merged-top3 + BM25-top3 may not fall below final rank 10).
  Short-query fallback: max rerank < 0.2 with ≤4 tokens → BM25 top-k.
- Interfaces to reuse: `DenseSemanticIndex + load_or_build(cache_path, ids, texts,
  titles, headings, chunk_types, model_name, ...)` (`kb_manager/dense.py`);
  metrics `P/R/Hit/MRR/nDCG/MAP` (`kb_manager/evaluation/metrics.py`,
  `RetrievalMetrics.compute_all`, ranx-backed `RanxRetrievalEvaluator`);
  `BenchmarkRunner`/`AsyncBenchmarkRunner` multi-gold aware
  (`kb_manager/evaluation/benchmark.py`).

## 3. Env table (code-defined defaults)

| Var | Default | Used in |
|---|---|---|
| `KB_KEYWORD_BOOST` | `3.0` (clamped 0–10) | search.py |
| `KB_RERANK_FUSION_ALPHA` | `0.7` | search.py |
| `KB_SYNONYM_ENABLED` | `true` | search.py → query_expansion beam |
| `KB_SYNONYM_BEAM` | `5` | search.py |
| `KB_RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | search.py |
| `KB_RERANKER_TOP_K` | `100` | search.py |
| `KB_HYDE_ENABLED` / `KB_HYDE_LLM` / `KB_HYDE_API_KEY` / `KB_HYDE_BASE_URL` | `false` / `gpt-4o-mini` / `OPENAI_API_KEY` / `OPENAI_BASE_URL` | search.py, config HyDEConfig (`KB_HYDE_NUM`→1) |
| `KB_EMBED_MODEL` / `KB_EMBED_DIM` / `KB_EMBED_BATCH` | `paraphrase-multilingual-MiniLM-L12-v2` / `384` / `64` | config.py |
| `KB_EMBED_DEVICE` / `KB_DEVICE` / `KB_RERANKER_DEVICE` | `cpu` | config.py |
| `KB_DB_MODE` (+ `KB_USE_PGVECTOR` legacy) / `KB_DB_URL` / `KB_SQLITE_PATH` | `sqlite` / override / `./data/kb_test.db` | config.py `_resolve_db_mode_and_url` |
| `KB_DB_HOST/PORT/NAME/USER/PASSWORD/ECHO` | localhost/5432/kb_manager/postgres/postgres/false | config.py |
| `KB_CHUNK_STRATEGY/MAX/PARENT_MAX/PARENT_SCOPE/DEDUP_QUESTIONS` | semantic/512/1536/sheet/true (`min_tokens=100`, overlap 50 in code) | config.py, chunker |
| `KB_XLSX_ENGINE` | `auto` | config.py |
| `KB_RAGAS_LLM/EMBED/API_KEY/BASE_URL/K` | gpt-4o-mini/text-embedding-3-small/''/''/5 | config.py |
| `KB_SOURCE_DIR` / `KB_OUTPUT_DIR` / `KB_WEB_HOST` / `KB_WEB_PORT` | kb-source or data / data/processed / 0.0.0.0 / 8000 | config.py |
| Dense cache model constant | same MiniLM-L12-v2 id | `kb_manager/dense.py::_MODEL_NAME` |

## 4. Cache / fingerprint

- Disk: `data/dense_embeddings.npz` (`ids`, `vectors`, `fingerprint`);
  `DenseSemanticIndex.fingerprint(texts, titles, headings, chunk_types, model_name,
  use_context)` = sha256 over model id + use_context flag + contextual texts
  (`Title:/Heading:/Type: Q&A/Content:`), so title/heading/model switches
  invalidate (F4). `load_or_build` skips caching when a test `embed_fn` is passed.
- Memory: `search.py::_index_cache` triple `(payload, chunk_count, fingerprint)`
  + `asyncio.Lock` double-checked rebuild; fast path on count match still
  recomputes the fingerprint to catch same-count content drift (F5);
  `_invalidate_index_cache()` drops it.

## 5. Test layout

- `tests/`: `conftest.py`, `test_characterization.py`, `test_chunker.py`,
  `test_cli.py`, `test_embedder.py`, `test_evaluation.py`, `test_parsers.py`,
  `test_pipeline.py`, `test_preprocessor.py`, `test_qa_massive.py`.
- `pyproject.toml [tool.pytest.ini_options]`: `testpaths=["tests"]`,
  `asyncio_mode="auto"`, NO custom markers — new tests use plain functions and
  `pytest.mark.skip(reason=...)` only.
- `tests/conftest.py::db_engine` builds a hand-written in-memory schema
  (`documents(id, source_path, file_hash, created_at)`,
  `chunks(id, document_id, content, heading_path, chunk_index, token_count, embedding)`)
  that does NOT match the ORM → new retrieval-training tests avoid it and must
  build tables via `Database.create_tables()` when a DB is needed (none of the
  scaffolding tests need a DB).

## 6. Freeze procedure

- Script `freeze_baseline.py`: pins `KB_DB_URL` to
  `sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db`, reads
  documents + `chunk_type` histogram + `qa_pair` count, sha256-hashes the DB
  file and `data/dense_embeddings.npz`, `git rev-parse HEAD`, snapshots all
  `KB_*` env, writes `artifacts/retrieval_training/baseline_freeze.json`.
- Frozen content (read 2026-09-12): `code_rev 96438c1…`, 21 docs,
  histogram `{body:552, qa_pair:508, qa_pair_parent:5, staff_profile:16,
  staff_profile_parent:3}`, dense MiniLM-L12-v2, reranker mMiniLMv2-L12,
  `retrieval_config {rrf_k:60, rerank_top_k:100, keyword_boost:3.0,
  synonym_beam:5, fusion_alpha:0.7, pin:"merged-top3+bm25-top3<=10"}`,
  `env.KB_DB_URL` → `data/kb_9_7_2026.db`. This file is the evaluation baseline
  for stage 5 (`compare_against_baseline`).
