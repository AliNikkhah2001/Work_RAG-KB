# KB Version Snapshots

Archived KB exports — each `v*/` is a full `kb_export.json` + `manifest.json` at a pipeline checkpoint.

| Version | Date | Chunks | Pipeline | Notes |
|---------|------|--------|----------|-------|
| `v1` | 1405-03 | 12 MB | Initial file-based | Baseline |
| `v2` | 1405-04 | 6.5 MB | File-based rerank fix | Latency 439ms rerank |
| `v4_retrieval` | 1405-05 | 9.2 MB | Retrieval v4 (RRF tuning) | Hit@5 0.536 |
| `v6` | 1405-06-15 | 3.6 MB | pgvector HNSW migration | 6593 chunks |
| `v7_iva_1405-05-31` | 1405-05-31 | 2.7 MB | IVA 15 Q fix (`RERANKER_TOP_K 50→100`) | Hit@5 73.3% on IVA 15 |
| `v10_1405-06-23` | 1405-06-23 | 3.0 MB | 1405-06-23 snapshot | Current KB source |
| `v11_1405-06-23` | 1405-06-23 | 1.4 KB | Tiny placeholder | Manifest only |

**Gaps:** `v3`, `v5`, `v8`, `v9` never materialized — pipeline jumped from file-based (`v2`) to `v4_retrieval` (RRF), then to pgvector (`v6`). `v8` exists conceptually as the live pgvector KB (6593 chunks, RTX 6000 Ada, 18.4s avg) but was not snapshotted to `versions/`. Naming is `vN` for major pipeline changes, `vN_date` for source-date snapshots, `vN_suffix` for feature branches (`v4_retrieval`, `v7_iva`).

**Canonical:** `components/knowledgebase/kb-manager/data/` (live), `eval/results/` (benchmarks). Snapshots are read-only archival.
