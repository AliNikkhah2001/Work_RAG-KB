# Knowledge Base Version 11 — 1405-06-23 (Post-Chunkfix Rebuild)

## Overview
This is version 11 of the Persian RAG Knowledge Base, built from the source directory `data/v10_source/1405-06-23` (41 xlsx files → 35 documents after dedup).

## Changes from v10 (chunkfix)
- **Overlap removal**: `_chunk_excel_rows` no longer applies overlap prefix to row-wise atomic chunks (`kb_manager/chunker/semantic.py:426`)
- **Keyword stripping**: Keywords removed from QA content strings; `"keyword"`/`"keywords"` added to `seen` exclusion set (`kb_manager/chunker/semantic.py:485-486,494`)
- **Dense context prefix**: `use_context` default changed from `True` to `False` in `kb_manager/dense.py:124,257` — content-only embeddings, no Title/Heading/Type prefix
- **Fingerprint inclusion**: `use_context` flag now included in dense fingerprint, so npz auto-rebuilds on mismatch

## Corpus Statistics
- **Source files**: 41 xlsx files in `data/v10_source/1405-06-23`
- **Documents**: 35 (after duplicate removal)
- **Chunks**: 2,227 (overlap-free, keyword-free, context-stripped)
- **QA pairs**: 702 (across all documents)

## Embeddings
- Dense embeddings: `data/dense_embeddings.npz` (auto-rebuilt with `use_context=False`)
- Fingerprint includes `use_context` flag for auto-rebuild detection

## Benchmark Results (v10 Post-Fix)
- **Hit@5**: 0.8585 (613/659 evaluated gold rows)
- **Funnel**: A=25 (retriever miss) | A'=8 (fusion loss) | B=13 (reranker loss) | D=613 (success)
- **Per-file performance**:
  - Company_CRM_Questions: hit@5=0.916 (304/332)
  - IndividualCRMQuestions: hit@5=0.933 (167/179)
  - ChequeQuestions: hit@5=0.521 (61/117) — keyword-stuffing + twin crowding
  - DisputeQuestions/Etebariti: hit@5=1.000

## Artifacts
- `artifacts/retrieval_training/retrieval_failures_v10_1405-06-23.jsonl` — 659 gold rows
- `artifacts/retrieval_training/live_dashboard.py` — live TUI with funnel + per-file breakdown
- `artifacts/retrieval_training/agent2_v10_1405_06_23.py` — benchmark with A/A' classification

## Directory Structure
```
data/
  kb_1405_06_23.db          — SQLite KB (rebuilt, overlap-free)
  kb_1405_06_23_pre_chunkfix.db  — pre-fix backup
  dense_embeddings.npz      — post-fix embeddings (use_context=False)
  dense_embeddings_pre_chunkfix.npz  — pre-fix backup
  v10_source/
    1405-06-23/             — source xlsx files
  dense_embeddings_v9_backup.npz  — v9 backup
artifacts/
  retrieval_training/       — benchmark scripts and results
  live_dashboard.py         — live display
versions/
  v10_1405-06-23/           — v10 export (kb_export.json, manifest.json, README.md, V10_REPORT.md)
  v11_1405-06-23/           — v11 export (this version)
```

## What Was Fixed
| Issue | File | Change |
|-------|------|--------|
| Overlap prefix in QA rows | `semantic.py:426` | Returns `chunks + parent_chunks` only for body chunks, not row-wise atomic chunks |
| Keywords duplicated into content | `semantic.py:485-486,494` | `"keyword"`/`"keywords"` added to `seen` set; stripped from `_format_qa_content` |
| Dense context prefix (Title/Heading/Type) | `dense.py:124,257` | `use_context` default `True→False`; fingerprint includes flag for auto-rebuild |

## Next Version
- Future iterations should address `ChequeQuestions` keyword-stuffing and near-duplicate twin chunks
- Consider weighted fusion or query rewriting for reranker losses (B=13 rows)