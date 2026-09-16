# v10 — KB 1405-06-23 ingest report (2026-09-16)

Source: `1405-06-23.zip` (41 files) staged with cleanup:
- removed byte-identical `ChequeQuestions(Advanced).xlsx` (sha `5f360127`)
- renamed `IndividualQuestions(V.3)` → `IndividualQuestions_V3.xlsx`,
  `ReasonCodeIndividual(V.3)` → `ReasonCodeIndividual_V3.xlsx`
- 5 PNGs auto-skipped by orchestrator (not parseable)

DB: `data/kb_1405_06_23.db` (gitignored, local only) — 29.4s full rebuild.

## Counts

- Documents: **35** created, 0 failed, 0 skipped
- Chunks: **2282** (0 skipped-incomplete)

| chunk_type | count | source |
|---|---|---|
| qa_pair (+9 parents) | 702 | 8 QA files (Company 330, Individual 172*, Cheque 118, categorized, Dispute, Etebarito, Public, quiz) |
| reason_detail (+4 parents) | 362 | 23 + 11 + 97 + 231 reason-code rows |
| kv_pair (+7 parents) | 854 | 10 generic/KV sheets incl. 598-row loan glossary |
| article (+7 parents) | 109 | 5 articles sheets |
| body | 188 | ICS matrix, Points_to_Note overflow, PDFs, DOCX |
| single_col_list (+1 parent) | 20 | Mobile_Banks (previously rejected) |
| staff_profile (+3 parents) | 16 | Board/Directors/Managers |

\* IndividualCRMQuestions shows 172 stored chunks vs 383 rows — remainder flagged
by the quality gate (see below); under review, not a silent drop.

## Schema / chunking changes in this version

1. Skip rule: rows drop only when ALL selected columns are empty (was: QA rows
   missing question/answer dropped).
2. `articles` sheets are row-wise (`article` type) instead of structural body.
3. New `kv_pair` type: `Key: | Value:` paragraph + `Summary:` line for long
   values + `metadata.raw_json` + col0 keywords; auto-detected on narrow
   (2–3 col) generic sheets.
4. Single-column sheets parse as `single_col_list` (was: whole file rejected).
5. Persian chunk labels extended (`متن دلیل`, `پیشنهاد بهبود`, `دسته‌بندی`,
   `نام سند`, `عنوان`, `محتوا`, …); legacy/dup columns dropped at parse
   (`Answer جدید/قدیم`, `Column 1`, `پاسخ سابق`, `بهبود…`, `کامنتها`);
   `نظر` review sheets excluded from ingest.
6. Extensionless/misnamed workbooks parse via ZIP-magic sniffing.
7. Retrieval: `POST /search/enhance` (offline Persian query enhancer);
   beam multi-query stays behind `KB_MULTIQUERY_ENABLED`.

## Quality-gate notes (warnings, non-blocking)

`N/M chunks failed validation` lines during ingest are quality-score warnings;
documents still stored (`documents_failed: 0`). Outlier:
`ReasonCodeIndividual_V3.xlsx` 21/148 — needs a look (likely long-cell
truncation). Follow-up: per-file recall benchmark (`run_benchmark.py` on a
fresh question set for this corpus).

## Chunking transparency

Open `/ingestion` → load the zip → per-file tree → file editor → Chunking View
shows every row mapped to its chunk with colors + header→label mappings.
Historical record: `GET /ingestion/kbs`.
