# Research: Efficient Chunking & Ingestion for Varied Tabular KB

**Date:** 2026-09-03
**KB:** `kb-source/1405-05-31` — 34 docs / 2074 chunks (v7) + zip-stage
**Goal:** Optimal retrieval for heterogeneous Excel formats (QA, glossary, staff bios, loan catalog, timeline)

---

## 1. Inventory — What Formats Actually Exist

Inspected via `/transparency` (live parse: `XlsxParser` + `_normalize_col` `overlap ≥60%`):

| # | Example File | Sheets / Rows | Headers (original) | Detected Schema | Current Chunks | Type Inferred |
|---|---|---|---|---|---|---|
| **A** | `واژگان معادل.xlsx` (47 KB) | 1 × 49 | `رتبه` , `گرید` | **generic / body** (0/15,0/5) | 7 (body, ~7 rows/chunk) | **Glossary / synonym map** — 2-col narrow, short values (1-3 tokens), e.g. `رتبه چک → امتیاز چک` |
| **B** | `PublicQuestions.xlsx(done).xlsx` (113 KB) | 1 × 60 | `Question, BriefAnswer, Answer, Keyword` | **crm_qa** (4/5) | 53 (qa_pair, 1 row=1 chunk) | **Keyword QA** — 4-col, long Answer (200-500 tokens), Keyword list `،`-separated |
| **C** | `ExecutiveManagement.xlsx` (16 KB) | 1 × 2 | `نام و نام خانوادگی, سمت, بخش, سوابق تحصیلی, سوابق شغلی, تخصص ها و مهارت های کلیدی, نکات مهم` | generic / body | 9 (body, split long row) | **Staff / entity profile** — 7-col wide, 2 rows, each cell is *multi-paragraph* bio (800-1500 tokens/row) |
| **D** | `لیست و عناوین وام ها.xlsx` (75 KB, also 136-row copy in `سایر/`) | 1 × 598 | `نام وام, نوع وام, نرخ سود وام, سقف وام, حداکثر زمان بازپرداخت, مبلغ قسط, نوع ضمانت, نیاز به سپرده, مجموع سود وام, مجموع وام و سود, وضعیت, حمایت‌شده, لینک` | generic / body | **597** (~1-2 rows/chunk, 232 tok) | **Loan catalog / wide tabular** — 13-col, 598 rows, mixed types (percent, money, link), short per-cell (2-5 tokens) |
| **E** | `ICS_Intro.xlsx` / `BoardOfDirectors.xlsx` / `DepartmentManagers.xlsx` (not yet fetched, inferred from names) | 1 × ~10-20 | `نام, سمت, تاریخ, رویداد` etc. | likely generic | ~? | **Timeline / org chart** — date-event or name-role hierarchies |
| **F** | `Company_CRM_Questions(done).xlsx`, `Individual_CRM_Questions(done).xlsx`, `ReasonCode_*` (3 files) | 1 × ~200-300 | `question, answer, keyword, model` etc. | crm_qa / reason_codes | ~200 each | **CRM QA + Reason Codes** — same as B but with `model` column |

**Key finding:** Only **2 of 6** real types map to existing schemas (`crm_qa`, `reason_codes`). **Glossary (A), Staff (C), Loan catalog (D), Timeline (E)** all collapse to `generic / body` → semantic splitter (`semantic.py:433`) treats them identically (pipe-row → token-limited grouping), which is suboptimal.

---

## 2. Current Pipeline — Why It Fails on Varied Tables

```mermaid
flowchart LR
  XLSX --> Parse[XlsxParser: headers=first row, _format_cell, _detect_schema 60%]
  Parse --> Text[_sheet_to_text: "header: value | header: value"]
  Text --> Prep[PersianPreprocessor: ZWNJ, digits, spaces]
  Prep --> Chunk{doc_type?}
  Chunk -->|qa_pair/reason_detail| RowChunk[1 row=1 chunk, Persian labels]
  Chunk -->|else body| StructChunk[article/section/heading/pipe-row splitter, max 512 tok, overlap 50]
```

**Problems per type:**

* **A Glossary (2-col, 49 rows):** Body splitter groups 7 rows/chunk → query `رتبه چک چیست` must retrieve a chunk containing 7 mappings, dilution. Better: 1 row=1 chunk + synonym index.
* **C Staff (7-col, 2 long bios):** Body splitter *splits a single person's bio* across 4-5 chunks (cut in middle of `سوابق شغلی` paragraph) → retrieval may return half-bio without name. Better: **per-entity chunk** (1 row=1 parent) + **field-level sub-chunks** (e.g., `سوابق تحصیلی` separate).
* **D Loan catalog (13-col, 598 rows):** Body groups 1-2 rows/chunk → 597 chunks, but each chunk is `"نام وام: X | نوع وام: Y | ... | لینک: URL"` with no column boosting. Query `وام ازدواج 4 درصد` must match across 13 columns equally — should boost `نام وام` + `نرخ سود` differently.
* **E Timeline:** Date ordering lost; body splitter ignores `تاریخ` column as sort key — should chunk chronologically and add temporal embeddings.

**Normalization note (fix applied 2026-09-03):** `_normalize_col` (`re.sub(r"[\s_\-]+","", lower)`) is **only for header matching** — cell values keep spaces. Persian `ZWNJ` fix in `persian.py:88` was over-inserting (`ر‌تبه`); fixed to word-end only (`(?![\u0600-\u06FF\u200c])`).

---

## 3. Research — Table-Aware Chunking Strategies (Literature + Practice)

| Strategy | When to Use | How | Retrieval Benefit | Cost |
|---|---|---|---|---|
| **Row-wise atomic (1 row=1 chunk)** | High-cardinality short rows (glossary A, loan D, QA B) | `header: value` per row, Persian labels, no split. Parent = sheet. | Exact row retrieval, no dilution. Works with current `qa_pair` path. | Many chunks (598 loan rows → 598 chunks) — okay: 384-dim embeddings cheap, BM25 char-3grams handle typo. |
| **Field-aware hierarchical** | Wide rows with long cells (staff C) | Parent = whole row (entity), Children = per-field chunks (`سوابق تحصیلی` alone) + parent aggregated `1536 tok`. Use `parent_scope=sheet` already supports this. | Query `سوابق شغلی کمیل شاعری` hits field chunk directly, not truncated half-bio. | Need field length heuristic: if cell >300 tok, split that field alone via sentence splitter, not whole row. |
| **Column-boosted / metadata-enriched** | Wide catalogs (D) with typed columns | Store `doc_metadata.fields` already (`semantic.py:284`) but not used at retrieval. Add `field_weights` at query time: boost `نام وام` ×3, `نوع وام` ×2, `لینک` ×0 (exclude). Alternative: generate synthetic `question` per row: `"وام {نام} چیست؟"` | Improves `keyword_only` 65%→? | Requires query analysis to detect intent (loan name vs rate). |
| **Timeline chronological** | Date-event tables (E) | Parse `تاریخ` column (Persian digits → ASCII via `persian.py:119`), sort rows, chunk as `date: event | ...` with temporal proximity in embedding (prepend date). | Enables `before/after` queries. | Need Persian date parser (`1403/02/15` etc.). |
| **Synonym / glossary expansion** | Glossary A | Already have `واژگان معادل.xlsx` — use as **query expansion** at retrieval (like `query_expansion.py` 74 entries), not as retrievable chunks. Could keep both but weight glossary chunks low. | Handles `رتبه` vs `گرید` synonym. | Risk of double-counting — better to use as expansion, not chunk. |
| **Table serialization variants** | All tables | Research shows `Markdown table` vs `header: value |` vs `JSON` vs `HTML` — for LLM, `header: value` is best for RAG (used now). Keep, but add `field_fa` Persian labels for model. | Consistent with current. | — |
| **Hybrid table+text** | Mixed `docx` bios | Keep as is — body splitter already handles paragraphs. | — | — |

**Retrieval layer (after chunking):**

* **BM25** (char-3grams, keyword×3) excels on glossary/loan names (exact match). Keep.
* **Dense** (MiniLM-L12) excels on QA semantic. Keep.
* **RRF k=60** already fuses — good for varied.
* **Reranker** (`mmarco-mMiniLMv2`) is generic; for tables, **field-weighted BM25** may beat reranker (which over-demotes golden chunks per v7 73.3% misses). Consider **table-specific reranker** or `RERANKER_TOP_K 50→100` already noted.

---

## 4. Proposal — Optimal Ingestion Pipeline (Type-Aware)

### 4.1 New Schema Detection (extend `xlsx_parser.py:92`)

```python
# Add to SCHEMA_DEFS / _detect_schema:
GLOSSARY_COLS = {"رتبه","گرید", "واژه","معادل"}  # 2-col synonym
STAFF_COLS = {"نام و نام خانوادگی","سمت","سوابق تحصیلی","سوابق شغلی"}  # 7-col long
LOAN_CATALOG_COLS = {"نام وام","نوع وام","نرخ سود وام","سقف وام","لینک"} # 13-col
TIMELINE_COLS = {"تاریخ","رویداد","شرح"}  # date-centric
# Threshold: overlap ≥60% OR exact 2-col glossary (رتبه+گرید)
```

Result `doc_type` mapping: `glossary → glossary`, `staff → staff_profile`, `loan_catalog → loan_row`, `timeline → timeline_event`, keep `qa_pair`, `reason_detail`.

### 4.2 Chunker per Type (extend `chunker/semantic.py:126`)

| doc_type | Strategy | `max_tokens` | Parent |
|---|---|---|---|
| `glossary` | 1 row=1 chunk, `رتبه: X | گرید: Y` + synonym expansion at query, no parent | 128 | none |
| `qa_pair` | 1 row=1 chunk (existing) | 512 | sheet |
| `loan_row` | 1 row=1 chunk, all 13 cols `header: value`, boost metadata `field_weights` | 256 | sheet (all loans) |
| `staff_profile` | 1 row=1 parent (entity), 2-4 field children if cell >300 tok (split field by sentences) | 512 parent/256 child | sheet |
| `timeline_event` | 1 row=1 chunk sorted by `تاریخ` (normalized), prepend date to embedding | 256 | sheet |
| `body` (fallback) | current structural splitter | 512 | sheet |

Implementation: Add `_chunk_glossary`, `_chunk_loan_rows`, `_chunk_staff`, `_chunk_timeline` similar to `_chunk_excel_rows`, reuse `FIELD_FA` + `doc_metadata.fields`.

### 4.3 Ingestion API Change (pipeline)

* `zip_browser` (`zip_browser.py`) already supports selective ingest — extend to show **detected type per file** in preview (use `_detect_schema_debug` from `transparency.py` + new types) so user sees `loan_catalog` vs `glossary` before ingest.
* Add `field_weights` to `Chunk.doc_metadata` for loan catalog (e.g. `{"نام وام":3, "نرخ سود وام":2, "لینک":0}`) — retrieval can read it.

### 4.4 Evaluation Plan (no merge until benchmark)

Use existing IVA 15 + `TestQuestions_IVA` + synthetic `regenerate` per type:

* **RetrievalMetrics:** Recall@1/5 per type (glossary, staff, loan, QA, timeline) vs baseline `body` uniform.
* **Answer groundedness:** RAGAS faithfulness per type.
* **Latency:** p95 per type (loan 598 rows will dominate).

Gate: `Recall@5 ↑` per type without `hallucination ↑`.

---

## 5. Immediate Next Steps (Prototype)

1. **Patch `xlsx_parser.py:87-116`** add `GLOSSARY/STAFF/LOAN/TIMELINE` defs + `_normalize_col` already correct.
2. **Patch `chunker/semantic.py`** add 4 new `_chunk_*` methods, wire in `chunk()` switch.
3. **Update `transparency.py:30`** `SCHEMA_DEFS` to show new types + `zip_preview.html` to display `detected_type` badge.
4. **Re-ingest** one file per type (`واژگان معادل`, `ExecutiveManagement`, `لیست و عناوین وام ها`) via `zip_browser` selective and compare `transparency_detail` chunks before/after.
5. **Benchmark** `run_iva_eval.py` + type-specific synthetic (ask `وام ازدواج بانک ملت چقدر سود دارد` should hit loan row directly).

---

## 6. Recommendation

**Do not use uniform body chunking for varied tabular KB.** Adopt **type-aware row-wise + field-hierarchical** pipeline above. It reuses existing `1 row=1 chunk` infrastructure (proven for QA) and extends it with minimal code (3 new schemas, 4 chunkers). Keep BM25+dense+RRF+reranker unchanged; add field-weighted retrieval as second-phase experiment.

**Risk:** Staff bio splitting needs Persian sentence splitter (`hazm`); fallback to `semantic.py:496` `_split_emergency` works.

**Owner:** Retrieval / Ingestion. **Est.:** 1d parser+chunker, 1d transparency+zip UI, 1d eval.

---

*References:* `kb_manager/parsers/xlsx_parser.py:87`, `kb_manager/chunker/semantic.py:126,183`, `kb_manager/preprocessor/persian.py:88`, `kb_manager/web/routes/transparency.py:30`, `docs/REMEDIATION_PLAN.md`
