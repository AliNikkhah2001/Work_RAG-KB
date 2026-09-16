# KV / Tree / Variable Chunking Plan — KB 1405-06-23 (41 files)

Audit tool: `kb_manager/parsers/{xlsx_parser,docx_parser,pdf_parser}.py` (existing code).
Audit artifacts: `kbaudit.json` (per-file sheets/headers/rows/vars) + scripts in temp.
Server: chunking view now renders tabular AND flowing-text sheets + extensionless workbooks.

## 0. Bugs fixed (this round)

1. **Chunking view dead-end on generic sheets** — `GET .../file/chunking` returned
   `is_tabular=false` with zero rows mapped ("Not a tabular type"). Now non-tabular
   sheets simulate `_chunk_structural` and map every row to its chunk
   (`chunk_type: "body (chunk k/n)"`). Verified: `DataUsers.xlsx` 114/114 rows
   into 26 structural chunks, 0 skipped. Frontend renders the table + notice.
2. **Extensionless workbooks invisible** — `IndividualQuestions(V.3)` and
   `ReasonCodeIndividual(V.3)` are valid XLSX (magic `PK\x03\x04`) with a bad
   name. Preview + chunking endpoints now sniff ZIP magic and parse them.
   Verified: `(V.3)` file = `qa_pair`, 383/383 rows mapped.
3. **`articles` sheets not row-wise** (previous round) — `article` added to
   row-wise types in `SemanticChunker.chunk()`; viz mirrors orchestrator mapping.

## 1. Format decision: paragraph + keywords (NOT markdown table, NOT JSON)

- KEEP the current `Header: value | Header: value` single-line paragraph per row.
  Why: BM25 tokenizes words (pipes are neutral separators); the dense MiniLM
  model was trained on natural text, not JSON braces or markdown pipes; our
  reranker scores natural sentences best.
- Do NOT emit markdown tables (`| a | b |` header + separator rows): the
  separator row (`|---|---|`) injects junk tokens, and multi-line cells break
  the table shape.
- Do NOT emit raw JSON: `{`, `"`, `:` pollute BM25 term stats and waste dense
  tokens; key order is unstable for hashing/dedup.
- Your instinct is right: **paragraph + keywords**. Concretely per chunk:
  `content` = paragraph line(s); `keywords` = dedicated array column
  (already exists on Chunk). For KV rows without Q/A, auto-generate
  `Summary:` first line + keyword list from both columns.

## 2. Per-file verdicts (31 xlsx + 1 docx + 2 pdf + 5 png + 2 misnamed)

### A. Row-wise QA — keep as-is (1 row = 1 chunk)
| File | Rows | Note |
|---|---|---|
| Company_CRM_Questions.xlsx | 332 | has junk `Column 1` empty col + `Answer ????` dup col — drop both via target columns |
| IndividualCRMQuestions.xlsx | 383 | has trailing `???????` col — review, likely drop |
| ChequeQuestions.xlsx / ChequeQuestions(Advanced).xlsx | 146 each | **BYTE-IDENTICAL (sha 5f360127)** — ingest one, exclude the other |
| DisputeQuestions.xlsx | 8 | keep |
| EtebaritoProblems.xlsx | 11 | keep |
| PublicQuestions.xlsx.xlsx | 61 | keep (fix double extension on rename) |
| Individual_CRM_Questions_categorized.xlsx | 333 | keep; 3 extra cols are useful filters |
| (trivia quiz, 2 rows, crm_qa) | 2 | keep if wanted; tiny |
| IndividualQuestions(V.3) [misnamed] | 383 | **rename to .xlsx**, ingest as qa_pair |

### B. Row-wise reason codes — keep as-is
23 + 11 + 97 + 231 rows. `Cheque_ReasonCode_for_Chatbot.xlsx` has 145
indented col-0 rows → hierarchy signal (see §5). `ReasonCodeIndividual(V.3)`
[misnamed] → rename to .xlsx, ingest.

### C. Row-wise articles (now fixed) — keep
BusinessCreditReport (11), ChequeReport (11), IndividualCreditReport pair
(17+18 sheets each), `16 ...` (17). **Verify the (1)-suffixed pair is not a
duplicate** (compare content hashes before ingesting both).

### D. Row-wise staff — keep
BoardOfDirectors (3), DepartmentManagers (11), ExecutiveManagement (2).

### E. Key/value + generic — NEW `kv_table` strategy (see §4)
| File | Shape | Recommendation |
|---|---|---|
| DataProviders(done).xlsx | 66×2 KV | kv_table, keywords from col0 |
| DataUsers.xlsx | 114×2 (Name\|Type) | kv_table; Type values become keyword facets |
| neobanks.xlsx | 21×2 (id\|name) | kv_table + **bank registry source** |
| Mobile_Banks.xlsx | 20×1 single col | parser currently REJECTS (<2 headers) → extend parser: single-col = list, 1 row = 1 chunk; also bank registry source |
| Glossary 598×2 (Persian) | 598 rows | kv_table, biggest recall win; keywords from Persian term col |
| Points_to_Note.xlsx | 13 rows, JSON-ish alert cells | kv_table; cells contain JSON blobs — render as `field: flattened values`, drop braces |
| ImportantLinks.xlsx | 15×3 (title\|url\|desc) | kv_table; URL kept verbatim in chunk (retrieval shows link) |
| ICS_Intro.xlsx | 26×4, col0 empty | drop empty col0 via target columns, then kv-ish flow |
| StockHolders(done).xlsx | 27×3 | kv_table |
| ICS_Business_Model_Comparison.xlsx | 3×5 matrix | transpose-aware: 1 row = 1 provider chunk (ICS vs ICBS vs ...), headers as aspects |

### F. Non-row content
- 2 PDFs (3 + 17 pages): structural chunking as today; verify 17-page PDF headings.
- 1 DOCX (15 sections): keep.
- 5 PNG charts: **orchestrator already skips them** (not xlsx/pdf/docx) — they only
  clutter the tree. Mark excluded-by-default in the suite (no KB dilution today).

### G. Empty / irrelevant
- **No fully-empty sheets found** (empty_rows=0 everywhere).
- Empty columns to drop via target columns: `Column 1` (Company_CRM),
  `ICS_Intro` col0, `IndividualCRMQuestions` trailing col.
- Mobile_Banks: unparseable today → parser extension (single-col list).
- `(V.3)` files: rename, do not keep both spellings.

## 3. Dilution risks (exclude or dedup)
1. `ChequeQuestions(Advanced).xlsx` — exact duplicate, exclude one.
2. `IndividualCreditReport(1).xlsx` vs `IndividualCreditReport.xlsx` — verify, likely dup.
3. 5 PNGs — auto-skipped by orchestrator; hide in suite tree by default.
4. `Column 1` / empty cols — drop via target columns (already supported).
5. Overlapping QA corpora (Company 332 vs categorized 333 vs Individual 383):
   keep all (dedup_questions handles overlaps) but benchmark per-file recall.

## 4. New `kv_table` chunk strategy (implementation)
- Detect: `len(headers)==2` and schema None, OR explicit schema_override=`kv_table`.
- Chunk = one row: `Key: <col0> | Value: <col1>` (+ optional `Summary:` line for
  long values: first sentence of value).
- Keywords: col0 tokens + value nouns → `keywords` array (BM25 `bm25_kw` leg).
- Tree KV (§5): if col0 indentation/hierarchy detected, emit `parent_key` chain
  `bank > program` and a parent chunk per top-level key aggregating children.
- Columns: add `Chunk.chunk_type="kv_pair"` (+ `kv_pair_parent`); parents excluded
  from BM25/dense by the existing `%_parent` rule automatically.

## 5. Variable templates (`<bank_name>`) — preprocessing expansion
- Scan found **no `<...>` placeholders in the current 41 files** — treat as a
  forward mechanism; add a `variables` detector column in the suite
  (`<name>`, `{name}`, `[NAME]` regex already in audit script).
- Registry: `data/variable_registries/bank_name.json` seeded from neobanks (21) +
  Mobile_Banks (20) + DataUsers bank-type rows; UI editable.
- Expansion (preprocessing, one-to-many): a template row containing
  `<bank_name>` fans out to N chunks, one per registry value, with
  `metadata.expanded_from` + `metadata.variable_values`. Cap: max 50 values per
  var; whitelisted vars only (`bank_name`, ...); expansion happens in
  `SessionFilteredXlsxParser` (session-scoped, reproducible per session.json).
- Tree case (bank → many programs): expansion composes with §4 parent chains —
  parent per (bank, program-group), children per template×value. Guard total
  chunk budget per file (default 5k) with a warning in the chunking view.

## 6. Keywords + summary for non-QA parts
- `Summary:` auto-line: first sentence (or first 25 words) of the longest value
  cell, prepended to chunk content for articles/KV rows.
- Keywords: union of (a) explicit keyword cols, (b) registry facets
  (bank/program names), (c) top TF terms of the row. Stored in `keywords`,
  never inline-duplicated into content.

## 7. Query rewriting + prompt-enhancing tools (retrieval layer)
- `kb_manager/query_reform.py` already has `MultiQueryGenerator` (beam5, mocked
  tests) — wire it behind `KB_MULTIQUERY_ENABLED` (default off): rewrite →
  beam5 → RRF over beams → existing pipeline. A/B via `/benchmarks`.
- Prompt-enhancer: new `kb_manager/query_enhance.py` — rule-based Persian
  expander (colloquial→formal map in `query_expansion.py`, 74 entries) +
  optional LLM rewrite (strict JSON, `KB_ALLOW_MOCK` gate, HyDE-style).
  Expose `POST /search/enhance {query} → {enhanced, beams}` for the UI + eval
  harness comparing hit@5 with/without.
- If you meant opencode-agent tooling instead, say so and I will add a local
  skill under `.opencode/` for prompt rewriting.

## 8. Work order
1. `kv_table` chunk type + parser single-col lists + `(V.3)` renames.
2. Variable registry + expansion in session parser + suite `variables` column.
3. Tree-KV parent chains + chunking-view support (colors per parent group).
4. Keywords/summary enrichment for non-QA rows.
5. Multi-query + enhancer behind flags, benchmarked before default-on.
