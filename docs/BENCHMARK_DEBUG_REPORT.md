# Benchmark Debug — Case-by-Case Analysis

**Generated:** 2026-09-07 13:45 UTC  
**DB:** `kb_9_7_2026.db` 21 docs / 1084 chunks (type-aware glossary/staff/loan) + `kb_1405.db` 32 docs / 2133 chunks (old)  
**Models:** `paraphrase-multilingual-MiniLM-L12-v2` (384), `mmarco-mMiniLMv2-L12-H384` reranker, `RERANKER_TOP_K=100` (was 50), `BM25 k1=1.5 b=0.75` + `char 3-grams`, `RRF k=60`

## Pass/Fail Criteria

| Benchmark | Pass Criteria | Fail Criteria | Current |
|---|---|---|---|
| **Massive QA** (573 verbatim, `test_qa_file_verbatim_recall`) | `hit@5` = expected chunk ID in `final_results` top-5 (verbatim question → same row chunk). **Pass if hit@5 ≥ 95%** (allows 5% due to near-duplicates) and `MRR` and `latency` not regressed. | `hit@5 < 95%` or any file < 90% | **12/573 (2.1%) — FAIL** (was 10/573, after `persian.py:96` fix still low) |
| **IVA 15** (`run_iva_eval.py`) | `doc_hit` = golden doc chunk in top-5. **Pass if 13/15 (86.7%)** (was 11/15 73.3% v7). Also `ans_hit ≥ 5/15` (≥70% answer tokens). | `doc_hit < 13/15` or `MRR < 0.6` | **11/15 (73.3%) — FAIL** (needs 2 more) |
| **Latency** | `p50 < 25s` CPU, `p95 < 35s` for 573 queries (with `RERANKER_TOP_K 100` + `HNSW 13ms` on GPU `18.4s` baseline). | `p95 > 40s` | `avg 31.9s` on new KB (slightly high due to 100) |
| **Ground truth distance:** For each miss, check `BM25 rank`, `Dense rank`, `merged rank (RRF)`, `final rank (rerank)`, and scores (`bm25_score`, `dense_score`, `hybrid_score`, `rerank_score`). GT should be within top-50 merged to be reranked; if `BM25 -1` and `Dense -1`, it's not retrieved at all → vocabulary gap. |

## Massive QA — 20 Failed Samples (first 3 detailed)

**Overall:** `total 573, passed 12, failed 561? Wait actual `massive_results.json` shows `passed 10, failed 20` in UI but script shows `12/573` — discrepancy due to intermediate save every 10 (hit 0.017). After full 573, hit is 2.1% (very low) because expected is `qa_pair_parent` (264 chunks per doc) but retrieval returns `qa_pair` children.

### Failed #1: `سامانه رسیدگی به اعتراض...` (DisputeQuestions.xlsx, expected `1549feae` parent)
- **GT:** `qa_pair_parent` `Sheet: Sheet1` parent of 264 `qa_pair` children, content `سوال: فرایند رسیدگی... 12 روز کاری`
- **Retrieved top-5:** `24acadcc` (bm25 638, dense 0.50, hybrid 0.032, rerank 1.0), `bb175e54` (329, 0.42, 0.03, 0.15) — both are `DisputeQuestions` qa_pair children but **not the parent**.
- **Ranks:** `GT BM25 -1` (not in BM25 top-50), `Dense 6` (found via dense), `merged 9` (in top-50 via RRF), `final -1` (reranker demoted from 9 → >5). **Root cause:** BM25 miss due to `parent` vs `child` content mismatch — parent contains `سوال: فرایند...` plus 264 QAs concatenated, while query is specific `سامانه رسیدگی... وارد نیست`. Dense found it at 6, but BM25 + RRF + rerank pushed it out. **Fix:** Massive benchmark expected should be `qa_pair` child, not parent. `build_iva_dataset.py:59` already groups by `qa_pair_parent` for IVA, but for massive verbatim, expected should be the exact row's child chunk, not parent. Change `expected` to child chunk ID (where `q[:30] in content` for child, not parent).

### Failed #2: `چرا اطلاعات شخصی من در گزارش غلط است؟`
- **GT:** Same parent `1549feae`
- **Retrieved:** `bb175e54` (152, 0.52, 0.032, 0.99), `ea038cca` (143, 0.51, 0.031, 0.75) — both are `DisputeQuestions` but not parent.
- **Ranks:** `BM25 -1, Dense -1, merged -1, final -1` — **not in top-50 at all**. Vocabulary gap: query `اطلاعات شخصی` vs GT parent content `فرایند رسیدگی` — no overlap. **Root cause:** Expected parent is wrong for this question; the question belongs to a different row's child, but parent contains all 264 QAs, so `q[:30]` match is ambiguous and picks wrong parent. Need per-row expected, not parent.

### Failed #3: `چرا اطلاعات وام های من در گزارش غلط است؟`
- Same as #2 — parent mismatch, not in top-50.
- **Root cause:** Same as #1/2 — expected should be child, not parent. Also `clean.py:118` lstrip handles `؛` but not this query.

**General massive failure root:** `test_qa_massive.py:44` expects `qa_pair_parent` (264 children) for every question, but verbatim query should match its own `qa_pair` child (single row), not the aggregated parent. The test's `q[:30] in content` for parent will match any question's first 30 chars that appear anywhere in the 264 concatenated QAs, but picks the parent's ID, while retrieval correctly returns the child. **Fix:** Change `expected` to the specific row's child chunk ID, not parent. Also increase `RERANKER_TOP_K` already 100 helps but not for parent vs child.

## IVA 15 — 4 Misses Detailed

| # | Query (truncated) | GT doc | Expected answer preview | Retrieved top-5 (BM25/Dense/Hybrid/Rerank) | GT Ranks | Why Miss |
|---|---|---|---|---|---|
| 11 | `این دلیل کاهش امتیاز یعنی چی: ؛؛وام‌های ضمانت شده توسط فرد، در ماه‌های` | `Individual_CRM_Questions` (264 chunks) | `وقتی ضامن تسهیلات...` | Top-5: `322cc2e2` (bm25 282, dense 0.41, hybrid 0.029, rerank 0.014), `b8a2c850` (0,0.41,0.029,0.014) — all are `DisputeQuestions`/`Etebarito` not reason_codes | `BM25 -1, Dense 6→ merged 9 → final -1` (reranker demoted) | **Truncated `؛؛` + reason_code vocabulary** — query has leading `؛؛` (now stripped in `clean.py:118` but gold rebuilt before fix) and is a reason-code `وام‌های ضمانت شده` which is `reason_detail` not `qa_pair`, but search returns `qa_pair` children. Need `reason_detail` boost or `RERANKER_TOP_K 100` already, but still demoted. |
| 12 | `این دلیل کاهش امتیاز رو چطوری بهبود بدهم؟ «وام‌های ضمانت` | Same | `اگر وام‌های فعالی دارید که به عنوان ضامن...` | Similar to 11, `b8a2c850` (0,0.41), `322cc2e2` etc. | `BM25 -1, Dense -1, merged -1, final -1` | **Not in top-50 at all** — query is truncated + colloquial `چطوری→چگونه` not expanded, plus `وام‌های ضمانت` needs `query_expansion` `وام‌های ضمانت→وام ضمانت`. |
| 14 | `رتبه چه فرقی با امتیاز داره؟` | `Individual_CRM_Questions_categorized` (115 chunks) | `امتیاز اعتباری به چندین دسته تقسیم می شود...` | Top-5: `322cc2e2` (0.031), `642e252e` (0.030), `d7af2aae` (0.014) — GT was in `Individual_CRM_Questions_categorized` but retrieved are `Dispute`/`Public` | `BM25 0.0, Dense 0.45, merged -1, final 3` — actually **HIT at rank 3** in `debug_iva_detailed.py` but `iva_results.json` says miss — discrepancy due to `doc_hit` vs `chunk_id` (gold doc has 115 chunks, GT chunk is one of them, but retrieved top-5 includes GT's doc but different chunk ID? `debug_iva_detailed.py` shows HIT for Q14 at rank 3 after rebuild, but `iva_results.json` says miss — need to check `doc_hit` vs `chunk_id` logic: `iva_results.json` checks `cid in exp_ids` where `exp_ids` is all 115 chunks of the doc, so if any of those 115 is in top-5, it's hit. For Q14, `debug` shows GT at rank 3 (in final), so should be hit, but `iva_results` says miss — maybe gold rebuilt before `clean.py` fix, so `q` is `رتبه چه فرقی...` but GT doc is `Individual_CRM_Questions_categorized` which has `رتبه چه فرقی با امتیاز داره؟` as a question, but the query is slightly different (`رتبه چه فرقی با امتیاز داره؟` exact), but `iva_debug` shows HIT. Confusing — need to re-run `run_iva_eval.py` after latest fixes. |
| 15 | `جزئیات قراردادهای منفی در گزارش یعنی چی؟` | `IndividualCreditReport` (115 chunks) | `در صورتی که برخی قراردادها وضعیت نامطلوب داشته باشند... 1. مشکوک‌الوصول 2. معوق 3. سررسید گذشته` | Top-5: all are `fe732b17` `24aa82a5` etc. with exact expected text `در صورتی که برخی قراردادها...` but from different docs (`BusinessCreditReport`, `IndividualCreditReport`, etc.) — content identical across 5 docs, but expected doc is single, so `doc_hit` fails even though content matches. | `BM25 0, Dense 0, merged -1, final -1` for expected doc, but retrieved has same content from other doc. **Root cause:** Duplicate content across docs (5 docs share same `جزئیات قراردادهای منفی` section). `build_iva_dataset.py:59` duplicate handling now groups by content hash, but `iva_results` still checks `cid in exp_ids` only for one doc. Need to consider content hash, not doc ID. |

## Ground Truth Distance vs Selected

For each miss, GT was either:
- **Not in top-50 merged** (`BM25 -1, Dense -1`) → vocabulary gap, truncated query, or wrong doc_type (reason_code vs qa_pair).
- **In top-50 but reranker demoted** (`merged 9 → final -1`) → cross-encoder `mmarco` doesn't understand Persian reason-code `وام‌های ضمانت شده` as relevant, scores 0.01 vs top-1 1.0.

**Distances:** `BM25` is sparse TF-IDF with `char 3-grams` + `keyword×3`, `Dense` is cosine `MiniLM-L12` (384), `Hybrid` is RRF `1/(60+rank)`, `Rerank` is `cross-encoder` logit. For Q11, GT `BM25 -1` (score 0) but `Dense 6` (0.41) gave `Hybrid 0.029` rank 9, then reranker `0.014` pushed to >5.

## Pass/Fail Criteria (Updated for New KB)

- **Massive QA (573 verbatim, new KB 21 docs, 1084 chunks):** Pass if `hit@5 ≥ 95%` (≥544/573) and `MRR ≥ 0.9`. Currently `12/573 (2%)` is **FAIL** due to parent vs child expected mismatch, not retrieval — fix `test_qa_massive.py:44` to expect child, not parent.
- **IVA 15:** Pass if `doc_hit ≥ 13/15 (86.7%)` and `ans_hit ≥ 5/15` and `MRR ≥ 0.6`. Currently `11/15 (73.3%)` with `Q11,12,14,15` miss — need `RERANKER_TOP_K 100` already, plus `query_expansion` `وام‌های ضمانت` and `clean` `؛` fix and `build_iva_dataset` duplicate handling. After fixes, target `13/15`.
- **Latency:** `p50 < 25s` CPU, `p95 < 35s` for 573 queries (with `RERANKER_TOP_K 100` ~31s avg, as seen `IVA avg 31.9s`).

## Next Steps

1. Fix `test_qa_massive.py` expected to be child chunk, not parent (change `q[:30] in content` for parent to exact row match for child).
2. Re-run `rebuild_after_fix.bat` already did `build_iva_dataset.py` duplicate handling, but need to re-run `run_iva_eval.py` after `clean.py` fix for `؛`.
3. Re-run massive via `http://127.0.0.1:8000/benchmarks/massive` (now shows `62/573 10.8%` live, `Latest Result` updated every 10 via `benchmarks.py:640`).
