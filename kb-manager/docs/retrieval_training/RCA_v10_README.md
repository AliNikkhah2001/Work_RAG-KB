# Root-Cause Analysis — v10 `1405-06-23` retrieval failures

**Date:** 2026-09-19 · **KB:** `kb_1405_06_23.db` (35 docs, 2,282 chunks) · **Eval:** 714 gold-mapped QA rows
**Baseline:** Hit@5 **0.9244** (660/714) · Hit@1 0.7703 · MRR 0.8369 · A=46 B=6 C=2 D=660 · 0 errors
**Models:** dense `paraphrase-multilingual-MiniLM-L12-v2` · reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
**Pipeline:** BM25 + dense → RRF (k=60) → cross-encoder rerank (top-5)

This document analyses **all 54 failures** (46 A + 6 B + 2 C) sample-by-sample. Every failure was dumped with its full gold text, its actual retrieved text at each stage, and word-level overlap stats, then judged independently by 4 analysis passes (one per file/batch) on both **lexical** (words) and **semantic** (answerhood) axes.

**Artifacts**
| File | Contents |
|---|---|
| `artifacts/retrieval_training/retrieval_failures_v10_1405-06-23.jsonl` | 714 rows: per-stage top-10 (BM25/dense/merged) + top-5 (final) with scores, `reranker_model`, exact 1-indexed gold rank |
| `artifacts/retrieval_training/rca/group_{A_company,A_individual,A_public,BC}.md` | per-failure dump: gold + all stage leaderboards |
| `artifacts/retrieval_training/rca_judge/judge_*.md` | per-failure **full gold text vs full retrieved-winner text** + Jaccard stats |
| `artifacts/retrieval_training/title_removal_experiment.json` / `_ablation.json` | dense-recall A/B of the embedding-prefix change |
| `artifacts/retrieval_training/dump_failures_v10.py`, `build_judge_corpus.py`, `title_removal_*.py` | reproducible scripts |

---

## 1. Headline finding — the dense embedding prefix is the single biggest defect

Every QA chunk is embedded (`.npz` dense index) as a **contextual string**, not as its content (`kb_manager/dense.py:57-76`, `web/routes/search.py:299-323`):

```
Title: ChequeQuestions              ← source FILE NAME (xlsx stem)
Heading: Sheet: سوالات حقوقی        ← sheet name
Type: Q&A
Content: سوال: … / پاسخ کامل: … / کلیدواژه‌ها: …
```

The `Title:` line injects the **file name** into the vector. This is the source of the "cheque-document gravity" you noticed: all `ChequeQuestions`/`Cheque_ReasonCode` chunks share the token `ChequeQuestions`, pulling them together in the semantic space and away from their true content.

### Measured effect (all 714 gold-mapped queries, same model)

| Dense gold recall | @1 | @5 | @10 | @20 | @50 | @100 |
|---|---:|---:|---:|---:|---:|---:|
| **WITH** prefix (current: Title+Heading+Type+Content) | 0.2115 | 0.4132 | 0.5070 | 0.6289 | 0.7465 | **0.8221** |
| **CONTENT ONLY** (no prefix) | **0.5924** | **0.7927** | **0.8473** | **0.8894** | **0.9300** | **0.9622** |
| Δ | **+0.381** | +0.380 | +0.340 | +0.261 | +0.184 | **+0.140** |

### Component ablation — which prefix part hurts

| Variant | @1 | @5 | @10 | @100 |
|---|---:|---:|---:|---:|
| full (Title+Heading+Type+Content) | 0.2115 | 0.4132 | 0.5070 | 0.8221 |
| NO TITLE (Heading+Type+Content) | 0.2927 | 0.5308 | 0.6317 | 0.8908 |
| NO HEADING (Title+Type+Content) | 0.2507 | 0.4706 | 0.5700 | 0.8571 |
| NO TYPE (Title+Heading+Content) | 0.2815 | 0.5070 | 0.5980 | 0.8768 |
| **CONTENT ONLY** | **0.5434** | **0.7591** | **0.8165** | **0.9580** |

**Every prefix component hurts; removing all of them is best.** Title alone costs ~7 pts recall@100; the full prefix costs ~14 pts. Per-file, content-only recovers the worst documents:

| File | recall@100 (prefix → content-only) |
|---|---|
| PublicQuestions.xlsx | 0.545 → **0.891** (+0.345) |
| Individual_CRM_Questions_categorized | 0.700 → 0.900 (+0.200) |
| Company_CRM_Questions | 0.831 → 0.970 (+0.139) |
| IndividualCRMQuestions | 0.827 → 0.955 (+0.128) |
| DisputeQuestions | 0.875 → 1.000 (+0.125) |
| ChequeQuestions | 0.923 → 0.983 (+0.060) |

> **Direct answer to your question: yes — removing the file title (and the sheet-name heading) from the embedded text is a large, measured win.** The dense leg's recall@100 would rise from 0.82 → 0.96, which should also lift the fused and final Hit@5. Note the pgvector ingestion path already embeds **content-only** (`orchestrator.py:366-369`), so the two dense backends are inconsistent — the `.npz` search-time index is the one with the harmful prefix.

---

## 2. Failure taxonomy — where the gold was lost

| Stage where lost | n | Meaning |
|---|---:|---|
| True retriever miss (neither BM25 nor dense top-100) | 34 | gold never entered any pool |
| **Fusion dilution** (a leg found it, RRF dropped it out of merged top-100 / >20) | 18 | 12 mislabelled `A` (`merged=None`) + all 6 `B` |
| Reranker demotion (merged ≤ 20, final > 5) | 2 | cross-encoder reordered it out of top-5 |

**Label caveat:** `classify()` uses `merged is None` → A. But 12 of the 46 "A" rows were found by **one leg** (dense r15–88, bm25 r36–99) and only lost in RRF. True retriever misses = **34**; fusion losses = **18**.

---

## 3. LLM-judge verdict — mechanism across all 54

Each failure was judged on **lexical** (shared/absent words) and **semantic** (does the text state the rule/amount the question asks for?) axes.

| Mechanism | n | Share | What it means |
|---|---:|---:|---|
| **RERANK / LEXICAL-BIAS** | 27 | 50% | winner is topically adjacent (same `امتیاز/رتبه/چک` tokens) but answers a *neighbouring* question |
| **RETRIEVAL-MISS** | 8 | 15% | gold is a valid answer but a scope/paraphrase sibling out-ranked it |
| **GOLD-MALFORMED** | 6 | 11% | gold chunk has the question but **no answer body** (or an answer for a different question) |
| **GOLD-LABEL-WRONG** | 6 | 11% | the retrieved winner is actually a **better** answer than the labelled gold |
| **FUSION-BIAS** | 5 | 9% | single-leg exact-match gold loses to a dual-leg adjacent-topic chunk |
| **RERANK-BIAS** | 2 | 4% | cross-encoder promotes a chunk that literally echoes the query string over the true answer |

**≈ 1 in 4 "failures" is a label/data problem, not a ranking problem.** Two golds contain the question with *no* `پاسخ` at all (#44, #47), one gold answers a different question (#153/#261), and several winners answer at least as well as the labelled gold (#19, #310, #59, #292).

### Sub-findings

- **Lexical overlap is anti-correlated with answerhood.** `Company_CRM_Questions#319`: winner Jaccard **0.219** vs gold **0.100** — the winner embeds a near-identical sibling question (`آیا باید تمام صاحبان امضا موافقت کنند؟`) but never answers it and pivots to a cheque question. Maximal lexical score, zero answerhood.
- **Distractor-prefix is structural.** Gold chunks are multi-QA blobs where the target `سوال:` is preceded by the tail of the *previous* row's answer + its `کلیدواژه‌ها`. ~11/22 Individual+Public golds have a clearly unrelated leading fragment, which both length-penalises BM25 and mean-pools away the embedding — the target question can be **verbatim in the gold** yet the gold is `None` at every stage.
- **Cheque-domain gravity well.** ~21% of A-Company winners and 9/22 Individual+Public winners are cheque-domain chunks, often for installment/tax/company queries (`#7`, `#9`, `#16`, `#22`). Combined with §1, the `Title: ChequeQuestions` prefix and the large keyword-stuffed `Cheque_ReasonCode` chunks are the mechanism.
- **Direction / aspect blindness.** Mirror-image questions (`company→person` vs `person→company`; score-changed-rank-unchanged vs the inverse) are treated as identical by both the fusion and the reranker.
- **A storage bug surfaced:** one retrieved chunk is stored in **Arabic Presentation-Forms** codepoints (`ﺍﻋﺘﺒﺎﺭﵐ`), giving Jaccard 0.000 against a normalised Persian query — a normalisation defect independent of ranking.

---

## 4. Full diagnosis table (all 54 samples)

### A — Company_CRM_Questions (24)

| # | query_id | gold answers? | winner answers? | mechanism | key lexical evidence |
|---|---|---|---|---|---|
| 1 | Company_CRM_Questions#13 | yes | no | RERANK/LEXICAL-BIAS | q↔gold shares منابع/داده/مالیات; q↔winner J=0.000; winner stored in Arabic presentation forms |
| 2 | Company_CRM_Questions#17 | yes | yes | RETRIEVAL-MISS | حقوقی gold out-ranked by حقیقی near-duplicate (inverted polarity) |
| 3 | Company_CRM_Questions#44 | **no** | no | GOLD-MALFORMED | gold = question + `دسته‌بندی/مدل`, **no پاسخ** |
| 4 | Company_CRM_Questions#47 | **no** | yes | GOLD-MALFORMED | gold = question + `دسته‌بندی: چک`, **no پاسخ**; winner has the rule |
| 5 | Company_CRM_Questions#84 | yes | no | RERANK/LEXICAL-BIAS | gold `اقساط/قدیمی/۵ سال` absent; winner only `امتیاز/تسهیلات/شرکت` |
| 6 | Company_CRM_Questions#130 | yes | partial | RETRIEVAL-MISS | mirror `ضمانتنامه` FAQ ranked above |
| 7 | Company_CRM_Questions#142 | yes | no | RERANK-BIAS + **CHEQUE** | q↔winner J=0.000; winner = cheque weighting |
| 8 | Company_CRM_Questions#148 | yes | no | RERANK/LEXICAL-BIAS | gold `رایگان/هزینه` absent; winner consent/privacy |
| 9 | Company_CRM_Questions#153 | **no** | no | GOLD-MALFORMED + **CHEQUE** | gold answer is for a different question (non-credit accounts) |
| 10 | Company_CRM_Questions#155 | yes | no | RERANK/LEXICAL-BIAS | gold `مختلف/یکپارچه/مؤسسات` absent |
| 11 | Company_CRM_Questions#167 | yes | partial | RETRIEVAL-MISS | answer-bearing fragment is only the tail of a sibling chunk |
| 12 | Company_CRM_Questions#181 | yes | no | RERANK/LEXICAL-BIAS | winner = "اعتراض به اطلاعات، نه امتیاز" |
| 13 | Company_CRM_Questions#192 | yes | yes | RETRIEVAL-MISS + **CHEQUE** | حقیقی C1–C3 chunk beat حقوقی gold |
| 14 | Company_CRM_Questions#194 | yes | partial | RETRIEVAL-MISS | sibling "محل سکونت" out-ranked |
| 15 | Company_CRM_Questions#204 | yes ("بله") | no | RERANK/LEXICAL-BIAS | q↔gold J=0.484 yet winner is the mirror FAQ |
| 16 | Company_CRM_Questions#205 | yes | no | RERANK-BIAS + **CHEQUE** | winner reverses causal arrow (cheque→company) |
| 17 | Company_CRM_Questions#228 | yes | no | RERANK/LEXICAL-BIAS | gold `بفرستم/بقیه/لینک` absent |
| 18 | Company_CRM_Questions#274 | yes | no | RERANK/LEXICAL-BIAS | winner = guarantee-record FAQ |
| 19 | Company_CRM_Questions#310 | yes | yes | GOLD-LABEL-WRONG | winner = actual NGS2 decline-reason record |
| 20 | Company_CRM_Questions#316 | yes | partial | RERANK/LEXICAL-BIAS | gold `عالی/سال/اقساط` absent |
| 21 | Company_CRM_Questions#317 | yes | no | RERANK/LEXICAL-BIAS | winner = mobile-bill FAQ |
| 22 | Company_CRM_Questions#319 | yes | no | RERANK-BIAS + **CHEQUE** | q↔winner J=0.219 > gold 0.100 — lexical reward-hack |
| 23 | Company_CRM_Questions#330 | yes | yes | RETRIEVAL-MISS | paraphrase sibling ranked above |
| 24 | Company_CRM_Questions#331 | yes | yes | RETRIEVAL-MISS | same paraphrase sibling |

### A — IndividualCRMQuestions (12) — dense almost never returns this doc

| # | query_id | gold answers? | winner answers? | mechanism | key lexical evidence |
|---|---|---|---|---|---|
| 25 | IndividualCRMQuestions#59 | yes | yes | GOLD-LABEL-WRONG | gold shares 13 query words; winner adds E3/250 |
| 26 | IndividualCRMQuestions#292 | yes | no | RERANK/LEXICAL-BIAS | gold `تامین‌کننده/BNPL/لندتک` absent; winner `وام/بانک/اپلیکیشن` |
| 27 | IndividualCRMQuestions#315 | yes | no | RERANK/LEXICAL-BIAS | gold `جایگزین/محاسبه` absent |
| 28 | IndividualCRMQuestions#340 | yes | no | RERANK/LEXICAL-BIAS | gold `فقدان موجودی/مسدودی` absent; winner = bounce *count* |
| 29 | IndividualCRMQuestions#347 | yes | partial | RETRIEVAL-MISS + **CHEQUE** | q↔gold 10 shared vs winner 1 |
| 30 | IndividualCRMQuestions#352 | yes | no | RERANK/LEXICAL-BIAS | colloquial `بدهیمو/بعدش` absent |
| 31 | IndividualCRMQuestions#353 | yes | no | RERANK/LEXICAL-BIAS | typo `نىارم` absent |
| 32 | IndividualCRMQuestions#361 | yes | no | RERANK/LEXICAL-BIAS | gold 12 shared incl. dates `1404/1405`; winner has none |
| 33 | IndividualCRMQuestions#365 | yes | no | RERANK/LEXICAL-BIAS | gold 14 shared (`صادرات/تیر/پاس`) ; winner only `چک` |
| 34 | IndividualCRMQuestions#366 | yes | no | RERANK/LEXICAL-BIAS | winner J=0.000; raw WoE/Fico dump |
| 35 | IndividualCRMQuestions#375 | yes | wrong domain | RERANK/LEXICAL-BIAS | winner = حقوقی version with opposite conclusion |
| 36 | IndividualCRMQuestions#376 | **no** | no | GOLD-MALFORMED | gold = question + `دسته‌بندی: چک`, no پاسخ |

### A — PublicQuestions (10) — 9/10 golds are multi-QA blobs

| # | query_id | gold answers? | winner answers? | mechanism | key lexical evidence |
|---|---|---|---|---|---|
| 37 | PublicQuestions.xlsx#3 | yes | no | RERANK/LEXICAL-BIAS | gold 10 shared; winner = cheque 60/40 weights |
| 38 | PublicQuestions.xlsx#16 | yes | no | RERANK/LEXICAL-BIAS | gold J=0.281/18 shared; winner J=0.10 |
| 39 | PublicQuestions.xlsx#26 | vague | yes | GOLD-LABEL-WRONG | winner gives actionable `ics24.ir/support` fix |
| 40 | PublicQuestions.xlsx#38 | yes | yes | GOLD-LABEL-WRONG | winner ≈ query verbatim |
| 41 | PublicQuestions.xlsx#39 | yes | yes | GOLD-LABEL-WRONG | winner ≈ query verbatim (کسب‌وکار) |
| 42 | PublicQuestions.xlsx#45 | yes | no | RERANK/LEXICAL-BIAS + **CHEQUE** | winner = cheque/کسری |
| 43 | PublicQuestions.xlsx#48 | yes | no | RERANK/LEXICAL-BIAS | gold `هوش مصنوعی/شورای تامین` absent |
| 44 | PublicQuestions.xlsx#55 | **no** | no | GOLD-MALFORMED | winner J=0.600 (query verbatim) but answerless |
| 45 | PublicQuestions.xlsx#58 | yes | no | RERANK/LEXICAL-BIAS | polysemy trap `حقوقی` = legal vs rights |
| 46 | ChequeQuestions#116 | **no** | no | GOLD-MALFORMED | winner J=0.909 (near-exact query), no پاسخ |

### B — fusion (6) and C — reranker (2)

| # | query_id | type | gold answers? | winner answers? | mechanism | evidence |
|---|---|---|---|---|---|---|
| 47 | Company_CRM_Questions#141 | B | yes | no | FUSION-BIAS | gold single-leg dense r15; winner = headcount QA, dual-leg |
| 48 | Company_CRM_Questions#151 | C | yes | partial | RERANK-BIAS | gold merged 14 → final 20; winner = holiday-update mirror, no duration |
| 49 | Company_CRM_Questions#202 | B | yes | no | FUSION-BIAS | gold dense r80, merged 94; winner = tax-debt QA, dual-leg |
| 50 | PublicQuestions.xlsx#24 | B | yes | inverse | FUSION-BIAS | winner = **opposite** mirror (rank changed) |
| 51 | IndividualCRMQuestions#63 | B | yes | no | FUSION-BIAS | gold bm25 r31, merged 78; winner = report-contents QA |
| 52 | IndividualCRMQuestions#261 | B | **no** | yes | GOLD-LABEL-WRONG | gold = foreign nationals; winner = query verbatim (mobile app) |
| 53 | IndividualCRMQuestions#338 | B | yes | no | FUSION-BIAS | gold dense r50; winner = installments factor |
| 54 | ChequeQuestions#140 | C | yes | no | RERANK-BIAS | winner repeats query's literal sentence but never states the threshold |

---

## 5. Worked examples — full retrieved text vs gold

### Example A — `Company_CRM_Questions#319` (lexical reward-hack)
**QUERY:** برای دریافت گزارش اعتباری شرکت، آیا باید تمام شرکا موافقت کنند؟

**GOLD** (merged=36, single-leg BM25):
```
پاسخ کوتاه: خیر، تنها فرایند احراز هویت مدیرعامل …
```
**WINNER** (final #1, rerank 0.9998):
```
سوال: برای دریافت گزارش اعتباری شرکت، آیا باید تمام صاحبان امضا موافقت کنند؟ …
سوال: من مدیر عامل یه شرکتم، چک شرکت برگشت خورده …
```
The winner **literally embeds a near-identical sibling question** (`تمام…موافقت…کنند`), so it scores higher lexically than the gold (J 0.219 vs 0.100) — but never answers the asked question and pivots to a cheque question.

### Example B — `PublicQuestions.xlsx#16` (distractor-prefix + fusion)
**QUERY (colloquial):** اطلاعات بیمه م که کارفرمام نداده، اگه فیش حقوقی بیارم حساب میشه؟

**GOLD** (bm25=74, dense=None, merged=None) — *leads with an unrelated prior Q&A*:
```
… (tail of previous answer) …
سوال: آیا اطلاعات بیمه و فیش حقوقی که کارفرما ارسال نکرده، در امتیاز لحاظ میشود؟
پاسخ کوتاه: خیر، فقط اطلاعات ثبت‌شده و ارسال‌شده از منابع رسمی …
```
The gold shares **18 content words** with the query (J 0.281) yet reaches only BM25 rank 74 → RRF `1/(60+74)=0.0075` cannot enter merged. The dense leg never returns it because the embedding is dominated by the unrelated prefix.

### Example C — `PublicQuestions.xlsx#55` (malformed gold)
**QUERY:** بخش امکانات کسب و کارها در اعتباریتو چیه؟

**GOLD:** `سوال: بخش امکانات کسب و کارها در اعتباریتو چیه؟` → `دسته‌بندی: …` (**no پاسخ**)
**WINNER** (J=0.600, contains the query verbatim): also answerless; the sibling chunk holding the real vocabulary (`«جاری» و «تکمیل شده»`) is what actually got ranked.
No retriever can recover a ground truth that has no answer.

### Example D — `ChequeQuestions#140` (reranker phrase-mirror)
**QUERY:** توی گزارش چک نوشته «میانگین موجودی حسابهای بانکی فرد در سال گذشته بسیار کم بوده است.» …

**GOLD** (merged=13 → final 9): `تعیین آستانههای عددی … توسط خود مدل اعتبارسنجی … امکان افشای جزئیات دقیق وجود ندارد.`
**WINNER** (reason-code ATR2-6, 0.99): repeats the query's quoted sentence verbatim and adds `پیشنهاد بهبود: … موجودی را افزایش دهد` — topically identical, but never answers *what the threshold is*. The cross-encoder rewards the literal string echo.

---

## 6. Decision — prioritised remediation

### P0 — Remove the embedding prefix (measured +0.14 dense recall@100)
Stop embedding `Title:` / `Heading:` / `Type:` in `kb_manager/dense.py::build_context_text` (or gate it off). Keep title/heading as **metadata and BM25 fields** if desired, but keep them out of the vector. This directly fixes the "cheque-document gravity" and lifts the worst docs (PublicQuestions +0.35, categorized +0.20). Also reconcile the pgvector path, which already embeds content-only.

### P1 — Per-question QA re-chunking (kills distractor-prefix + many fusion losses)
Split each QA row into its **own** chunk (`سوال / پاسخ کوتاه / پاسخ کامل / دسته‌بندی`), question first, no previous-row prefix. ~11/22 Individual+Public golds have unrelated leading text; removing it restores BM25 term statistics and the embedding.

### P2 — Replace flat RRF + widen the rerank cut (fixes 18 fusion losses)
Lower `rrf_k` 60→10, weight/normalise the legs, add a leg-top-3 union guarantee, and widen final top-5 → top-10 (golds sit at final 6/9/10/11). Every fusion failure is single-leg vs dual-leg; the consensus bonus currently rewards being retrievable twice, not being correct.

### P3 — Reranker calibration (fixes the 27 lexical-bias + 2 phrase-mirror)
The cross-encoder rewards query-string echo over answerhood. Hard-negative-train on "topically identical but non-answering" pairs; add an answerhood-aware prompt/score. This is the single largest mechanism (50%).

### P4 — Gold/benchmark hygiene (≈1 in 4 failures are label defects)
Repair golds with missing answers (#44, #47, #376, #55, #116), mislabels (#261), and mis-aligned answers (#153); relabel where a cleaner twin is the true answer (#19, #59, #292, #24, #167, #338). Report a "gold-valid" subset to stop pessimistic bias.

### P5 — Query rewriting + normalisation
Colloquial/typo→formal (`نىارم`, `بدهیمو`, `قسطمو`, `پاس نشد`) and Arabic presentation-forms normalisation. These drive the dense `None` results on otherwise-BM25-reachable golds.

### Measurement change
Split the label: **A** = neither leg found it; **A′** = leg hit but RRF lost it. The current `merged is None` boundary under-reports fusion by 12 and over-states retriever failure.
