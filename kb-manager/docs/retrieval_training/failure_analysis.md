# Retrieval failure analysis (Agent 2 — v10 frozen baseline)

## Header / determinism
- code HEAD: 2825faef717d8cc3dbfb4dd350a5b795f9bf3999
- tag v10-retrieval-baseline: 96438c1e8caac5689f5caef11868265c4316e8a1
- kb_manager/ tracked files clean: True (only untracked dir kb_manager/retrieval_training/ from parallel work, not this agent; `git diff 96438c1 HEAD -- kb_manager/` empty)
- HEAD vs tag: freeze commit only (adds baseline_freeze.json); `git diff 96438c1 HEAD -- kb_manager/` empty
- db: data/kb_9_7_2026.db sha256=9eac43aedee0af1810f309f5cf43a6710f6bacb7af58759861cccac0683941d1 (freeze=9eac43aedee0af1810f309f5cf43a6710f6bacb7af58759861cccac0683941d1 match=True)
- npz: data/dense_embeddings.npz sha256=5a4909985bc5e78958a65e2bd8f91e295f061a0444196ae8dca1ce5ea0f4b923 (freeze=5a4909985bc5e78958a65e2bd8f91e295f061a0444196ae8dca1ce5ea0f4b923 match=True)
- models: dense=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 reranker=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- retrieval config: {"rrf_k": 60, "rerank_top_k": 100, "keyword_boost": 3.0, "synonym_beam": 5, "fusion_alpha": 0.7, "pin": "merged-top3+bm25-top3<=10"}
- env: KB_DB_URL=sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_9_7_2026.db KB_SOURCE_DIR=D:/Code/KB/kb-source/KB_9.7.2026
- run at (UTC): 2026-09-12T14:07:22.762118+00:00 (finalize pass; search passes ran 2026-09-12 ~10:30-13:00 UTC and ~13:10-13:55 UTC) top_k=100 per-query-timeout=600s
- elapsed: search ~139.1 min (pass 1, full 511) + ~44.3 min (pass 2, remaining 152 after harness timeout at row 325) + finalize 0.4 min; single-query mean ~16 s (CPU rerank of 100 candidates)

## Method (replica of benchmarks.py::_run_massive)
- QA files: rglob *.xlsx minus ~$/TestQuestion/واژگان معادل/محدودیت ها stems, first dir with crm_qa sheets; GT 3-step: metadata.fields.question==q → content startswith/in 'سوال: {q}' → q[:30] in content.
- Ranks: first-index match over steps.bm25_results / dense_results / merged_candidates / final_results (each ≤100 since top_k=100).
- Classes (D checked first so Hit@5 == D rate): D final≤5; else A merged None; else B merged>20 (gold in ≥1 exposed leg top-100 verified below); else C (merged≤20, final>5 or absent).

## Counts
- total QA rows: 573
- evaluated (gold-mapped): 511 (= jsonl lines)
- skipped: 62 (empty-question=1 no-gold-chunk=61 file-not-indexed=0)
- A retriever_failure (gold not in merged top-100): 18
- B fusion_failure (leg top-100 → merged>20): 2 (with gold in ≥1 exposed leg top-100: 2/2)
- C reranker_failure (merged≤20 → final>5/absent): 2
- D success (final≤5): 489
- error (timeout/exception): 0

## Aggregate (over evaluated non-error rows)
- N=511 Hit@1=0.8376 (428/511) Hit@5=0.9569 MRR=0.8884

## Comparison vs known baseline (data/massive_results_baseline_oldcode.json: 573 total, 489 pass, 22 fail, 62 skipped)
- overlap_rows=511 baseline_hit5=489 ours_hit5=489 newly_fixed=2 newly_broken=2
- newly fixed ids: ["Individual_CRM_Questions#45", "PublicQuestions#39"]
- newly broken ids: ["Individual_CRM_Questions#3", "PublicQuestions#48"]

## Examples: A — retriever_failure
- Individual_CRM_Questions#100 bm25=None dense=None merged=None final=None
  - Q: اگر هیچ وامی نگرفتم رتبه اعتباریم جند حساب میشه؟
  - gold: 9efb8777-f623-4470-b9ef-805ef46c1be4
- Individual_CRM_Questions#224 bm25=None dense=None merged=None final=None
  - Q: انواع مختلف تسهیلات چه تاثیری در امتیاز اعتباری دارد؟
  - gold: 9e550066-152d-4626-a9d3-a34540b8cbc2
- Individual_CRM_Questions#239 bm25=None dense=None merged=None final=None
  - Q: سابقه منفی در گزارش اعتباری با چه منطق و مبنایی درج می شود؟
  - gold: 44e6083c-36bc-4ea5-b967-75d7a3c43784
- Individual_CRM_Questions#292 bm25=None dense=None merged=None final=None
  - Q: وام و اعتبار کدوم بانک ها و اپلیکیشن ها توی گزارش هست؟
  - gold: 41f3f954-32e1-4b5e-9caa-a67e4e973310

## Examples: B — fusion_failure
- Individual_CRM_Questions#19 bm25=27 dense=None merged=96 final=9
  - Q: چه عواملی امتیاز اعتباری من را کاهش می‌دهد؟
  - gold: efbe9b76-0961-400f-b02c-b67167fadcf0
- PublicQuestions#24 bm25=1 dense=None merged=38 final=10
  - Q: امتیازم (مثال امتیاز) عوض شده ولی رتبم (مثال رتبه) عوض نشده. چرا؟
  - gold: 06aacada-0a0a-4df8-b829-c2db23b0f4c2

## Examples: C — reranker_failure
- Individual_CRM_Questions#17 bm25=13 dense=46 merged=7 final=11
  - Q: چقدر طول می‌کشد تا گزارش اعتباری من به‌روز شود؟
  - gold: 6f60f4ec-c241-40bd-a373-d8a9e7d41d2e
- Individual_CRM_Questions#6 bm25=2 dense=23 merged=4 final=8
  - Q: چه مدت زمان طول می‌کشد تا بهبود در رفتار مالی من،امتیاز اعتباری و رتبه اعتباری‌ام را ارتقا دهد؟
  - gold: 5564723b-890e-484f-86ab-a22ce454109e

## Skipped log (query_id, reason)
- PublicQuestions#20 no-gold-chunk :: PublicQuestions.xlsx :: مهمترین درسی که باید در مورد اعتبارسنجی بدانم چیست؟
- PublicQuestions#30 no-gold-chunk :: PublicQuestions.xlsx :: اعتبارسنجی چیست؟
- PublicQuestions#31 no-gold-chunk :: PublicQuestions.xlsx :: هدف از اعتبارسنجی چیست؟
- PublicQuestions#44 no-gold-chunk :: PublicQuestions.xlsx :: رتبم چنده؟
- PublicQuestions#46 no-gold-chunk :: PublicQuestions.xlsx :: امتیازم چنده؟
- PublicQuestions#53 no-gold-chunk :: PublicQuestions.xlsx :: آیا امکان دریافت گزارش اعتباری از طریق اپلیکیشن موبایل وجود دارد؟
- PublicQuestions#55 no-gold-chunk :: PublicQuestions.xlsx :: بخش امکانات کسب و کارها در اعتباریتو چیه؟
- PublicQuestions#56 no-gold-chunk :: PublicQuestions.xlsx :: «جاری» و «تکمیل شده» در امکانات کسب و کارها یعنی چی؟
- Individual_CRM_Questions#84 no-gold-chunk :: Individual_CRM_Questions.xlsx :: نمره منفی برااقساط معوقمه ک 3 ماه از پرداختش گذشته و پرداخت کردم و برام نمره منف
- Individual_CRM_Questions#166 no-gold-chunk :: Individual_CRM_Questions.xlsx :: من یک وام 100 میلیونی با اقساط 60 ماهه از بانک ملی گرفتم و 30 قسط رو پرداخت کردم
- Individual_CRM_Questions#288 empty-question :: Individual_CRM_Questions.xlsx :: 
- ChequeQuestions#3 no-gold-chunk :: ChequeQuestions.xlsx :: من یه چک ضمانت دادم به بانک فلان (اسم بانک) برا اون رتبه م شده C2؟(یا هر رتبه و 
- ChequeQuestions#5 no-gold-chunk :: ChequeQuestions.xlsx :: من چکم خط خوردگی داشت که برگشت خورد. رتبه م برا اون شده C2؟ (یا هر رتبه و امتیاز
- ChequeQuestions#6 no-gold-chunk :: ChequeQuestions.xlsx :: من چکم تاریخش غلط بود برگشت خورد. رتبه م برا اون شده C2؟ (یا هر رتبه و امتیازی)
- ChequeQuestions#8 no-gold-chunk :: ChequeQuestions.xlsx :: میتونم برا شرکت گزارش چک بگیرم؟
- ChequeQuestions#26 no-gold-chunk :: ChequeQuestions.xlsx :: من کلی چک دست مردم دارم. امتیازم چرا شده D1؟
- ChequeQuestions#28 no-gold-chunk :: ChequeQuestions.xlsx :: من یه حساب مشترک با همسرم (یا هر کسی) دارم. میتونم گزارش چک بگیرم؟
- ChequeQuestions#30 no-gold-chunk :: ChequeQuestions.xlsx :: من چکم رو منتقل کردم. میتونم گزارش اون جکو بگیرم؟
- ChequeQuestions#38 no-gold-chunk :: ChequeQuestions.xlsx :: برا گرفتن دسته چک برای شرکت گزارش اعتباری لازمه؟
- ChequeQuestions#42 no-gold-chunk :: ChequeQuestions.xlsx :: اینکه وام قبلی رو هنوز پس ندادم تو گزارش چکم موثره؟
- ChequeQuestions#47 no-gold-chunk :: ChequeQuestions.xlsx :: من سهام دارم تو بورس، این توی گزارش چک میاد؟
- ChequeQuestions#59 no-gold-chunk :: ChequeQuestions.xlsx :: تو گزارش چک از اطلاعات بیمه و مالیات هم استفاده میشه؟
- ChequeQuestions#62 no-gold-chunk :: ChequeQuestions.xlsx :: چه طوری A شم؟
- ChequeQuestions#64 no-gold-chunk :: ChequeQuestions.xlsx :: من کارمند بانکم. تو گزارش چکم موثره؟
- ChequeQuestions#65 no-gold-chunk :: ChequeQuestions.xlsx :: من کارمند بانکم. امتیازم چرا C2  شده؟
- ChequeQuestions#75 no-gold-chunk :: ChequeQuestions.xlsx :: من در دو زمان مختلف گزارش اعتباری خود را دریافت کرده ام و در این فاصله هیچ قسط ی
- ChequeQuestions#81 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه چک من سبز شده. چی کار کنم که آبی شود؟
- ChequeQuestions#82 no-gold-chunk :: ChequeQuestions.xlsx :: چرا گزارش چک مهمه؟
- ChequeQuestions#83 no-gold-chunk :: ChequeQuestions.xlsx :: تعداد زیادی از چک هایی که من صادر کرده ام، نقد شده است و امتیازم B شده است. اما 
- ChequeQuestions#85 no-gold-chunk :: ChequeQuestions.xlsx :: کدوم امتیازها خوبه؟
- ChequeQuestions#86 no-gold-chunk :: ChequeQuestions.xlsx :: کدوم رتبه ها خوبه؟
- ChequeQuestions#87 no-gold-chunk :: ChequeQuestions.xlsx :: کدوم رتبه ها بده؟
- ChequeQuestions#88 no-gold-chunk :: ChequeQuestions.xlsx :: کدوم امتیازها بده؟
- ChequeQuestions#89 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه چک من A شده اما سابقه بدهی معوق و مشکوک الوصول دارم. چطور ممکنه؟
- ChequeQuestions#90 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه چک من B شده اما سابقه بدهی معوق و مشکوک الوصول دارم. چطور ممکنه؟
- ChequeQuestions#91 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه A یعنی چی؟
- ChequeQuestions#92 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه B یعنی چی؟
- ChequeQuestions#93 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه C یعنی چی؟
- ChequeQuestions#94 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه D یعنی چی؟
- ChequeQuestions#95 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه E یعنی چی؟
- ChequeQuestions#96 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه A1 یعنی چی؟
- ChequeQuestions#97 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه A2 یعنی چی؟
- ChequeQuestions#98 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه A3 یعنی چی؟
- ChequeQuestions#99 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه B1 یعنی چی؟
- ChequeQuestions#100 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه B2 یعنی چی؟
- ChequeQuestions#101 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه B3 یعنی چی؟
- ChequeQuestions#102 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه C1 یعنی چی؟
- ChequeQuestions#103 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه C2 یعنی چی؟
- ChequeQuestions#104 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه C3 یعنی چی؟
- ChequeQuestions#105 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه D1 یعنی چی؟
- ChequeQuestions#106 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه D2 یعنی چی؟
- ChequeQuestions#107 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه D3 یعنی چی؟
- ChequeQuestions#108 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه E1 یعنی چی؟
- ChequeQuestions#109 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه E2 یعنی چی؟
- ChequeQuestions#110 no-gold-chunk :: ChequeQuestions.xlsx :: رتبه E3 یعنی چی؟
- ChequeQuestions#111 no-gold-chunk :: ChequeQuestions.xlsx :: کدوم دلایل برگشت چک توی امتیاز مهمه؟
- ChequeQuestions#112 no-gold-chunk :: ChequeQuestions.xlsx :: گزارش چک به چه دردی می خوره؟
- ChequeQuestions#113 no-gold-chunk :: ChequeQuestions.xlsx :: توی گزارش چک نوشته «میانگین موجودی حسابهای بانکی فرد در سال گذشته بسیار کم بوده 
- ChequeQuestions#114 no-gold-chunk :: ChequeQuestions.xlsx :: توی گزارش چک نوشته «فرد در 2 سال گذشته تعداد زیادی چک برگشتی داشته است.». زیاد ب
- ChequeQuestions#115 no-gold-chunk :: ChequeQuestions.xlsx :: من امتیاز چکم از یه نفر دیگه بیشتر شده. یعنی ریسکم کمتره؟
- ChequeQuestions#116 no-gold-chunk :: ChequeQuestions.xlsx :: برای دریافت گزارش چک برا حساب مشترک آیا باید تک تک پول بدم؟
- ChequeQuestions#118 no-gold-chunk :: ChequeQuestions.xlsx :: خطوط قرمز چیه؟

## Determinism audit (correction to the "deterministic" assumption)
- Mechanism: kb_manager/query_expansion.py beam-4 ("reworded") shuffles with `random.Random(hash(query) % 2**32)`; str `hash()` is salted per process (PYTHONHASHSEED) → beam queries differ across processes → BM25/dense pools and ranks jitter cross-process. Within one process results are stable.
- Incident: pass 1 (detached) survived the harness timeout-kill and ran to completion (full 511 rows, DONE A=18 B=2 C=2 D=489 err=0 hit5=0.9569 mrr=0.8930, 139.1 min) while pass 2 (--resume) concurrently appended the remaining 152 rows → 18 duplicate lines + 2 torn fragments. Repaired via raw_decode recovery + dedupe by query_id (first occurrence kept); 16 dup ids, of which 13 had rank-only diffs (no class flips among the kept set beyond what is reported).
- Cross-process agreement: BOTH full passes independently produced A=18 B=2 C=2 D=489 (identical class counts); MRR differs only at rank-noise level (0.8930 vs 0.8887/0.8884). Class assignments are robust to the seed noise.
- Probe (8 dup-cluster queries, PYTHONHASHSEED=0 vs 1, same code/DB): leg ranks jitter (e.g. #66 bm25 3→4 dense 25→6 merged 2→1; #69 merged 1→3; #76 final 1→2) but final≤5 (class D) unchanged in all 8. Table: #66 (2,1)→(1,1); #69 (1,1)→(3,1); #70 (1,1)→(1,1); #71 (1,1)→(1,1); #72 (3,1)→(3,1); #74 (9,4)→(8,4); #76 (1,1)→(1,2); #77 (1,1)→(1,1) as (merged,final) seed0→seed1.
- Boundary sensitivity in final data: 30 rows with merged rank 15–25 (B/C boundary zone; only 2 are class B, rest rescued to D); 13 rows with final rank 4–7 (C/D boundary zone). These ids are the most likely to flip class under a different hash seed.
- Recommendation: set PYTHONHASHSEED=0 (or seed the beam-4 shuffle deterministically, e.g. hashlib) before generating retrieval-training labels; treat single-rank differences as noise, class labels as stable.
- Double-sampled ids (16 total per repair log, first occurrence kept; 10 verified in printout, all class D in kept data): ChequeQuestions#66 #69 #70 #71 #72 #74 #76 #77 #78 #79 (+6 more in the same file window, ids not printed before rewrite).
