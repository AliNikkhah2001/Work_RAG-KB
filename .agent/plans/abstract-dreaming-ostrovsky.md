# Plan: Fix All Weak Spots (W1-W5) — One-by-One with Benchmark Regeneration

## Context
User requested comprehensive fix for all retrieval weak spots across v2→v4, with per-fix benchmark regeneration, across all complexities (Low/Med/High). Constraints clarified:
- **Latency:** Not tight — keep cross-encoder reranker. Target <5s acceptable (vs <2s strict). No need to disable reranker for non-keyword. Keep quality over speed.
- **Gemma 30B:** Pure black-box API via `KB_LLM_BACKEND=vllm/ollama` — **no local Gemma code edits**. Create dedicated branch + report for remote 2×GPU machine execution later.
- **Hit vs Top1:** Do both, compare, report trade-offs. **Hit prioritized** (Hit@5 > Top1) because missing the answer entirely is worse than ranking #2 vs #1, but Top1 matters for user trust. Will report matrices: Hit-priority vs Top1-priority operating points.
- **Branches:** Dedicated `feat/*` per sub-part, **not merged to master/main** until green. Current master is v4 baseline (89b70c7). New work stays on `feat/fix-all-*` branches.

## Version Baseline (Evidence)
- **v1:** 355 docs, 8291 chunks (3088 QA), Hit@5 68.3% MRR 0.266 — no dedup, TF-IDF leg
- **v2:** 355 docs, 6208 chunks (974 QA), Hit 90.0% MRR 0.736 — dedup fixed (+21.7% Hit, biggest win)
- **v3:** 355 docs, 6208 chunks, BM25+Dense, Hit 89.2% Top1 72.5% MRR 0.787 latency 1.9s (-34% by dropping TF-IDF)
- **v4:** 416 docs, 7758 chunks (1887 QA, 910 dup), BM25+char3gram + Dense contextual + Reranker mmarco-mMiniLM, Hit 90% Top1 65% MRR 0.775 latency 15.8s — typo Top1 90%→100%, keyword MRR +85% (0.180→0.333) but net Top1 regresses, corpus dedup regressed.

## Execution Plan — 8 Phases Ordered by Dependencies

### Phase 0 — Baseline Lock & Checkbox Plan (Low, 0.5d) ✅ Current
- [x] Create `IMPLEMENTATION_PLAN.md` checkbox list, `PERSIAN_RESOURCES.md`, `docs/synthetic-generation.md`
- [ ] Regenerate v4 baseline on **full 120q** `test_questions.json` (currently 20q small) → `versions/v4_baseline_120q/`
- [ ] Commit checkbox plan to README (Executive Summary → Detailed sections), push to `master` as tracking board — update `[x]` per phase.

### Phase 1 — W1-A: Keyword Junk Sanitization (Low, 0.5d) — **Highest ROI**
**Root:** `evaluation/query_formats.py:159` `format_keyword_only()` keeps `"گزارش اعتباری مدل: حقیقی و حقوقی…"` as part. Junk `مدل:` leaks via `generate_test_questions.py` regex.
**Fix:** Clean `format_keyword_only()` — `if ":" in part: continue`, strip `…\u2026\u200c`, regex `مدل\s*:\s*[^\n,،]+` removal, normalize `\u064a→\u06cc` before split. Same in `generate_test_questions.py`. One-off DB: `UPDATE chunks SET fields=json_remove where keyword LIKE "%مدل:%"`.
**Branch:** `feat/fix-keyword-extraction` (isolated from master)
**Test:** Regenerate `test_questions_small.json` 20q → benchmark → expect keyword Hit 33%→60%, MRR 0.33→0.55. Keep Hit-priority vs Top1-priority report.
**Files:** `evaluation/query_formats.py`, `generate_test_questions.py`

### Phase 2 — W1-D: Char n-gram Tuning (Low, 0.5d)
**Root:** `search.py:30` map has bug `\u0667` duplicated (8 missed, 9 never), char 3-gram weight = word 1.0 (dilutes IDF).
**Fix:** Fix `\u0667`→`\u0667`/`\u0668`, weight char 0.3 vs word 1.0, add bi-gram len<4, ablation log.
**Branch:** `feat/char-ngram-tuning` (extends Phase1 branch or standalone)

### Phase 3 — W5-A: FaMTEB Benchmark Execution (Low, 0.5d) — **No Gemma needed**
**Fix:** `pip install datasets ranx`, run `run_benchmark.py --famteb --max-samples 100 --top-k 5` (600 samples: synper_qa, nq_fa, miracl/fa). Offline HF cache afterwards. Extend `run_benchmark.py` to emit `Recall@100/MAP@100` via `famteb.py:282`/`evaluation/metrics.py`.
**Branch:** `feat/benchmark-famteb-live` — never merged until green, generates `versions/v5_famteb/` plots.
**Test:** nDCG@10 baseline for Persian, compare to FaMTEB leaderboard (Jina/BGE-m3).

### Phase 4 — W1-B: HyDE Black-Box Wiring (Low/Med, 1d) — **On feat/hyde-wiring**
**Root:** `query_reform.py:80` HyDE implemented but **not imported** in `search.py`; `embed_and_search` bug `dense.embed_query` (should be `dense.search(hyde_doc.content)`).
**Fix (black-box only):** Inject `llm.py:VLLMClient/OllamaClient` via `KB_LLM_BACKEND` env. In `search_knowledge_base()`: `if detect_query_type()=="keyword_only" or len(tokens)<5 → hyde_doc = HyDEGenerator(llm).generate(query) → dense.search(hyde_doc.content) → RRF(original + hyde)` with `weighted_rrf` 0.6/0.4. No Gemma code edits — report branch for remote.
**Branch:** `feat/hyde-wiring` (dedicated, not merged). Provide `docs/hyde-remote-report.md` with `curl` examples for 2×GPU machine.
**Test:** A/B keyword Hit/MRR, expect +10-15% recall (Gao et al. 2022). If `KB_LLM_BACKEND==mock` → skip.

### Phase 5 — W3: Multi-Query Rewriting (Low/Med, 1d) — **Conversational**
**Root:** `MultiQueryGenerator` `query_reform.py:160` beam 5 pseudo, `generate_beam()` alias, never wired.
**Fix:** Wire: `search_knowledge_base()` if `conversational|reworded|keyword_only` → generate 5 rewrites → `reciprocal_rank_fusion` across 5 Dense searches. Use `llm.py` black-box. Keep RRF k=60, L=5.
**Branch:** `feat/multi-query-wiring`
**Test:** reworded MRR +3-6% (Kostric & Balog CMQR), paraphrase Top1 regression should recover.

### Phase 6 — W4: Corpus Dedup Re-enable (Med, 1d)
**Fix:** Re-enable `dedup_questions=True` on 416-doc source `1405-05-20`, rebuild → expect 7758→~6200 chunks, Hit +5-10% (v1→v2 precedent). Snapshot `v5_dedup`.
**Branch:** `feat/corpus-dedup`

### Phase 7 — W2: Latency Optimization (Med, 1d) — Keep Reranker (Not Tight)
**Fix:** Since latency not tight, *keep* reranker but optimize: quantize (INT8 via optimum), `RERANKER_TOP_K` 50→30, batch 32→64, async rerank. Target 15.8s→~6s (still <5s per query warm? No, per 20q = 0.3s/q). Report Hit-priority (keep reranker) vs Top1-priority (disable reranker) operating points — **Hit prioritized** per user.
**Branch:** `feat/latency-optimize`

### Phase 8 — W5-B: Synthetic Generation Branch + Remote Report (Med, 1-2d)
**Fix:** Create **dedicated branch** `feat/synthetic-gemma-remote` **not merged to master** — contains `synthetic_generation/` + `docs/synthetic-generation-remote.md` with hardware spec 2×24GB, `vllm --tensor-parallel-size 2`, `ollama pull gemma2:27b`, `KB_LLM_BACKEND=ollama KB_LLM_MODEL=gemma2:27b synthetic_generation/run_generation.py --limit 10` smoke test, full 50K QA + 15K conv, validator threshold 0.7, `human_audit 100`. Fix `base.py` dead inheritance, orphaned `typo_injection.txt`/`keyword_extraction.txt` wiring note (deferred). No execution here — branch pushed for remote pull.
**Branch:** `feat/synthetic-gemma-remote`

### Phase 9 — Docs, Plots, Release (Low, 0.5d)
* Update `README.md` with executive summary v2→v5 table (Hit-priority vs Top1-priority), 7 detailed retrieval sections with LaTeX, update checkboxes `[x]`, add Persian resources links.
* Regenerate 7 comparison plots `data/plots/{version_comparison_overall,per_format,radar,keyword_only_improvement,latency_breakdown}` with new v5 numbers.
* Create version snapshot `v5_complete` (120q + FaMTEB) — immutable `versions/v5/`.

## Verification Strategy (After Each Phase)
- Syntax: `python -m py_compile <file>`; unit `pytest`
- Benchmark: 20q → 120q → FaMTEB 100/sample; record `data/benchmark_results.json`, `data/ir_metrics.json`, `versions/<ver>/manifest.json`
- Comparison: `create_comparison_plots.py` → `BENCHMARK_COMPARISON.md` auto-update
- PR: each `feat/*` PR only merges after green; master checkboxes updated `[ ]→[x]` per phase.
