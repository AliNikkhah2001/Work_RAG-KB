# Comprehensive Report — W1 Keyword-only & W5 Evaluation Coverage

## W1 — Short / Keyword-only Queries: Hit 33-45% → Fix

### Problem Statement
Keyword-only benchmark queries (`format_keyword_only`) have Hit@5 33-45% across v2→v4 (vs 90% overall). MRR 0.180→0.333 (+85% after v4 char n-grams) but Hit actually dropped 45%→33% on 20q sample. Short queries (<5 tokens, e.g., "امتیاز اعتباری تسهیلات بازپرداخت") suffer **vocabulary mismatch** + **length imbalance** (BM25 needs term overlap, Dense needs context).

### Root Cause — 3 Layers (Verified via `data/test_questions.json`)
1. **Source pollution:** Excel `Keyword` column already contains `Keyword + " مدل: " + Model`. DB `fields["keywords"] == "بروزرسانی، بازپرداخت، وام، گزارش اعتباری مدل: حقیقی و حقوقی…"` — `…` is truncated cell `U+2026`.
2. **Extraction:** `evaluation/query_formats.py:159` `re.split(r"[،,؛;\n]+")` keeps `"گزارش اعتباری مدل: حقیقی و حقوقی…"` as one part; `strip("[]\"' ")` does not remove `:` or `…`.
3. **Retrieval:** Tokens `مدل: حقیقی` appear in 823/977 questions (IDF ≈0) — dilutes BM25; Dense short-query embedding is sparse vs long chunk `Title+Heading+Content`.

Example raw:
```
"query": "بروزرسانی بازپرداخت وام گزارش اعتباری مدل: حقیقی و حقوقی…"
```
After old code → parts = `["بروزرسانی","بازپرداخت","وام","گزارش اعتباری مدل: حقیقی و حقوقی…"]` → 25% noise.

### Fix Implemented (Branch `feat/fix-keyword-extraction` → `master:15a8c04`)
- `evaluation/query_formats.py:159` — `re.sub(r"\s*مدل\s*:\s*[^\n,،؛]+","",keywords)` before split, strip `…\u2026`, ZWNJ→space, filter `if ":" in p or "مدل" in p: continue`, junk_exact `{"حقیقی","حقوقی"}`.
- `web/routes/search.py:30` — fix `\u0667` dup bug (`\u0667:8` → `\u0668:8`, `\u0667:7` kept).
- Expected: Hit 33%→60%, MRR 0.33→0.55 on 20q (to be verified via full 120q regeneration `versions/v4_baseline_120q/`).

### Additional Mitigations (Planned, Low/Med Complexity)
| Mitigation | Mechanism | Gain | Status |
|------------|-----------|------|--------|
| **HyDE (W1-B)** | `query_reform.py:HyDEGenerator` Persian prompt → pseudo-doc → `dense.search(hyde_doc)` → RRF(original+hyde) 0.6/0.4 | +10-15% recall for <5-word queries (Gao et al. 2022) | **Wire as black-box** `feat/hyde-wiring` — fix `embed_and_search` bug `dense.embed_query`→`dense.search` |
| **Entity boost** | `keyword_extraction.txt` → DSL `must(original) + should(entities)` | +3-6% MRR, avoids drift | `feat/entity-boost` |
| **Char n-gram tuning** | Weight word 1.0 / char 0.3, bi-gram len<4 | +5% typo | Done in v4, verify |

### Verification Plan
- Regenerate `test_questions_small.json` 20q → benchmark → `BENCHMARK_COMPARISON.md` update
- A/B: baseline vs sanitized vs sanitized+HyDE (Hit-priority, Hit@5 primary per user)
- Commit branch `feat/fix-keyword-extraction` already pushed → merged to `master:15a8c04`

---

## W5 — Evaluation Coverage: Blind Spots → Fix

### Current Coverage
- Only `data/benchmark_results.json` 20q small (and `test_questions.json` 120q full exists but not run on v4). `versions/v4_retrieval/benchmarks` 5q truncated. `ir_metrics.json` only `map@5/mrr@5/ndcg@5/recall@5` — missing `Recall@100/MAP@100`.

### Undone Tasks (from `IMPLEMENTATION_PLAN.md` + `famteb.py` + `synthetic_generation/`)

| Task | Code Status | Runtime Status | Fix |
|------|-------------|--------------|-----|
| **FaMTEB 6 datasets** `famteb.py:360` (`synper_qa`, `synper_chatbot_rag_topics`, `nq_fa` BEIR-Fa, `miracl/miracl fa`) | **Implemented** `load_famteb_retrieval_dataset` / `load_famteb_benchmark` / `compute_all_metrics` | **Never executed** — needs `pip install datasets` + HF download (once, offline cacheable). `run_benchmark.py --famteb --max-samples 100` exists but not run. | Run on `feat/benchmark-famteb-live`: `run_benchmark.py --famteb --max-samples 100 --top-k 5` → 600 samples → `versions/v5_famteb/` |
| **FaMTEB loaders schema mismatch** | Implemented but skips rows where `query_field` empty → 0 queries if column names differ (smoke test 15:30 showed 0q / 108k generation) | Fix column mapping: inspect HF `synper_qa` actual columns via `ds.column_names` |
| **Synthetic Gemma 30B** `synthetic_generation/run_generation.py` + `qa_generator.py` + `conv_generator.py` + `validator.py` + `prompts/*.txt` | **Code complete** 58-line `config.yaml` (8 samples/chunk, 30% conv, threshold 0.7) | **0 outputs** `output/` empty — requires Gemma. `base.py` dead inheritance, `typo_injection.txt`/`keyword_extraction.txt` orphaned | **Dedicated branch** `feat/synthetic-gemma-remote` **not merged** — report for 2×GPU remote: `KB_LLM_BACKEND=ollama KB_LLM_MODEL=gemma2:27b synthetic_generation/run_generation.py --limit 10` smoke |
| **Metrics extension** `famteb.py:282` `compute_all_metrics` hit/recall/map/ndcg@1,3,5,10,20,100 + mrr | Dead code — `run_benchmark.py` only calls `RanxRetrievalEvaluator` @k=5 | Extend `run_benchmark.py` to emit `Recall@100/MAP@100/NDCG@10` |
| **Config drift** `synthetic_generation/config.yaml` vs `config.py:144` | YAML `model.backend/base_url/tensor_parallel` ignored — `llm.py:168` reads `KB_LLM_BACKEND` env defaults `mock` → silent mock | Document env vs yaml in `docs/synthetic-generation.md` |

### Black-Box API — Exact Steps (No Local Gemma Code Edits)

**FaMTEB (Gemma-free, HF cacheable):**
```powershell
pip install datasets ranx
huggingface-cli download MCINext/synthetic-persian-qa-retrieval --repo-type dataset
python run_benchmark.py --famteb --max-samples 100 --top-k 5
python run_benchmark.py --famteb --famteb-datasets synper_qa miracle_fa --max-samples 10 --top-k 10
# outputs: data/benchmark_results.json, data/ir_metrics.json, data/plots/{hit_rate_by_format,mrr_by_format}
```

**Synthetic generation (requires LLM API, run on remote 2×24GB machine):**
```powershell
$env:KB_LLM_BACKEND="ollama"      # or openai/vllm/mock
$env:KB_LLM_MODEL="gemma2:27b"
$env:OLLAMA_BASE_URL="http://localhost:11434" # ollama pull gemma2:27b; ollama serve
# or: $env:KB_LLM_BACKEND="vllm"; $env:KB_LLM_BASE_URL="http://localhost:8000/v1" # docker vllm/vllm-openai:google/gemma-2-27b-it --tensor-parallel-size 2
python synthetic_generation/run_generation.py --config synthetic_generation/config.yaml --db-path data/kb_test.db --output-dir synthetic_generation/output --limit 10
# outputs: synthetic_generation/output/synthetic_qa.jsonl, synthetic_conversations.jsonl, generation_metadata.json
```

### Next Actions (All Prioritized, Different Complexities)
- [x] **W1-A (Low)** — Sanitization done, pushed, merged
- [ ] **W1-B (Low/Med)** — Wire HyDE on `feat/hyde-wiring` (black-box), benchmark A/B, update `BENCHMARK_COMPARISON.md`
- [ ] **W5-A (Low)** — Execute FaMTEB on `feat/benchmark-famteb-live`, generate `versions/v5_famteb/` + plots
- [ ] **W5-B (Med)** — Keep `feat/synthetic-gemma-remote` as report branch for remote (see `docs/synthetic-generation-remote.md` to be created)
- [ ] **Docs (Low)** — Mark README checkboxes `[x]` per phase, regenerate `BENCHMARK_COMPARISON.md` with Hit-priority vs Top1-priority report

### Hit vs Top1 Trade-off (Per User: Hit prioritized)
- **Hit@5 prioritized** — missing answer entirely is worse than rank #2 vs #1
- **Top1 kept** for trust — will report both: operating points (Hit-optimal: keep reranker Top1 65% vs Top1-optimal: disable reranker Top1 72.5% but Hit 89.2%). Report in `BENCHMARK_COMPARISON.md` when both measured.

