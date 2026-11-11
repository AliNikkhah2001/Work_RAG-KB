# Verification — subspace distill & draft prompts v1.0.4

**Date:** 1405-08-20 distill run  
**Reviewer role:** VERIFIER agent  
**Artifacts:** `subspace_counts.json`, `clusters/*.json`, `topic_cards.yaml`, `hard_pairs.jsonl`  
**Draft prompts:** `DRAFT_prompts_orchestrator_v1.0.4.yaml`

## Executive summary

| Area | Verdict |
|------|---------|
| QA counts & arithmetic | **PASS** |
| Topic cards (intents / sticky) | **PASS** (minor summary bleed) |
| Cluster quality (large subspaces) | **CONDITIONAL FAIL** |
| Hard-pair mining | **CONDITIONAL PASS** |
| Draft prompts v1.0.4 | **PASS** (ready for pack integration + runtime alignment) |
| Enable `subspace_ambiguity_route` now | **NO** — enable after 9-way protos/clf + prompt pack pin |

---

## 1. Count sanity

| Check | Expected | Observed | Result |
|-------|----------|----------|--------|
| Total QA | README | 1078 | PASS |
| Skipped incomplete | ~24 | 24 | PASS |
| Sum of `per_subspace` | = total | 1078 | PASS |
| Subspace count | 9 L2 | 9 | PASS |
| Hard pairs lines | cap 40 | 40 | PASS |
| Cluster count (README) | 57 | (not re-counted; spot-check OK) | PASS |

**Distribution notes (sane but skewed):**

- `facilities_person` 451 / `facilities_legal` 330 dominate (~73% of corpus).
- `customers_*` 3 each — taxonomy placeholders only; do not over-fit prompts.
- `guarantor` / `supplier_legal` 13 — adequate for topic cards, weak for hard-pair quotas.

---

## 2. Cluster / summary spot-check

### Pass

- **guarantor.json:** 3 clusters, representatives match legal/collateral product language; medoids readable.
- **credit_ladder.json:** clusters align with «طرح/انتخاب/استفاده از نردبان».
- **cheque.json** (sampled): distinct cheque vs rank themes in smaller clusters.

### Issues (residual)

1. **Mega-cluster (`facilities_person_c00`, size 330):** ~73% of subspace in one blob; summaries collapse to generic «امتیاز/اقساط». Confirms METHOD_NOTES duplicate-bank inflation; hurts sticky/threshold logic (≥3 chunks look homogeneous).
2. **Template duplication:** Several subspaces repeat identical `summary_fa` across adjacent clusters (e.g. guarantor c00/c01; credit_ladder c00/c02). Safe for cards, misleading if pasted into prompts (draft avoids this).
3. **Topic card bleed:** `cheque` in_scope line mentions «اشخاص حقوقی/شرکت‌ها» for one small cluster — likely mis-tagged questions in source xlsx, not product truth. Keep `cheque` scoped to حقیقی in prompts.
4. **credit_ladder outlier:** Representative «رمز برام فرستادین چرا؟» in card — OTP/auth noise; low risk if not promoted to sticky phrases.

---

## 3. Hard-pair quality

### Strengths

- Priority buckets filled as designed:  
  `cheque↔facilities_person` (10), `facilities_legal↔facilities_person` (12),  
  `facilities_legal↔general` (9), `credit_ladder↔facilities_person` (9).
- High-value clarify triggers present: cross-report comparisons, shared «امتیازم چنده؟», صیاد wording fork, person vs legal parallel questions.

### Weaknesses

| Issue | Severity | Mitigation in v1.0.4 prompts |
|-------|----------|------------------------------|
| **No pairs** for guarantor, supplier_legal, customers_* | Medium | Explicit keyword routes in hybrid/suggest |
| **13/40 pairs** with similarity &lt; 0.70 (mostly credit_ladder↔facilities) | Medium | Use curated 12 examples; ignore weakest pairs |
| **Exact duplicates** (similarity 1.0) across subspaces | Medium | Data hygiene; teaches `unclear` not single L2 |
| **رتبه A/B/D/E ↔ C** char-sim artifacts | Low | Omitted from prompt examples |
| `facilities_legal↔general` pairs at ~0.60 for «رتبم/امتیازم» cross-swap | Low | Covered by `unclear` rule |

**Verdict:** **CONDITIONAL PASS** for prompt distillation; rebalance hard_pairs in a future run (guarantor↔facilities_person, supplier↔customers_legal, dedup before clustering).

---

## 4. Draft prompts v1.0.4

| Requirement | Status |
|-------------|--------|
| Keys: classify, hybrid_classify, clarify, specify_judge, suggest_subspace | PASS |
| L2 ids (9 + ambiguous + unclear) | PASS |
| Topic-card intents, no recursive summaries | PASS |
| ≤12 hard-pair-style examples (question→label) | PASS (12 in hybrid_classify) |
| Sticky / clarify contracts | PASS |
| Impact-query → unclear | PASS |
| L1 grouped clarify (4 groups) | PASS |
| Compact size | PASS |

**Integration gaps (not blockers for draft):**

- Orchestrator `hybrid_classify.py` Gemma fallback still maps only 4 Persian buckets — must extend parser for new L2 labels before pack pin.
- `SUBSPACES` / `subspace_protos.npz` / joblib clf still trained on **4** labels — 9-way artifacts required before trusting hybrid classifier fast path.
- `suggest_subspace` output `unclear` is already consumed in `infer_specification.py`; `ambiguous` in hybrid should set `applied_subspace=None` + clarify (verify graph edge).

---

## 5. `subspace_ambiguity_route`

**Current config (given):** `subspace_ambiguity_enabled=True`, `subspace_ambiguity_route=False`, scorer protos **4-way**.

**Recommendation:** **Keep routing OFF** for this release; keep **compute ON** for audit.

Enable routing when **all** are true:

1. KB registry taxonomy lists 9 L2 subspaces and retrieval filters accept them.
2. Regenerated `subspace_protos.npz` (+ optional clf) over 9 labels with updated centroids.
3. Prompt pack **v1.0.4** pinned; Gemma escalation parses full L2 set.
4. Eval smoke: cheque↔facilities impact queries → clarify rate ↑; sticky thrash ↓ on 3-chunk bleed cases.

**Rationale:** Turning route on with 4-proto scorer while prompts advertise 9 L2 labels will increase false ambiguous signals and sticky/filter mismatch (worse than status quo).

---

## 6. Residual risks

1. **Sticky thrash** on generic follow-ups until sticky contract + history-aware suggest_subspace land in graph (not prompt-only).
2. **Under-clarify** on «امتیازم چنده» if impact rule not wired in retrieve heuristics alongside LLM judge.
3. **Filter bleed** from mega-cluster retrieval in facilities_person until dedup/second-pass clustering.
4. **Tiny subspaces** (customers_*) may never retrieve well — acceptable; clarify L1 grouping covers UX.
5. **Draft in reports/** — must copy to `packs/prompts/orchestrator/v1.0.4.yaml` and registry refresh to deploy.

---

## 7. Checklist (ship gate)

- [ ] Dedup or down-weight «جایگزین» facilities xlsx before re-cluster
- [ ] Add hard-pair quota: guarantor, supplier_legal, customers_*
- [ ] Build 9-way proto bundle + retrain hybrid clf
- [ ] Extend Gemma label parser for 9 L2 + ambiguous/unclear
- [ ] Pin v1.0.4 prompt pack; eval clarify/sticky metrics on golden set
- [ ] Then set `subspace_ambiguity_route=true` in staged rollout
