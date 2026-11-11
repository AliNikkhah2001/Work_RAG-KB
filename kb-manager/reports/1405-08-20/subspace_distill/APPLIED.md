# Applied to code (2026-11-11)

## KB (`fix/kb-ingest-schema-flaws`)

- `packs/prompts/orchestrator/v1.0.4.yaml` — L2 judge/classify/clarify/suggest from distill draft
- `packs/taxonomy/v1.0.2.yaml` — 9 L2 nodes (+ supplier / customers)
- `packs/retrieval/v1.0.0.yaml` pins: schemas 1.0.1, taxonomy 1.0.2, canned 1.0.2, prompts_orchestrator 1.0.4, prompts_guardrails 1.0.1
- Prior ingest fixes retained: schema 1.0.1, incomplete QA skip, glossary synonyms, greeting canned, taxonomy AND tokens

## Orchestrator (`feature/l2-subspaces-prompts`)

- `subspace_routing.py` — shared L2 map, hierarchy extract (no silent person default), impact→clarify, label parse
- `config` / `content_registry` / fallback canned+prompts — 9-way maps
- `classify` / `infer_specification` / `retrieve` / `hybrid_classify` / `api` sticky declarations — L2 aware
- `graph.should_clarify` — honor `needs_clarification` before ≥3-chunk bypass
- `subspace_ambiguity_route` remains **False** until 9-way embedding protos are rebuilt

## Not done in this apply

- Rebuild `subspace_protos.npz` / SVM for 9 labels
- Parent gitlink advance / PRs (await explicit commit request)
- Live DB ingest of 1405-08-20
