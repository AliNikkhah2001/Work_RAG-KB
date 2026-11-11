# Subspace QA distill — 1405-08-20

## Method

1. Walk all `.xlsx` under `kb-source/1405-08-20/پایگاه دانش`.
2. Parse with `kb_manager.parsers.get_parser` (PYTHONPATH=kb-manager). Keep sheets with schema `crm_qa` or Q/A-like headers.
3. Drop incomplete rows (missing question **or** no non-empty answer field). **No Postgres writes.**
4. Assign subspace from folder path (cheque / facilities_* / credit_ladder / guarantor / supplier_legal / customers_* / general / other).
5. Cluster **within** each subspace on question text only via TF-IDF `char_wb` (3–5) + AgglomerativeClustering (cosine, average).  
   `K = min(12, max(3, int(sqrt(n))))` for `n >= 8`; tiny subspaces → single cluster.
6. Per cluster: medoid representatives (≤5), short Persian summary from medoids, distinctive TF-IDF tokens.
7. Hard pairs: nearest neighbors across subspaces on the global TF-IDF space; target cosine ≥ 0.55, softened to ~0.35–0.45 for sparse priority buckets; cap 40 with quotas on cheque↔facilities_person, facilities_person↔facilities_legal, credit_ladder↔facilities_person, general↔facilities_legal.

## Counts

| Metric | Value |
|---|---|
| Total QA kept | 1078 |
| Incomplete skipped | 24 |
| Subspaces | 9 |
| Clusters (all) | 57 |
| Hard pairs | 40 |

### Per subspace

| Subspace | Questions | Clusters |
|---|---:|---:|
| facilities_person | 451 | 12 |
| facilities_legal | 330 | 12 |
| cheque | 160 | 12 |
| general | 83 | 9 |
| credit_ladder | 22 | 4 |
| guarantor | 13 | 3 |
| supplier_legal | 13 | 3 |
| customers_legal | 3 | 1 |
| customers_person | 3 | 1 |

## Artifacts

- `subspace_counts.json`
- `clusters/<subspace>.json`
- `topic_cards.yaml`
- `hard_pairs.jsonl`
- `recursive_summaries.md`
- `METHOD_NOTES.md`
