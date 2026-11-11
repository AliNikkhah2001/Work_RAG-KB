# Method notes — subspace distill

## What worked

- Parser `get_parser` + schema `crm_qa` cleanly recovered 1078 Q/A rows; 24 blanks match the known blank/corrupt CSV count (~24).
- Path-based subspace routing separated cheque vs facilities_person vs facilities_legal without DB taxonomy.
- TF-IDF `char_wb` (3–5 grams) is robust for Persian (no tokenizer/stemmer required) and fast on ~1k questions.
- Agglomerative clustering with cosine + average linkage gave stable small-K topic groups; medoids via centroid cosine are readable.
- Hard-pair mining on a **global** TF-IDF space surfaces cross-subspace confusions (cheque↔facilities, person↔legal, ladder↔facilities, general↔legal).

## Limitations

- Summaries are heuristic (medoid + keyword themes), not editor-curated product copy.
- `اصلی` and `جایگزین` facilities banks both included → duplicate questions inflate sizes/clusters for `facilities_person`.
- Clustering uses char_wb n-grams; **displayed** distinctive tokens are word/bi-gram TF-IDF (contrastive) for readability.
- Tiny subspaces (`customers_*`, sometimes others) are a single dump cluster — limited topic structure.
- Hard-pair threshold is empirical; near-duplicates across domains may dominate over subtle confusions.
- sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) was available but **not required**; TF-IDF kept the run offline-fast and deterministic.

## Alternatives

- Dense multilingual embeddings + HDBSCAN/Agglomerative for softer Persian paraphrases.
- Dedup questions within subspace (or prefer `اصلی` over `جایگزین`) before clustering.
- Word n-grams with a Persian normalizer (`hazm`/`preprocessor`) for cleaner distinctive terms.
- Supervised subspace classifier trained on these labels for runtime sticky routing.
- Human pass over `topic_cards.yaml` in_scope / sticky_phrases before wiring into orchestrator.

## Hard-pair rebalance

Important confusion buckets were quota-filled first (`cheque↔facilities_person`, `facilities_person↔facilities_legal`, `credit_ladder↔facilities_person`, `general↔facilities_legal`), capping exact cross-domain duplicate questions so paraphrastic near-misses can appear. Threshold softened to ~0.42–0.50 when a priority bucket was sparse.
