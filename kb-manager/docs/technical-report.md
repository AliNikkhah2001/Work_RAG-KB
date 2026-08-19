---
layout: default
title: Technical Report
description: Embedding Model, Retrieval Methods & Evaluation Metrics for Persian RAG Knowledge Base
mathjax: true
---

# KB Manager — Technical Report

> **Persian (Farsi) Credit Scoring Knowledge Base** — Dense semantic retrieval with `paraphrase-multilingual-MiniLM-L12-v2`, hybrid BM25 + Dense fusion, and comprehensive IR evaluation.

---

## 1. Embedding Model

| Property | Value |
|----------|-------|
| **Model** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Architecture** | MiniLM (12-layer Transformer encoder, 384 hidden dims, 12 attention heads) |
| **Training Objective** | Contrastive learning on multilingual paraphrase pairs (STS, ParaCrawl, WikiMatrix, etc.) |
| **Languages** | 50+ including Persian (Farsi), Arabic, English |
| **Max Sequence Length** | 128 tokens (truncation) |
| **Output Dimension** | 384 (L2-normalized) |
| **Pooling** | Mean pooling over token embeddings |
| **Model Size** | ~120M parameters, ~470 MB on disk |
| **Inference (CPU)** | ~70 ms/query (batch=1) |
| **Inference (GPU)** | ~5 ms/query |

### Embedding Function

Each chunk \(c\) is mapped to a unit vector on the 384-dimensional hypersphere:

$$
\mathbf{e}_c = \frac{\text{MiniLM}(\text{content}_c)}{\|\text{MiniLM}(\text{content}_c)\|_2} \in \mathbb{R}^{384}, \quad \|\mathbf{e}_c\|_2 = 1
$$

Query embedding is computed online with the same encoder:

$$
\mathbf{e}_q = \frac{\text{MiniLM}(q)}{\|\text{MiniLM}(q)\|_2}
$$

Cosine similarity = dot product (since both normalized):

$$
\text{sim}(q, c) = \mathbf{e}_q^\top \mathbf{e}_c = \cos \theta_{q,c}
$$

---

## 2. Retrieval Pipeline (v3)

### 2.1 BM25 — Lexical Retrieval

Okapi BM25 with Persian-aware tokenization:

$$
\text{score}_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

**Parameters:**
- \(k_1 = 1.5\) (term frequency saturation)
- \(b = 0.75\) (length normalization)
- \(\text{IDF}(t) = \log \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1.0\)
- \(N\) = total chunks, \(\text{df}(t)\) = document frequency of term \(t\)

**Persian tokenization:** Unicode range `[\u0600-\u06FF\u0750-\u077F\u200C\u200D\d]+`, stopword removal, length > 1. Character normalization: ي/ى→ی, ك→ک, ZWNJ→space, alef variants→ا.

---

### 2.2 Dense Semantic Retrieval

Precomputed L2-normalized embedding matrix \(\mathbf{E} \in \mathbb{R}^{N \times 384}\) where \(N\) = number of chunks.

Query embedding \(\mathbf{e}_q\) computed online. Cosine similarities via BLAS matmul:

$$
\mathbf{s} = \mathbf{E} \mathbf{e}_q \in \mathbb{R}^N, \quad \mathbf{e}_q = \text{normalize}(\text{MiniLM}(q))
$$

Top-\(k\) chunks: \(\text{argmax}_k(\mathbf{s})\). Complexity: \(O(N \cdot d)\) with \(d=384\), dominated by optimized matmul (microseconds on CPU).

---

### 2.3 Reciprocal Rank Fusion (RRF)

Two ranked lists (BM25, Dense) fused without score calibration:

$$
\text{RRF}(d) = \sum_{i=1}^{L} \frac{1}{k + \text{rank}_i(d)}
$$

- \(L = 2\) (BM25, Dense)
- \(\text{rank}_i(d)\) = 1-based rank in retriever \(i\)
- \(k = 60\) (empirical constant)

**Advantages:** Rank-based (not score-based), robust to different score distributions, parameter-free beyond \(k\).

---

## 3. Evaluation Metrics (Mathematical Formulas)

### Notation

- \(Q\) = set of test queries, \(|Q|\) = total queries
- \(\mathcal{R}_q \subset \mathcal{C}\) = relevant chunk IDs for query \(q\) (typically \(|\mathcal{R}_q|=1\))
- \(\mathcal{A}_q = [a_1, a_2, \dots, a_K]\) = retrieved chunk IDs at ranks \(1..K\)
- \(\text{rel}_q(c) = \mathbb{1}[c \in \mathcal{R}_q]\) = binary relevance
- \(K\) = cutoff (typically 5)
- \(\text{rank}_q = \min \{ i : a_i \in \mathcal{R}_q \}\) (or \(\infty\) if no hit)

---

### 3.1 Hit Rate @ K (Recall@K)

$$
\text{Hit@}K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\left[ \mathcal{A}_q \cap \mathcal{R}_q \neq \varnothing \right]
$$

Binary per-query: 1 if any relevant chunk in top-\(K\), else 0.

---

### 3.2 Top-1 Accuracy

$$
\text{Top-1} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\left[ a_1 \in \mathcal{R}_q \right]
$$

Strict: relevant chunk must be ranked #1.

---

### 3.3 Mean Reciprocal Rank (MRR)

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
$$

where \(\frac{1}{\infty} = 0\). Rewards early retrieval. Range: \([0, 1]\).

---

### 3.4 Mean Average Precision @ K (MAP@K)

Precision at rank \(i\) for query \(q\):

$$
P_q(i) = \frac{1}{i} \sum_{j=1}^{i} \text{rel}_q(a_j)
$$

Average Precision for query \(q\):

$$
\text{AP}_q = \frac{1}{|\mathcal{R}_q|} \sum_{i=1}^{K} P_q(i) \cdot \text{rel}_q(a_i)
$$

Mean over all queries:

$$
\text{MAP@}K = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}_q
$$

Measures precision across all relevant positions up to \(K\).

---

### 3.5 Normalized Discounted Cumulative Gain @ K (NDCG@K)

Discounted Cumulative Gain:

$$
\text{DCG@}K_q = \sum_{i=1}^{K} \frac{\text{rel}_q(a_i)}{\log_2(i+1)}
$$

Ideal DCG (relevant items ranked first):

$$
\text{IDCG@}K_q = \sum_{i=1}^{\min(K, |\mathcal{R}_q|)} \frac{1}{\log_2(i+1)}
$$

Normalized:

$$
\text{NDCG@}K = \frac{1}{|Q|} \sum_{q \in Q} \frac{\text{DCG@}K_q}{\text{IDCG@}K_q}
$$

Logarithmic discount penalizes lower ranks. Range: \([0, 1]\).

---

### 3.6 Precision @ K

$$
\text{Precision@}K = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathcal{A}_q \cap \mathcal{R}_q|}{K}
$$

With \(|\mathcal{R}_q|=1\), maximum is \(1/K\).

---

### 3.7 Recall @ K

$$
\text{Recall@}K = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathcal{A}_q \cap \mathcal{R}_q|}{|\mathcal{R}_q|}
$$

With \(|\mathcal{R}_q|=1\), equals Hit@K.

---

## 4. Query Format Transformations (Robustness Testing)

Each ground-truth question generates 6 variants targeting different similarity bands (Jaccard token-set similarity after Persian normalization):

| Format | Transformation | Jaccard Band |
|--------|---------------|--------------|
| `verbatim` | Strip trailing ؟ | 0.80–1.00 |
| `paraphrase` | Synonym swap (3) + middle shuffle (2) + ask wrapper | 0.45–0.79 |
| `reworded` | Drop 40% tokens + synonym swap + shuffle (3) + ask wrapper + filler | 0.05–0.44 |
| `keyword_only` | Extract keywords field → split on `[،,;؛\n]` → first 6 tokens | 0.00–0.30 |
| `typo` | Persian typo map (ي→ى, ك→ک, ZWNJ drop) | 0.40–0.85 |
| `conversational` | Formal→informal (می‌شود→میشه, می‌توانم→می‌تونم) + prefix/suffix | 0.05–0.45 |

**Persian normalization for similarity:** ي/ى→ی, ك→ک, ZWNJ→space, alef variants→ا, Arabic-Indic digits→ASCII.

---

## 5. Corpus Fingerprinting (Cache Invalidation)

Dense embedding cache (`data/dense_embeddings.npz`) invalidated when corpus changes:

$$
\text{fp}(\{t_i\}_{i=1}^N) = \text{SHA256}\left( \big\|_{i=1}^N \big( \text{len}(t_i) \parallel t_i \big) \right)
$$

Cache hit iff: stored fingerprint == current fingerprint **AND** chunk count matches. Rebuild triggered automatically on KB change.

---

## 6. Benchmark Results (v3: BM25 + Dense)

| Format | Hit@5 | Top-1 | MRR | Avg Latency |
|--------|-------|-------|-----|-------------|
| verbatim | 100% | 90% | 0.942 | 9.8s (cold) |
| paraphrase | 100% | 95% | 0.975 | 279 ms |
| typo | 100% | 90% | 0.942 | 257 ms |
| conversational | 100% | 85% | 0.917 | 277 ms |
| reworded | 95% | 65% | 0.760 | 265 ms |
| keyword_only | 40% | 10% | 0.188 | 243 ms |
| **Overall** | **89.2%** | **72.5%** | **0.787** | **~1.9 s** |

**v2 → v3 delta:** Top-1 **+9.2%**, MRR **+6.9%**, Latency **−34%** (TF-IDF leg removed).

---

## 7. Latency Breakdown

| Stage | v2 (BM25+TF-IDF) | v3 (BM25+Dense) |
|-------|------------------|-----------------|
| BM25 index build | ~27s (first query) | ~27s (first query) |
| Dense index build | N/A | ~120s (first run, cached) |
| Query: BM25 search | ~10 ms | ~10 ms |
| Query: Dense encode | N/A | ~70 ms |
| Query: Dense matmul | N/A | <5 ms |
| Query: TF-IDF cosine | ~1-2 s (O(N)) | **Removed** |
| **Total (warm)** | **~2.8 s** | **~1.9 s** |

---

## 8. Key Findings

1. **Dense embeddings** are the single biggest win for semantic queries (paraphrase, reworded, typo, conversational) — Top-1 jumped from ~63% → 72%.
2. **Keyword_only** weakness (40% hit) is a **test-data artifact**: generator extracts merged "keywords + model" header (`کلیدواژه‌ها: بروزرسانی، بازپرداخت، وام، گزارش اعتباری مدل: حقیقی و حقوقی...`). Not a retrieval gap.
3. **Latency** dropped ~34% by removing O(N) TF-IDF full scan; dense encode + matmul is fast.
4. **Embeddings cached to disk** keyed by corpus fingerprint — rebuild only on KB change.

---

## 9. Running the Benchmark

```bash
# From kb-manager/
KB_DB_URL="sqlite+aiosqlite:///data/kb_test.db" python run_benchmark.py test_questions.json 5
```

Results written to `data/benchmark_results.json`, `data/ir_metrics.json`, and plots to `data/plots/`.

---

## References

- Robertson, S., Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in IR.
- Craswell, N. et al. (2020). *Overview of the TREC 2020 Deep Learning Track*.
- Thakur, N. et al. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models*.
- Reimers, N., Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
- Lin, J. et al. (2022). *Rank Fusion: A Survey of Methods and Applications*. ACM Computing Surveys.

---

*KB Manager — ICS Credit Scoring Knowledge Base*