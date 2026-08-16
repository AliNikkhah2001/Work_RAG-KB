# Retrieval Evaluation Methods, Frameworks & Metrics

## 1. Standard IR Metrics

### 1.1 Precision@K
- **What it measures**: Fraction of top-K retrieved documents that are relevant
- **Formula**: `Precision@K = |relevant ∩ top-K| / K`
- **When to use**: When you want to measure noise in results (how many retrieved items are actually useful)
- **Limitation**: Doesn't consider rank order; doesn't account for relevant documents not retrieved
- **Python**: `pytrec_eval`, `ranx`, `sentence-transformers` evaluation

### 1.2 Recall@K
- **What it measures**: Fraction of all relevant documents that were retrieved in top-K
- **Formula**: `Recall@K = |relevant ∩ top-K| / |all relevant|`
- **When to use**: When completeness matters (finding ALL relevant documents)
- **Limitation**: Not rank-aware; a relevant doc at position 1 counts same as position K
- **Python**: `pytrec_eval`, `ranx`

### 1.3 Hit Rate@K
- **What it measures**: Whether at least one relevant document appears in top-K
- **Formula**: `Hit@K = 1 if any relevant doc in top-K, else 0` (averaged over queries)
- **When to use**: Simple "did we find anything?" check; good for single-answer QA
- **Python**: Custom implementation or `ranx`

### 1.4 Mean Reciprocal Rank (MRR@K)
- **What it measures**: How high the FIRST relevant document appears in results
- **Formula**: `MRR = (1/|Q|) × Σ(1/rank_i)` where rank_i is position of first relevant doc
- **Range**: 0 to 1 (1.0 = relevant doc always at rank 1)
- **When to use**: When you need ONE good answer (e.g., "I'm Feeling Lucky" search, single-answer QA)
- **Limitation**: Only considers first relevant result; ignores subsequent relevant docs
- **Python**: `pytrec_eval`, `ranx`, `scikit-learn` (custom)

### 1.5 Mean Average Precision (MAP@K)
- **What it measures**: Average precision across all recall levels, averaged over queries
- **Formula**:
  ```
  AP@K = (1/min(|relevant|, K)) × Σ_{k=1}^{K} P@k × rel(k)
  MAP = (1/|Q|) × Σ AP@K
  ```
  where `rel(k)` is 1 if document at rank k is relevant, 0 otherwise
- **Range**: 0 to 1
- **When to use**: When you need to measure both precision and recall across the entire ranking; good for multi-answer scenarios
- **Limitation**: Binary relevance only (relevant/not relevant)
- **Python**: `pytrec_eval`, `ranx`, `scikit-learn`

### 1.6 Normalized Discounted Cumulative Gain (NDCG@K)
- **What it measures**: Quality of ranking considering graded relevance and position discounting
- **Formula**:
  ```
  DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log2(i + 1)
  NDCG@K = DCG@K / IDCG@K
  ```
  where IDCG@K is the ideal DCG@K (perfect ranking)
- **Range**: 0 to 1 (1.0 = perfect ranking)
- **When to use**: When relevance is graded (not just binary); most popular metric for web search evaluation
- **Advantage**: Handles multi-level relevance; position-aware; easily interpretable
- **Python**: `pytrec_eval`, `ranx`, `scikit-learn` (custom)

### 1.7 Precision and Recall (Set-based)
- **Precision**: `|relevant ∩ retrieved| / |retrieved|`
- **Recall**: `|relevant ∩ retrieved| / |relevant|`
- **F1**: `2 × (precision × recall) / (precision + recall)`
- **When to use**: Basic diagnostics; not suitable for ranked results

### Metric Comparison Summary

| Metric | Rank-aware | Graded relevance | Best for |
|--------|-----------|-----------------|----------|
| Precision@K | No | No | Noise measurement |
| Recall@K | No | No | Completeness |
| Hit Rate@K | No | No | Binary success |
| MRR@K | Yes | No | Single-answer QA |
| MAP@K | Yes | No | Multi-answer retrieval |
| NDCG@K | Yes | Yes | Web search, complex relevance |

---

## 2. RAG-Specific Metrics

### 2.1 Faithfulness (Groundedness)
- **What it measures**: Whether every claim in the generated answer is supported by the retrieved context
- **Formula**: `Faithfulness = (claims supported by context) / (total claims in answer)`
- **How it works**:
  1. Extract individual claims from the answer using LLM
  2. For each claim, check if it can be inferred from the retrieved context
  3. Compute ratio of supported claims
- **Range**: 0 to 1 (1.0 = all claims grounded in context)
- **When to use**: Detecting hallucinations; ensuring LLM doesn't add unsupported information
- **Python**: `ragas`, `deepeval`, `trulens`

### 2.2 Answer Relevance
- **What it measures**: Whether the generated answer actually addresses the question
- **How it works** (RAGAS approach):
  1. Generate N hypothetical questions from the answer
  2. Compute cosine similarity between original question and generated questions
  3. Average similarity = answer relevance score
- **Range**: 0 to 1 (1.0 = answer perfectly addresses question)
- **When to use**: Detecting answers that are factually correct but miss the point
- **Python**: `ragas`, `deepeval`

### 2.3 Context Relevance (Context Precision)
- **What it measures**: Whether retrieved context chunks are focused and relevant to the query
- **Formula**: `Context Precision = |relevant chunks| / |total retrieved chunks|`
- **Weighted version**: Rewards relevant chunks appearing at top of ranked list
- **When to use**: Measuring retrieval noise; determining if too many irrelevant chunks dilute context
- **Python**: `ragas`, `deepeval`, `trulens`

### 2.4 Context Recall
- **What it measures**: Whether ALL information needed to answer the question is present in retrieved context
- **Formula**:
  1. Break ground truth answer into individual claims
  2. Check each claim against retrieved context
  `Context Recall = (claims attributable to context) / (total claims in ground truth)`
- **Requires**: Ground truth answer
- **When to use**: Detecting retrieval misses; measuring if retriever found all needed information
- **Python**: `ragas`, `deepeval`

### 2.5 RAG Triad (TruLens)
Three metrics evaluated together:
1. **Context Relevance**: Are retrieved chunks relevant to the query?
2. **Groundedness (Faithfulness)**: Is every claim in the answer supported by context?
3. **Answer Relevance**: Does the answer address the question?

High scores on all three = confidence the app is hallucination-free.

### 2.6 Claim-Level F1 (RAGChecker)
- **What it measures**: Overall correctness at claim level
- **Formula**:
  ```
  Precision = |correct response claims| / |total response claims|
  Recall = |correct ground truth claims covered| / |total ground truth claims|
  F1 = harmonic mean of precision and recall
  ```
- **When to use**: Fine-grained evaluation of long-form answers

---

## 3. Evaluation Frameworks

### 3.1 RAGAS (Retrieval Augmented Generation Assessment)
- **GitHub**: https://github.com/explodinggradients/ragas
- **Paper**: Es et al., EACL 2024
- **Key Features**:
  - Reference-free evaluation (no ground truth needed for core metrics)
  - Three core metrics: Faithfulness, Answer Relevance, Context Relevance
  - Additional metrics: Context Precision, Context Recall
  - Uses LLM-as-judge paradigm
  - Integrates with LangChain and LlamaIndex
- **Metrics**:
  - `faithfulness`: Claims supported by context / total claims
  - `answer_relevancy`: Similarity of generated questions to original
  - `context_precision`: Relevant chunks in top positions
  - `context_recall`: Claims in ground truth attributable to context
- **Installation**: `pip install ragas`
- **Best for**: Quick RAG evaluation without human annotation

### 3.2 DeepEval
- **GitHub**: https://github.com/confident-ai/deepeval
- **Key Features**:
  - Pytest-native evaluation framework
  - RAG metrics + Agent metrics + Custom metrics
  - Runs locally (no external API calls required for NLP metrics)
  - G-Eval for custom criteria
  - CI/CD integration
- **RAG Metrics**:
  - `AnswerRelevancyMetric`
  - `FaithfulnessMetric`
  - `ContextualRecallMetric` (requires expected_output)
  - `ContextualPrecisionMetric` (requires expected_output)
  - `ContextualRelevancyMetric`
  - `RagAnswerMetric` (average of above)
- **Installation**: `pip install deepeval`
- **Best for**: Teams wanting pytest-style testing for LLM apps

### 3.3 TruLens
- **GitHub**: https://github.com/truera/trulens
- **Key Features**:
  - RAG Triad evaluation (Context Relevance, Groundedness, Answer Relevance)
  - Stack-agnostic instrumentation
  - Agentic evaluations (7 evaluators for agent behavior)
  - Feedback functions with multiple LLM providers
- **RAG Metrics**:
  - `Groundedness`: Response supported by context
  - `ContextRelevance`: Chunks relevant to question
  - `PromptResponseRelevance`: Response relevant to prompt
  - `Answerability`: Whether question is answerable from source
  - `Comprehensiveness`: Summary coverage of source key points
- **Installation**: `pip install trulens`
- **Best for**: Comprehensive RAG evaluation with multiple feedback providers

### 3.4 ARES (Automated RAG Evaluation System)
- **Paper**: Saad-Falcon et al., NAACL 2024
- **Key Features**:
  - Fine-tunes lightweight DeBERTa judges on synthetic data
  - Prediction-Powered Inference (PPI) for confidence intervals
  - Evaluates: Context Relevance, Answer Faithfulness, Answer Relevance
  - Only ~150 human annotations needed
- **Best for**: When you need calibrated confidence intervals

### 3.5 RAGChecker
- **Paper**: Ru et al., 2024
- **Key Features**:
  - Claim-level entailment checking
  - Fine-grained metrics for retriever and generator separately
  - Retriever metrics: claim recall, chunk-level precision
  - Generator metrics: claim precision, claim recall, F1
- **Best for**: Diagnosing whether errors come from retrieval or generation

### 3.6 BERGEN
- **GitHub**: https://github.com/naver/bergen
- **Key Features**:
  - End-to-end RAG benchmarking library
  - Standardized RAG experiments
  - Multiple retriever, reranker, LLM comparisons
  - Surface-based and LLM-based metrics
- **Best for**: Reproducible RAG research and benchmarking

### 3.7 Open-RAG-Eval (Vectara)
- **GitHub**: https://github.com/vectara/open-rag-eval
- **Key Features**:
  - TREC-RAG benchmark metrics
  - UMBRELA (retrieval quality) - no ground truth needed
  - AutoNuggetizer (answer quality) - no ground truth needed
  - Golden answer evaluation when reference available
  - Connectors for Vectara, LlamaIndex, LangChain
- **Best for**: Production RAG evaluation without ground truth

### Framework Comparison

| Framework | Reference-free | Metrics | CI/CD | Local models |
|-----------|---------------|---------|-------|--------------|
| RAGAS | Yes (core) | Faithfulness, Relevance, Context | No | No (uses OpenAI) |
| DeepEval | Partial | RAG + Agent + Custom | Yes (pytest) | Yes |
| TruLens | Yes | RAG Triad + Agentic | No | No |
| ARES | Partial | Faithfulness, Relevance, Context | No | Yes (DeBERTa) |
| RAGChecker | No (needs GT) | Claim-level F1 | No | Yes |
| BERGEN | Partial | Multiple | No | Yes |
| Open-RAG-Eval | Yes (core) | UMBRELA, AutoNuggetizer | No | Yes |

---

## 4. Synthetic Data Generation Methods

### 4.1 LLM-Generated Queries from Documents
- **Method**: Sample passages → Generate queries using LLM → Document is ground truth for generated query
- **Models**: T5-based (BeIR query-gen), GPT-4, FLAN-T5
- **Quality**: Queries may not reflect natural user language
- **Reference**: Rahmani et al., SIGIR 2024

### 4.2 LLM-Generated Relevance Judgments
- **Method**: Use LLM to judge document-query relevance instead of human annotators
- **Quality**: Cohen's κ ~0.24-0.64 depending on prompt design
- **Key finding**: LLM-based judgments produce similar system rankings to human judgments (Kendall's τ ~0.86)
- **Reference**: Thomas et al., Faggioli et al.

### 4.3 TREC Pooling Approach
- **Method**:
  1. Run multiple retrieval systems on same queries
  2. Take top-K from each system
  3. Deduplicate into a pool
  4. Have humans judge only the pooled documents
- **Advantage**: Focuses human effort on most likely relevant documents
- **Reference**: Microsoft ISE Blog, 2025

### 4.4 GPT-4 Assisted Ranking
- **Method**:
  1. Use GPT-4 to score document relevance (0-5 scale)
  2. Calibrate with ~30% human-annotated subset
  3. Set threshold to capture ~90% of relevant documents
  4. Only experts review documents above threshold
- **Advantage**: Reduces 10,500 documents to ~246 for expert review
- **Reference**: Microsoft ISE Blog, 2025

### 4.5 Synthetic Test Collections (GenTREC)
- **Method**:
  1. Use LLM to generate documents from TREC topics
  2. Generated document is relevant to its generating prompt
  3. Generate non-relevant documents for distractors
- **Result**: 96,196 documents, 300 topics, ~$126 cost
- **Finding**: System ranking compatible with traditional TREC evaluations
- **Reference**: Türkmen et al., 2025

### 4.6 Subset Sampling (SPEAR)
- **Method**:
  1. Aggregate all chunks recalled by all retrievers being compared
  2. Use this as a "pseudo ground truth" subset
  3. Compute precision/recall relative to this subset
- **Advantage**: Enables relative comparison of retrievers at low cost
- **Reference**: SPEAR paper, 2025

### 4.7 Implicit Signals from User Behavior
- **Signals**: Clicks, add-to-carts, session duration, copy-paste
- **Method**: Aggregate interaction data; high positive interaction rate = relevant
- **Challenges**: Noisy, sparse, presentation bias
- **Best for**: Relative comparisons (A/B testing), not absolute ground truth

### 4.8 Human Annotation Best Practices
1. **Query Selection**: Cover head, torso, and tail queries
2. **Candidate Generation**: Pool from multiple retrieval methods
3. **Guideline Definition**: Clear relevance criteria with examples
4. **Annotation**: Multiple independent annotators
5. **Adjudication**: Resolve disagreements
6. **Inter-Annotator Agreement**: Measure Cohen's κ or Krippendorff's α
7. **Tools**: Labelbox, Amazon SageMaker Ground Truth, doccano

---

## 5. Cranfield Paradigm (Standard IR Evaluation Methodology)

### Core Components
1. **Document Collection**: Set of documents to search
2. **Topics/Queries**: Information needs (questions)
3. **Relevance Judgments**: Human assessments of document-query relevance
4. **Evaluation Measures**: Metrics computed from judgments

### Process
1. Create test collection (documents + queries + relevance judgments)
2. Run retrieval systems on queries
3. Compare system output against relevance judgments
4. Compute metrics (Precision, Recall, MAP, NDCG, etc.)

### Modern Extensions
- **TREC evaluations**: Large-scale evaluation campaigns
- **Pooling**: Focus human effort on top-K results from multiple systems
- **Graded relevance**: Moving beyond binary relevant/not-relevant
- **Online evaluation**: A/B testing, interleaving

### Key Insight
The ranking task is a "carefully calibrated level of abstraction" - sufficiently faithful to real tasks to be informative, but sufficiently abstract to be broadly applicable and feasible.

---

## 6. Hybrid Search Evaluation

### Fusion Quality Metrics
1. **NDCG comparison**: Fused ranking vs. BM25-only vs. vector-only
2. **Kendall's Tau / Spearman correlation**: Consistency between fused and input rankings
3. **Improvement percentage**: `(fused_NDCG - max(bm25_NDCG, vector_NDCG)) / max(...)`

### Key Findings
- BM25 excels at exact-match queries; vector search at semantic queries
- Fusion should beat both inputs; if not, signals are too similar or contradictory
- Alpha tuning is critical (not symmetric; often α=0.2-0.3, not 0.5)
- Reciprocal Rank Fusion (RRF) is "plug and play" - no calibration needed
- Weighted linear combination can outperform RRF with ~40 annotated queries

### Fusion Methods
1. **Reciprocal Rank Fusion (RRF)**: `score(d) = Σ 1/(k + rank_i(d))`
2. **Weighted Sum of Scores**: Normalized scores combined with weight α
3. **Distribution-Based Score Fusion (DBSF)**
4. **Relative Score Fusion (RSF)**

### Python Libraries
- `rank_bm25`: BM25 implementation
- `ranx`: Comprehensive IR evaluation
- `pytrec_eval`: TREC-compatible evaluation
- `mteb`: Massive Text Embedding Benchmark with hybrid support

---

## 7. Recommended Evaluation Pipeline

### Phase 1: Retrieval Evaluation (Offline)
```
1. Create ground truth dataset:
   - 50-100+ representative queries
   - Relevance judgments (human or LLM-assisted)
   - Format: {query_id: {doc_id: relevance_score}}

2. Run retrieval systems:
   - BM25 baseline
   - Vector search baseline
   - Hybrid search
   - Your production system

3. Compute metrics:
   - NDCG@10 (primary)
   - MRR@10 (single-answer scenarios)
   - Recall@100 (completeness)
   - MAP (multi-answer scenarios)
   - Hit Rate@K (simple success check)
```

### Phase 2: End-to-End RAG Evaluation
```
1. Using RAGAS (reference-free):
   - faithfulness: Catch hallucinations
   - answer_relevancy: Catch off-topic answers
   - context_precision: Measure retrieval noise
   - context_recall: Measure retrieval completeness (needs GT)

2. Using DeepEval (CI/CD integration):
   - FaithfulnessMetric
   - AnswerRelevancyMetric
   - ContextualRelevancyMetric

3. Segment by query type:
   - Simple factual
   - Multi-hop reasoning
   - Comparison queries
   - Temporal queries
```

### Phase 3: Continuous Monitoring
```
1. Sample production queries
2. Run RAGAS/DeepEval metrics
3. Track metric trends over time
4. Alert on metric degradation
5. Periodically refresh ground truth dataset
```

### Quick Start Commands
```bash
# Install core libraries
pip install ragas deepeval pytrec_eval ranx

# RAGAS evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

# DeepEval evaluation
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is AI?",
    actual_output="AI is...",
    retrieval_context=["chunk1", "chunk2"]
)
evaluate(test_cases=[test_case], metrics=[FaithfulnessMetric(), AnswerRelevancyMetric()])
```

---

## 8. Key References

1. **Cleverdon (1962)**: Original Cranfield experiments
2. **Järvelin & Kekäläinen (2002)**: NDCG definition
3. **Es et al. (2024)**: RAGAS framework (EACL 2024)
4. **Saad-Falcon et al. (2024)**: ARES framework (NAACL 2024)
5. **Ru et al. (2024)**: RAGChecker
6. **Rahmani et al. (2024)**: Synthetic Test Collections (SIGIR 2024)
7. **Salemi & Zamani (2024)**: Evaluating Retrieval Quality in RAG
8. **BERGEN (2024)**: RAG Benchmarking Library
9. **Open-RAG-Eval (2024)**: Vectara's evaluation toolkit
10. **MIRAGE (2025)**: Metric-intensive RAG benchmark
