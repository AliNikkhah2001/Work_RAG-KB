# KB Manager — Implementation Plan v2

## Overview

This plan covers 6 workstreams to improve the KB management system's chunking quality, retrieval evaluation, and testing infrastructure.

---

## Workstream 1: Fix Broken QA Chunks

### Problem
44 out of 1,782 QA chunks (2.5%) contain only a question with no answer, brief answer, or keywords. These come from Excel rows where the Answer column is empty.

### Solution
Filter out incomplete QA rows during ingestion. Add an `incomplete` flag to metadata for rows with only a question.

### Files to Change
- `kb_manager/chunker/semantic.py` — `_chunk_excel_rows()`: skip rows where no answer field exists, or mark as `incomplete: true` in metadata
- `kb_manager/pipeline/orchestrator.py` — log warning when incomplete rows are encountered

### Decision
**Skip incomplete rows** — a QA chunk without an answer is useless for retrieval. Log the count of skipped rows.

---

## Workstream 2: Structured Chunk Format with Persian Field Names

### Current Format (flat pipe-delimited)
```
Question: مبلغ پرداختی... | BriefAnswer: طی 24 ساعت... | Answer: با توجه به... | Keyword: 24 ساعت
```

### Proposed Format
Content text uses Persian field names with newlines:
```
سوال: مبلغ پرداختی پس از خطای سامانه...
پاسخ کوتاه: طی 24 ساعت آینده انجام خواهد شد.
پاسخ کامل: با توجه به رویه اعتبارسنجی...
کلیدواژه‌ها: 24 ساعت، کسر مبلغ، عدم ثبت تراکنش
```

Metadata stores structured fields for filtering:
```json
{
  "schema": "crm_qa",
  "sheet_name": "چت‌بات-سوال و جواب",
  "fields": {
    "question": "مبلغ پرداختی...",
    "brief_answer": "طی 24 ساعت...",
    "answer": "با توجه به...",
    "keywords": ["24 ساعت", "کسر مبلغ"]
  }
}
```

### Benefits
- **Human-readable** content with clear Persian labels
- **Machine-readable** metadata for filtering and faceted search
- **Search-friendly** content text for BM25/embedding (newlines improve tokenization)
- **Keyword boost**: keywords stored separately for weighted BM25 matching

### Files to Change
- `kb_manager/chunker/semantic.py` — `_chunk_excel_rows()`: format content with Persian field names, store structured fields in metadata
- `kb_manager/parsers/xlsx_parser.py` — pass raw field values through to chunker

### Field Name Mapping (Persian)
| English | Persian | Purpose |
|---------|---------|---------|
| Question | سوال | Primary retrieval target |
| BriefAnswer | پاسخ کوتاه | Quick summary, snippet display |
| Answer | پاسخ کامل | Full answer for generation |
| Keywords | کلیدواژه‌ها | BM25 boost, filtering |
| Model | مدل | Metadata for filtering |
| ReasonCode | کد دلیل | Reason code identifier |
| Explanation | توضیحات | Detailed explanation |

---

## Workstream 3: Hierarchical Chunking Structure

### Architecture

```
Level 0: Document (title, source, schema, total_chunks)
│
├── Level 1: Section/Sheet (=== Sheet: name ===)
│   │
│   ├── Level 2: Atomic Chunks (QA pairs, reason codes)
│   │   ├── content: Human-readable with Persian field names
│   │   ├── metadata.fields: Structured field values
│   │   └── parent_id → Level 3 chunk
│   │
│   └── Level 2: Body Chunks (articles, paragraphs)
│       ├── content: Text with heading path
│       └── parent_id → Level 3 chunk
│
└── Level 3: Parent Context (1024–2048 tokens)
    └── Aggregated Level 2 chunks from same section
```

### Retrieval Flow
1. **Search** on Level 2 child chunks (128–512 tokens) for precision
2. **Retrieve** Level 3 parent chunk (1024–2048 tokens) for LLM context
3. **QA pairs** are always atomic — question + answer together in one Level 2 chunk

### Implementation
- Add `parent_id` column to chunks table (nullable, FK → chunk.id)
- During ingestion, create parent chunks by aggregating Level 2 chunks within each section
- During retrieval, search Level 2, then fetch Level 3 parent for context

### Files to Change
- `kb_manager/models/database.py` — add `parent_id` column to Chunk model
- `kb_manager/chunker/semantic.py` — add `_build_hierarchy()` method
- `kb_manager/web/routes/search.py` — after finding top-K child chunks, fetch parent for context
- Migration: `ALTER TABLE chunks ADD COLUMN parent_id VARCHAR(36) REFERENCES chunks(id)`

---

## Workstream 4: Optimal Chunk Size Configuration

### Research Findings
| Strategy | Size | When to Use |
|----------|------|-------------|
| Fixed small | 64–128 tokens | Fact-based exact match |
| Fixed medium | 256–512 tokens | General RAG sweet spot |
| Hierarchical parent | 1024–2048 tokens | LLM generation context |
| QA pair (atomic) | Variable (50–500) | Always preserve boundaries |
| Paragraph Group | Variable | Best overall (nDCG@5 ≈ 0.459) |

### Recommendation
- **QA pairs:** Atomic (no size normalization) — each row = one chunk
- **Body text:** 256–512 tokens with 10–20% overlap
- **Parent chunks:** 1024–2048 tokens (aggregated children)
- **Overlap:** 50–80 tokens (sentence-boundary aware)

### Current Config
```python
SemanticChunker(max_tokens=512, min_tokens=100, overlap_tokens=50)
```

### Proposed Config
```python
SemanticChunker(
    max_tokens=400,        # slightly smaller for better retrieval
    min_tokens=64,         # allow smaller chunks for QA pairs
    overlap_tokens=50,     # keep current overlap
    parent_max_tokens=1536, # new: parent chunk size
)
```

### Embedding Model Considerations
| Model | Max Tokens | Optimal Chunk |
|-------|-----------|---------------|
| all-MiniLM-L6-v2 | 256 | 128–256 |
| BGE v1.5 | 512 | 256–512 |
| BGE-M3 | 8192 | 512–1024 |
| Jina v3 | 8192 | 128–512 |

**Persian-specific:** Use ParsBERT or multilingual models (BGE-M3, Jina v3) with sentence-boundary preservation.

---

## Workstream 5: Retrieval Evaluation Framework

### Metrics to Implement

#### Phase 1: Offline Retrieval Metrics
| Metric | Formula | Library |
|--------|---------|---------|
| Precision@K | relevant_in_top_k / K | Custom |
| Recall@K | relevant_in_top_k / total_relevant | Custom |
| Hit Rate@K | % queries with ≥1 relevant in top-K | Custom |
| MRR | 1/rank_of_first_relevant | Custom |
| NDCG@K | DCG@K / IDCG@K | `ranx` or custom |
| MAP | Average of precision@r for each relevant doc | `ranx` |

#### Phase 2: RAG-Specific Metrics
| Metric | What it Measures | Framework |
|--------|-----------------|-----------|
| Faithfulness | Answer stays faithful to context | RAGAS |
| Answer Relevance | Answer addresses the question | RAGAS |
| Context Relevance | Retrieved chunks are relevant | RAGAS |
| Context Recall | All necessary context retrieved | RAGAS |

### Evaluation Frameworks

| Framework | Approach | Install |
|-----------|----------|---------|
| **RAGAS** | Reference-free, LLM-as-judge | `pip install ragas` |
| **DeepEval** | pytest-native, CI/CD | `pip install deepeval` |
| **ranx** | Standard IR metrics, fast | `pip install ranx` |

### Recommended Evaluation Pipeline

```
Phase 1: Offline Retrieval Evaluation (ranx)
├── Build qrels (query → relevant doc IDs)
├── Build run (query → ranked doc IDs with scores)
├── Compute: Precision@K, Recall@K, MRR, NDCG@K, MAP
└── Compare: BM25 only vs Semantic only vs Hybrid
    │
Phase 2: End-to-End RAG Evaluation (RAGAS)
├── Build evaluation dataset (query, ground_truth, contexts)
├── Compute: Faithfulness, Answer Relevance, Context Recall
└── Compare: Different chunk sizes, different retrieval strategies
    │
Phase 3: Continuous Monitoring
├── Log retrieval results per query
├── Track hit rate over time
└── Alert on quality degradation
```

---

## Workstream 6: Synthetic Test Data Generation

### Method 1: LLM-Generated Queries from Chunks
```
For each QA chunk:
  1. Feed question + answer to LLM
  2. Prompt: "Generate 3 natural questions that this passage answers"
  3. Gold standard: the original chunk is the expected result
  4. Store: (generated_query, [chunk_id], relevance=1.0)
```

### Method 2: Paraphrase Variations
```
For each existing question:
  1. Generate 5 paraphrases using LLM
  2. Tests robustness to wording variations
  3. Store: (paraphrase, [original_chunk_id], relevance=1.0)
```

### Method 3: Negative Sampling
```
For each query:
  1. Pair with 2-3 irrelevant chunks as negatives
  2. Enables precision/recall calculation
  3. Store: (query, [irrelevant_chunk_ids], relevance=0.0)
```

### Method 4: Cross-Reference Queries
```
Create queries that span multiple chunks:
  1. "What are all the ways to check credit score?" → multiple reason code chunks
  2. Tests multi-chunk retrieval
  3. Store: (query, [chunk_id_1, chunk_id_2, ...], relevance=1.0)
```

### Evaluation Dataset Schema
```json
{
  "query": "چگونه اعتبارسنجی چک انجام می‌شود؟",
  "expected_chunks": ["chunk_id_1", "chunk_id_2"],
  "expected_answer": "...",
  "relevance_scores": {"chunk_id_1": 1.0, "chunk_id_2": 0.8},
  "category": "factual|procedural|comparative",
  "difficulty": "easy|medium|hard"
}
```

### Files to Create
- `kb_manager/evaluation/generator.py` — synthetic data generator
- `kb_manager/evaluation/metrics.py` — IR metrics implementation
- `kb_manager/evaluation/runner.py` — evaluation pipeline runner
- `kb_manager/evaluation/datasets/` — stored evaluation datasets

---

## Implementation Order

| Phase | Workstream | Estimated Effort |
|-------|-----------|-----------------|
| **Phase 1** | Fix broken QA chunks (filter incomplete rows) | Small |
| **Phase 2** | Structured chunk format with Persian field names | Medium |
| **Phase 3** | Hierarchical chunking (parent-child) | Large |
| **Phase 4** | Synthetic test data generator | Medium |
| **Phase 5** | Retrieval evaluation framework (ranx + RAGAS) | Medium |
| **Phase 6** | Optimal chunk size tuning + re-ingestion | Small |

**Total estimated effort:** 6-8 implementation sessions

---

## Key Decisions Needed

1. **Incomplete QA rows:** Skip entirely, or keep with `incomplete: true` flag?
   - **Recommendation:** Skip — useless for retrieval

2. **Parent chunk aggregation:** Per-sheet or per-document?
   - **Recommendation:** Per-sheet (matches natural document structure)

3. **Evaluation framework:** RAGAS only, or RAGAS + ranx?
   - **Recommendation:** ranx for retrieval metrics + RAGAS for RAG quality

4. **Chunk content format:** Newlines or keep pipe-delimited?
   - **Recommendation:** Newlines with Persian field names (better readability + tokenization)

5. **Re-ingestion:** Full rebuild after all changes, or incremental?
   - **Recommendation:** Full rebuild (schema changes require it)
