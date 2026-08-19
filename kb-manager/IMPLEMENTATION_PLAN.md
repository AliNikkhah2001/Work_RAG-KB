# KB Manager - Retrieval System v4 Implementation Plan

## Overview
Complete overhaul of the retrieval system with 7 improvement methods, synthetic data generation pipeline, and comprehensive documentation.

---

## Phase 1: Documentation & Planning

### 1.1 Persian-Specific Resources Documentation
- [ ] Create `PERSIAN_RESOURCES.md` with all available Persian NLP resources
- [ ] Document FaMTEB datasets with HuggingFace links
- [ ] Document MIRACL-Fa, BEIR-Fa, NQ-Fa datasets
- [ ] Document Persian embedding models (ParsBERT, FaBERT, ParsGPT, etc.)
- [ ] Document Persian tokenizers (Hazm, Parsivar)
- [ ] Add resource comparison table

### 1.2 Implementation Plan Documentation
- [ ] Create `IMPLEMENTATION_PLAN.md` (this file)
- [ ] Define feature branch strategy
- [ ] Document technical decisions and trade-offs

---

## Phase 2: Feature Branch Creation

### Branch Strategy
```
main
├── feat/cross-encoder-reranker
├── feat/char-ngram-bm25
├── feat/hyde-pseudo-doc
├── feat/multi-query-rewriting
├── feat/contextual-retrieval
├── feat/benchmark-famteb
├── feat/synthetic-generation
└── feat/readme-overhaul
```

### Branch Creation Tasks
- [ ] Create `feat/cross-encoder-reranker` branch
- [ ] Create `feat/char-ngram-bm25` branch
- [ ] Create `feat/hyde-pseudo-doc` branch
- [ ] Create `feat/multi-query-rewriting` branch
- [ ] Create `feat/contextual-retrieval` branch
- [ ] Create `feat/benchmark-famteb` branch
- [ ] Create `feat/synthetic-generation` branch
- [ ] Create `feat/readme-overhaul` branch

---

## Phase 3: Core Retrieval Improvements

### 3.1 Cross-Encoder Reranker (`feat/cross-encoder-reranker`)
- [ ] Create `kb_manager/reranker.py` with `CrossEncoderReranker` class
- [ ] Implement batch pairwise scoring
- [ ] Add model loading with caching (mDeBERTa-v3-base-xsmall)
- [ ] Add rerank method: query + candidates → top-k
- [ ] Add configuration options (model_name, batch_size, device)
- [ ] Write unit tests
- [ ] Integrate into search pipeline (after RRF, before final top-k)

### 3.2 Character n-gram BM25 + Persian Normalization (`feat/char-ngram-bm25`)
- [ ] Modify `_tokenize()` in `search.py` to add Persian char 3-grams
- [ ] Add Persian character normalization (ي→ی, ك→ک, ZWNJ handling)
- [ ] Add Arabic-Indic digit normalization
- [ ] Combine word tokens + character n-grams
- [ ] Update BM25 indexing to handle mixed tokens
- [ ] Test with typo benchmark format

### 3.3 HyDE Pseudo-Document Generation (`feat/hyde-pseudo-doc`)
- [ ] Create `kb_manager/query_reform.py` with `HyDEGenerator` class
- [ ] Implement prompt template for hypothetical document generation
- [ ] Add LLM integration (use existing embedder/llm config)
- [ ] Add caching for generated pseudo-documents
- [ ] Integrate into search: embed pseudo-doc + combine with query embedding

### 3.4 Multi-Query Rewriting + RRF (`feat/multi-query-rewriting`)
- [ ] Add `MultiQueryGenerator` class to `query_reform.py`
- [ ] Implement beam search (beam=5-10) for diverse rewrites
- [ ] Add query type prompts (verbatim, paraphrase, conversational, etc.)
- [ ] Implement RRF fusion for multiple query results
- [ ] Add configurable weights per query type

### 3.5 Contextual Retrieval (`feat/contextual-retrieval`)
- [ ] Modify `dense.py` to prepend chunk context to embedding text
- [ ] Context format: `title + heading_path + content`
- [ ] Rebuild dense index with contextual embeddings
- [ ] Test recall improvement on benchmark

### 3.6 Query Reformulation Pipeline Integration
- [ ] Update `search_knowledge_base()` in `search.py`
- [ ] Pipeline: BM25+ngram → Dense → RRF → Rerank → top-k
- [ ] Add performance monitoring/logging
- [ ] Ensure backward compatibility

---

## Phase 4: Benchmark & Evaluation

### 4.1 FaMTEB Dataset Integration (`feat/benchmark-famteb`)
- [ ] Add dataset loading for: SynPerQARetrieval, SynPerChatbotRAGTopicsRetrieval, BEIR-Fa, MIRACL-Fa, NQ-Fa
- [ ] Extend `run_benchmark.py` to support multiple dataset formats
- [ ] Add MAP@K, Recall@100 metrics
- [ ] Add bootstrap confidence intervals
- [ ] Add statistical significance testing

### 4.2 Evaluation Metrics Enhancement
- [ ] Implement all 7 metrics with proper LaTeX formulas
- [ ] Add per-format breakdown
- [ ] Add latency profiling (P50, P95, P99)
- [ ] Generate comparison plots (v2 vs v3 vs v4)

---

## Phase 5: Synthetic Data Generation

### 5.1 Pipeline Infrastructure (`feat/synthetic-generation`)
- [ ] Create `synthetic_generation/` directory structure
- [ ] Create `config.yaml` with generation parameters
- [ ] Create prompt templates (QA, conversational, typo, keyword)
- [ ] Implement `QAGenerator`, `ConversationalGenerator` classes
- [ ] Implement `SyntheticValidator` with LLM-as-judge

### 5.2 Gemma 30B Deployment Guide
- [ ] Create `docs/synthetic-generation.md` with:
  - Hardware requirements (2×24GB GPU)
  - vLLM tensor parallel setup
  - 4-bit AWQ quantization config
  - Generation command examples
  - Quality validation pipeline

### 5.3 Quality Validation
- [ ] Semantic similarity filtering (query ↔ target chunk)
- [ ] Conversation retrieval success validation
- [ ] Human audit sampling (100 samples)
- [ ] Diversity metrics (query type coverage)

---

## Phase 6: Documentation Overhaul

### 6.1 README Overhaul (`feat/readme-overhaul`)
- [ ] Executive Summary table (Retrieval System v4)
- [ ] Detailed Retrieval Methods (7 sections with math + code)
- [ ] Evaluation Methodology (7 metrics with LaTeX)
- [ ] Query Format Taxonomy (6 types + impact)
- [ ] Persian Benchmark Integration
- [ ] Synthetic Data Generation Guide
- [ ] Performance Comparison Tables

### 6.2 Technical Report Update
- [ ] Sync `docs/technical-report.md` with implementation
- [ ] Add new formulas for cross-encoder, RRF, HyDE
- [ ] Update benchmark results tables

---

## Phase 7: Integration & Testing

### 7.1 End-to-End Testing
- [ ] Run full benchmark on `test_questions.json`
- [ ] Target: Hit@5 > 95%, Top-1 > 80%, MRR > 0.85
- [ ] Run FaMTEB evaluation (BEIR-Fa, MIRACL-Fa)
- [ ] Latency profiling (P50 < 200ms, P99 < 500ms)

### 7.2 Merge Strategy
- [ ] Create PRs for each feature branch
- [ ] Code review checklist
- [ ] Merge to main after all pass

---

## Persian-Specific Resources Reference

| Resource | Type | Link | Description |
|----------|------|------|-------------|
| FaMTEB | Benchmark | https://huggingface.co/collections/MCINext/famteb-66b... | 63 datasets, 7 tasks |
| SynPerQARetrieval | Dataset | https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval | Synthetic QA (GPT-4o-mini) |
| SynPerChatbotRAGTopicsRetrieval | Dataset | https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval | Chatbot conversations + topics |
| SynPerChatbotRAGSumSRetrieval | Dataset | https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval | Summary retrieval |
| BEIR-Fa (NQ-Fa) | Dataset | https://huggingface.co/datasets/MCINext/nq-fa | Translated Natural Questions |
| MIRACL-Fa | Dataset | https://huggingface.co/datasets/miracl/miracl | Multilingual IR (Persian subset) |
| ParsBERT | Model | https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased | Persian BERT |
| FaBERT | Model | https://huggingface.co/m3hrdadfa/fa-bert-base | Persian BERT variant |
| ParsGPT | Model | https://huggingface.co/HooshvareLab/gpt2-fa | Persian GPT-2 |
| Hazm | Library | https://github.com/roshan-research/hazm | Persian NLP toolkit |
| Parsivar | Library | https://github.com/ipsina/parsivar | Persian text processing |

---

## Technical Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cross-encoder model | mDeBERTa-v3-base-xsmall | Multilingual, 27M params, fast inference |
| Gemma quantization | 4-bit AWQ via vLLM | Fits 2×24GB GPU with tensor parallel |
| Inference engine | vLLM | Tensor parallelism, high throughput |
| Synthetic sample target | 50K QA + 15K conversations | Adequate coverage, manageable generation time |
| Validation method | LLM-as-judge (Gemma) + 100 human | Cost-effective quality assurance |
| FaMTEB integration | Supplement existing benchmark | Preserve custom query formats |
| Reranker training | Off-the-shelf first | Faster iteration, fine-tune later |

---

## Progress Tracking

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| Phase 1: Documentation | 🔄 In Progress | - | - |
| Phase 2: Branch Creation | ⏳ Pending | - | - |
| Phase 3: Core Retrieval | ⏳ Pending | - | - |
| Phase 4: Benchmark | ⏳ Pending | - | - |
| Phase 5: Synthetic Gen | ⏳ Pending | - | - |
| Phase 6: Docs Overhaul | ⏳ Pending | - | - |
| Phase 7: Integration | ⏳ Pending | - | - |

---

## Next Immediate Actions

1. [ ] Create `PERSIAN_RESOURCES.md` with all resources
2. [ ] Create feature branches for each sub-part
3. [ ] Start with `feat/cross-encoder-reranker` implementation
4. [ ] Run initial benchmark to establish baseline