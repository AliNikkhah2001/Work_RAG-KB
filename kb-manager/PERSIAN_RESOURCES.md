# Persian NLP Resources for KB Manager

Comprehensive catalog of Persian (Farsi) datasets, models, and tools relevant to the KB Manager retrieval system.

---

## Benchmark Suites

### FaMTEB (Farsi Massive Text Embedding Benchmark)
- **Link**: https://huggingface.co/collections/MCINext/famteb-66b...
- **Paper**: https://arxiv.org/abs/2502.11571
- **Description**: 63 datasets across 7 tasks (classification, clustering, pair classification, reranking, retrieval, summary retrieval, STS)
- **Tasks relevant to KB Manager**:
  - **Retrieval**: 15+ datasets including synthetic and translated
  - **Reranking**: MIRACL-Reranking, WikipediaRerankingMultilingual
  - **Pair Classification**: Paraphrase detection, entailment
- **Models evaluated**: 15 Persian/multilingual models (Jina, BGE-m3, ParsBERT, etc.)

### BEIR-Fa (BEIR Persian)
- **Link**: https://huggingface.co/datasets/MCINext/nq-fa (part of BEIR-Fa)
- **Source**: Google Translate of BEIR benchmark (Thakur et al., 2021)
- **Datasets**: NQ-Fa, HotpotQA-Fa, FiQA-Fa, ArguAna-Fa, Touche2020-Fa, etc.
- **Use case**: Zero-shot retrieval evaluation on translated English benchmarks

### MIRACL-Fa (Multilingual Information Retrieval Across Languages)
- **Link**: https://huggingface.co/datasets/miracl/miracl (Persian subset: `miracl/miracl` with `lang="fa"`)
- **Paper**: Zhang et al., 2023
- **Description**: Wikipedia-based retrieval with human-generated queries in 18 languages including Persian
- **Size**: ~1000 queries for Persian, Wikipedia passages as corpus
- **Metrics**: nDCG@10, Recall@100

---

## Synthetic Datasets (FaMTEB)

### SynPerQARetrieval
- **Link**: https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval
- **Generation**: GPT-4o-mini on curated Persian web content
- **Format**: Question-answer pairs for retrieval evaluation
- **Human validation**: 98% accuracy
- **Use case**: Evaluate embedding models on Persian QA retrieval

### SynPerChatbotRAGTopicsRetrieval
- **Link**: https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval
- **Generation**: GPT-4o-mini, 175 topics × 19 tone variations
- **Format**: Multi-turn conversations + topic labels
- **Human validation**: 93% accuracy
- **Use case**: RAG chatbot topic retrieval evaluation

### SynPerChatbotRAGSumSRetrieval
- **Link**: https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval
- **Format**: Conversation-summary pairs for summary retrieval task
- **Human validation**: 99% accuracy
- **Use case**: Conversation summarization retrieval

### SynPerChatbotTopicsRetrieval
- **Link**: https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval
- **Format**: User queries → topic-relevant chatbot responses
- **Use case**: Chatbot response retrieval

### SynPerQAPC (Query Paraphrase Classification)
- **Link**: https://huggingface.co/datasets/MCINext/synthetic-persian-qa-pair-classification
- **Format**: Question pairs with paraphrase labels
- **Use case**: Query reformulation evaluation

---

## Persian Embedding Models

### Multilingual Models (Evaluated on FaMTEB)
| Model | Link | Dim | Notes |
|-------|------|-----|-------|
| **Jina Embeddings v2** | https://huggingface.co/jinaai/jina-embeddings-v2-base-en | 768 | Top FaMTEB performer |
| **BGE-m3** | https://huggingface.co/BAAI/bge-m3 | 1024 | Strong multilingual, supports long context |
| **GTE Multilingual** | https://huggingface.co/thenlper/gte-multilingual-base | 768 | Good retrieval performance |
| **paraphrase-multilingual-MiniLM-L12-v2** | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 388 | **Current KB Manager model** |
| **LaBSE** | https://huggingface.co/sentence-transformers/LaBSE | 768 | Language-agnostic BERT sentence embedding |

### Persian-Specific Models
| Model | Link | Type | Notes |
|-------|------|------|-------|
| **ParsBERT** | https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased | Encoder | Base Persian BERT |
| **FaBERT** | https://huggingface.co/m3hrdadfa/fa-bert-base | Encoder | Alternative Persian BERT |
| **ParsRoBERTa** | https://huggingface.co/HooshvareLab/roberta-base-parsbert-uncased | Encoder | RoBERTa architecture for Persian |
| **ParsGPT** | https://huggingface.co/HooshvareLab/gpt2-fa | Decoder | Persian GPT-2 |
| **Tooka-SBERT** | https://huggingface.co/partai/Tooka-SBERT | Bi-encoder | Persian sentence transformer |

### Reranking Models (Cross-Encoders)
| Model | Link | Type | Lang |
|-------|------|------|------|
| **mDeBERTa-v3-base-xsmall** | https://huggingface.co/microsoft/mdeberta-v3-base-xsmall | Cross-encoder | Multilingual (incl. Persian) |
| **mMARCO-MiniLM** | https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Cross-encoder | Multilingual |
| **Persian-Reranker** | https://huggingface.co/HooshvareLab/persian-reranker | Cross-encoder | Persian-specific |

---

## Persian NLP Libraries

### Hazm (هزم)
- **Link**: https://github.com/roshan-research/hazm
- **Features**: Tokenization, lemmatization, POS tagging, chunking, NER, dependency parsing
- **Install**: `pip install hazm`
- **Use case**: Persian text preprocessing, tokenization for BM25

### Parsivar (پارسی‌ور)
- **Link**: https://github.com/ipsina/parsivar
- **Features**: Tokenization, stemming, POS tagging, NER, spell checking, informal-to-formal conversion
- **Install**: `pip install parsivar`
- **Use case**: More comprehensive Persian text processing than Hazm

### Persian-Tools
- **Link**: https://github.com/persian-tools/persian-tools
- **Features**: Text normalization, number conversion, date conversion
- **Install**: `pip install persian-tools`

---

## Persian Query Reformulation Resources

### Datasets for Conversational Query Reformulation
| Dataset | Link | Description |
|---------|------|-------------|
| **QReCC** | https://github.com/apple/ml-qrecc | Conversational QA with rewrites |
| **TopiOCQA** | https://github.com/adlakha/topiocqa | Topic-switching conversational QA |
| **CAsT-19/20** | https://www.treccast.ai/ | TREC Conversational Assistance Track |

### Persian Paraphrase Datasets
| Dataset | Link | Description |
|---------|------|-------------|
| **Parsinlu Query Paraphrase** | https://huggingface.co/datasets/HooshvareLab/parsinlu-query-paraphrase | Query paraphrase detection |
| **Farsi Paraphrase Detection** | https://huggingface.co/datasets/HooshvareLab/farsi-paraphrase-detection | Sentence-level paraphrase |
| **ExaPPC** | https://github.com/sadeghii/exapc | 2.3M Persian paraphrase pairs |

---

## Persian Typo/Noise Resources

### Common Persian Typo Patterns
| Correct | Typo | Type |
|---------|------|------|
| می‌شود | میشه | Contraction |
| می‌کنند | می‌کنن | Contraction |
| دسترسی | دسترسى | ي vs ی |
| اعتبارى | اعتباری | ي vs ی |
| قراردآد | قرارداد | Diacritic omission |
| تشکيل | تکمیل | Phonetic substitution |

### ZWNJ (Zero Width Non-Joiner) Handling
- **Unicode**: U+200C
- **Purpose**: Separate compound word parts (e.g., `می‌روند` vs `میروند`)
- **Common issue**: Missing ZWNJ in user queries
- **Solution**: Normalize ZWNJ to space or handle in tokenizer

---

## Data Loading Examples

### Load FaMTEB Datasets
```python
from datasets import load_dataset

# Synthetic QA Retrieval
synper_qa = load_dataset("MCINext/synthetic-persian-qa-retrieval")

# Chatbot RAG Topics
chatbot_topics = load_dataset("MCINext/synthetic-persian-chatbot-rag-topics-retrieval")

# Chatbot RAG Summary
chatbot_summary = load_dataset("MCINext/synthetic-persian-chatbot-rag-summary-retrieval")

# NQ-Fa (BEIR-Fa)
nq_fa = load_dataset("MCINext/nq-fa")

# MIRACL Persian
miracle_fa = load_dataset("miracl/miracl", "fa")
```

### Load Persian Models
```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Current KB Manager model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Persian BERT
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")

# Cross-encoder reranker
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("microsoft/mdeberta-v3-base-xsmall")
```

### Use Hazm for Preprocessing
```python
from hazm import Normalizer, WordTokenizer, Lemmatizer

normalizer = Normalizer()
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()

text = normalizer.normalize("می‌شود و دسترسی")
tokens = tokenizer.tokenize(text)
lemmas = [lemmatizer.lemmatize(t) for t in tokens]
# Output: ['می‌شود', 'و', 'دسترسی']
```

---

## Evaluation Metrics for Persian Retrieval

### Standard IR Metrics (implemented in KB Manager)
- **Hit@K** / **Recall@K**: Binary relevance
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Precision@K**: Precision at cutoff

### Persian-Specific Considerations
1. **Tokenization**: Use Hazm/Parsivar for consistent tokenization
2. **Normalization**: Apply Persian normalization before metric computation
3. **Partial matching**: Character n-grams help with morphological variants
4. **Compound words**: ZWNJ handling affects exact match metrics

---

## Quick Start: Adding FaMTEB Evaluation to KB Manager

```bash
# Install dependencies
pip install datasets sentence-transformers

# Run evaluation on FaMTEB retrieval datasets
python -c "
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Load MIRACL Persian
dataset = load_dataset('miracl/miracl', 'fa', split='dev')
queries = [ex['query'] for ex in dataset]
corpus = {ex['docid']: ex['text'] for ex in dataset}
# ... run retrieval evaluation
"
```

---

## References

1. **FaMTEB Paper**: Zinvandi et al., "FaMTEB: Massive Text Embedding Benchmark in Persian Language", EMNLP 2025 Findings
2. **MIRACL Paper**: Zhang et al., "MIRACL: A Multilingual Retrieval Dataset", 2023
3. **BEIR Paper**: Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models", 2021
4. **ParsBERT Paper**: Farahani et al., "ParsBERT: Transformer-based Model for Persian Language Understanding", 2020
5. **Hazm**: Roshan et al., "Hazm: A Python Library for Persian Language Processing"

---

## Next Steps for KB Manager

1. **Add MIRACL-Fa evaluation** to `run_benchmark.py`
2. **Test cross-encoder** with `microsoft/mdeberta-v3-base-xsmall` on Persian
3. **Integrate Hazm tokenization** into BM25 for better Persian handling
4. **Use FaMTEB synthetic datasets** for training query reformulation models
5. **Evaluate ParsBERT vs MiniLM** for dense retrieval on Persian credit scoring domain