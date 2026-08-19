# KB Manager - Knowledge Base Management System

> **ICS Credit Scoring Knowledge Base** — Process, version, and manage your Persian-language knowledge base for RAG agents.

## Overview

KB Manager is a complete pipeline for ingesting, preprocessing, chunking, embedding, and managing a knowledge base stored in PostgreSQL with pgvector. It is designed for **Persian (Farsi)** content in the credit scoring domain but is generalizable to any language.

### Key Features

- **Multi-format parsing**: Excel (xlsx), PDF, DOCX with schema auto-detection
- **Persian preprocessing**: ZWNJ normalization, Unicode cleanup, spell checking (Hazm + Shekar)
- **Pluggable chunking**: Semantic (structure-aware) and fixed-size strategies
- **Pluggable embeddings**: Sentence-transformers with content-hash caching
- **pgvector storage**: HNSW-indexed vector search + full-text hybrid search
- **Version tracking**: Every document change is versioned with rollback support
- **Quality gates**: Automated validation before indexing
- **CLI + Web UI**: Command-line and browser-based management
- **Docker-ready**: PostgreSQL + pgvector + app in one compose stack

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Source Files│───▶│   Parsers    │───▶│Preprocess│───▶│ Chunker  │───▶│ Embedder │
│  (xlsx/pdf/  │    │  (schema     │    │ (Persian │    │(semantic/│    │(sentence │
│   docx)      │    │   detection) │    │  clean)  │    │  fixed)  │    │ -trans.) │
└─────────────┘    └──────────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                                            │
                                                                            ▼
                                                                     ┌──────────┐
                                                                     │ pgvector │
                                                                     │(PostgreSQL│
                                                                     │ + HNSW)  │
                                                                     └──────────┘
```

## Quick Start

### 1. Start PostgreSQL with pgvector (Docker)

```bash
docker compose up -d postgres
```

This starts PostgreSQL 16 with pgvector on port 5432.

### 2. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 3. Initialize Database

```bash
psql -h localhost -U postgres -d postgres -f scripts/init_db.sql
```

### 4. Ingest Knowledge Base

```bash
# Full rebuild
kb-manager ingest --source-dir /path/to/kb/source --full

# Incremental update (only changed files)
kb-manager ingest --source-dir /path/to/kb/source
```

### 5. Start Web UI

```bash
kb-manager serve
# Open http://localhost:8000
```

### 6. Search

```bash
kb-manager search --query "امتیاز اعتباری چیست؟" --top-k 5
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `kb-manager ingest` | Ingest documents into the KB |
| `kb-manager status` | Show current KB statistics |
| `kb-manager search` | Search the knowledge base |
| `kb-manager serve` | Start the web server |
| `kb-manager inspect` | Inspect a file's structure |

### Ingest Options

```bash
kb-manager ingest --source-dir ./data --full --model "BAAI/bge-m3"
```

- `--source-dir / -s`: Source directory containing KB files
- `--full`: Full rebuild (ignore change detection)
- `--model / -m`: Embedding model override

## Web UI

The web interface provides:

- **Dashboard**: Overview with document/chunk counts and domain breakdown
- **Documents**: Browse, filter, upload, and manage documents
- **Chunks**: View and edit individual chunks, mark as verified
- **Pipeline**: Trigger ingestion runs, view job history
- **Versions**: Version history and rollback for documents
- **Monitoring**: Staleness reports, metrics, quality indicators

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KB_DB_HOST` | localhost | PostgreSQL host |
| `KB_DB_PORT` | 5432 | PostgreSQL port |
| `KB_DB_NAME` | kb_manager | Database name |
| `KB_DB_USER` | postgres | Database user |
| `KB_DB_PASSWORD` | postgres | Database password |
| `KB_EMBED_MODEL` | paraphrase-multilingual-MiniLM-L12-v2 | Embedding model |
| `KB_EMBED_DIM` | 384 | Embedding dimensions |
| `KB_EMBED_BATCH` | 64 | Embedding batch size |
| `KB_CHUNK_STRATEGY` | semantic | Chunking strategy |
| `KB_CHUNK_MAX` | 512 | Max tokens per chunk |
| `KB_SOURCE_DIR` | ./data | Default source directory |
| `KB_WEB_HOST` | 0.0.0.0 | Web server host |
| `KB_WEB_PORT` | 8000 | Web server port |

### Config Files

- `configs/default.yaml` — Default configuration
- `configs/chunking/semantic.yaml` — Semantic chunking parameters
- `configs/chunking/fixed.yaml` — Fixed-size chunking parameters

## Project Structure

```
kb-manager/
├── kb_manager/           # Main package
│   ├── config.py         # Configuration management
│   ├── cli.py            # CLI interface
│   ├── models/           # SQLAlchemy models + schemas
│   │   ├── database.py   # ORM models
│   │   └── schemas.py    # Pydantic schemas
│   ├── parsers/          # File parsers
│   │   ├── xlsx_parser.py
│   │   ├── pdf_parser.py
│   │   └── docx_parser.py
│   ├── preprocessor/     # Text preprocessing
│   │   ├── persian.py    # Persian-specific normalization
│   │   ├── clean.py      # Generic cleaning
│   │   └── pipeline.py   # Chained pipeline
│   ├── chunker/          # Text chunking
│   │   ├── semantic.py   # Structure-aware chunking
│   │   └── fixed.py      # Fixed-size chunking
│   ├── embedder/         # Vector embeddings
│   │   └── sentence_transformer.py
│   ├── pipeline/         # Ingestion orchestration
│   │   ├── orchestrator.py
│   │   ├── versioning.py
│   │   └── quality.py
│   └── web/              # FastAPI web UI
│       ├── app.py
│       ├── routes/
│       └── templates/
├── tests/                # Test suite
├── configs/              # Configuration files
├── scripts/              # Database scripts
├── data/                 # Source data (gitignored)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kb_manager --cov-report=html

# Run specific test file
pytest tests/test_preprocessor.py -v
```

### Code Quality

```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy kb_manager/
```

### Adding a New Parser

1. Create a class inheriting from `BaseParser` in `kb_manager/parsers/base.py`
2. Implement `parse()` and `can_parse()` methods
3. Register in `kb_manager/parsers/registry.py`

### Adding a New Chunking Strategy

1. Create a class inheriting from `BaseChunker` in `kb_manager/chunker/base.py`
2. Implement `chunk()` method
3. Register in `kb_manager/chunker/registry.py`

## Data Schema

### Supported Input Formats

**Schema A: Reason Codes**
| Column | Description |
|--------|-------------|
| `reason_code` | Unique code (e.g., CNMCTA1) |
| `model_name` | Model name (حقوقی/حقیقی/چک) |
| `model_id` | Model version (ICS.Credit.C-V.2) |
| `brief_explanation` | Brief reason explanation |
| `detailed_explanation` | Detailed explanation |
| `reason_text` | Short reason text |
| `improvement_suggestions` | How to improve |
| `keywords` | Comma-separated keywords |
| `feature_name` | Feature identifier |
| `data_source` | Data source name |

**Schema B: CRM Q&A**
| Column | Description |
|--------|-------------|
| `Question` | User question |
| `Model` | Domain (حقوقی/حقیقی/چک) |
| `BriefAnswer` | Short answer |
| `Answer` | Full answer |
| `Keyword` | Keywords |

**Schema C: Articles**
| Column | Description |
|--------|-------------|
| `DocumentName` | Document identifier |
| `Title` | Article title |
| `SectionTitle` | Section heading |
| `Content` | Main content |
| `Type` | Content type |
| `Version` | Version |
| `Author(s)` | Author |
| `Keywords` | Keywords |
| `Summary` | Summary |

## Database Schema

### Tables

- `documents` — Source document metadata
- `chunks` — Text chunks with embeddings
- `document_versions` — Version history
- `ingestion_jobs` — Pipeline job tracking
- `retrieval_logs` — Query logs for monitoring

### Indexes

- HNSW index on `chunks.embedding` for vector search
- GIN index on `chunks.content_tsv` for full-text search
- GIN trigram index on `chunks.content` for fuzzy search

## Docker

### Full Stack

```bash
docker compose up -d
```

This starts:
- PostgreSQL 16 with pgvector (port 5432)
- KB Manager web UI (port 8000)

### Just PostgreSQL

```bash
docker compose up -d postgres
```

## Retrieval Benchmark & Improvements

### Retrieval Pipeline (v3)

The search pipeline now uses a **two-stage hybrid** retriever:

1. **BM25** (lexical) — Okapi BM25 with Persian-aware tokenization
2. **Dense semantic** — `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) cosine similarity over precomputed chunk embeddings
3. **Fusion** — Reciprocal Rank Fusion (RRF, k=60) over both legs

The TF-IDF cosine leg was removed (redundant with BM25, O(n) per-query latency).

### Benchmark Results (120 queries, 6 formats, Top-5)

| Format | Hit@5 | Top-1 | MRR | Avg Latency |
|--------|-------|-------|-----|-------------|
| **verbatim** | 100% | 90% | 0.942 | 9.8s (cold model load) |
| **paraphrase** | 100% | 95% | 0.975 | 279ms |
| **typo** | 100% | 90% | 0.942 | 257ms |
| **conversational** | 100% | 85% | 0.917 | 277ms |
| **reworded** | 95% | 65% | 0.760 | 265ms |
| **keyword_only** | 40% | 10% | 0.188 | 243ms |
| **Overall** | **89.2%** | **72.5%** | **0.787** | **~1.9s** |

> **v2 → v3 delta**: Top-1 **+9.2%**, MRR **+6.9%**, Latency **−34%** (TF-IDF leg removed)

### Key Findings

- **Dense embeddings** (MiniLM multilingual) are the single biggest win for semantic queries (paraphrase, reworded, typo, conversational) — Top-1 jumped from ~63% → 72%.
- **Keyword_only** remains weak (40% hit) because its test queries are corrupted: the generator extracts a merged "keywords + model" header line (`کلیدواژه‌ها: بروزرسانی، بازپرداخت، وام، گزارش اعتباری مدل: حقیقی و حقوقی...`). This is a **data-quality issue in the test generator**, not a retrieval gap. Real user keyword queries would be cleaner.
- **Latency** dropped from ~2.8s → ~1.9s by removing the O(n) TF-IDF full scan; dense query encode (~70ms) + matmul (µs) is fast.
- Embeddings are **cached to disk** (`data/dense_embeddings.npz`) keyed by corpus fingerprint — rebuild only when KB changes.

### Generated Plots (in `data/plots/`)

- `hit_rate_by_format.png` — Hit@5 per query format
- `mrr_by_format.png` — MRR per query format
- `latency_distribution.png` — Query latency histogram
- `qa_duplication.png` — Deduplication statistics

### Running the Benchmark

```bash
# From kb-manager/
KB_DB_URL="sqlite+aiosqlite:///data/kb_test.db" python run_benchmark.py test_questions.json 5
```

## Technical Report: Embedding Model, Retrieval Methods & Evaluation Metrics

### Embedding Model

| Property | Value |
|----------|-------|
| **Model** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Architecture** | MiniLM (12-layer Transformer encoder, 384 hidden dims, 12 attention heads) |
| **Training Objective** | Contrastive learning on multilingual paraphrase pairs (STS benchmark, ParaCrawl, WikiMatrix, etc.) |
| **Languages** | 50+ languages including Persian (Farsi), Arabic, English |
| **Max Sequence Length** | 128 tokens (truncation applied) |
| **Output Dimension** | 384 (L2-normalized) |
| **Pooling** | Mean pooling over token embeddings |
| **Model Size** | ~120M parameters, ~470 MB on disk |
| **Inference** | CPU: ~70 ms/query (batch=1), GPU: ~5 ms/query |

The model maps each chunk's text content to a 384-dimensional unit hypersphere where semantic similarity corresponds to cosine similarity:

$$
\mathbf{e}_c = \text{normalize}\left( \text{MiniLM}(\text{chunk}_c) \right) \in \mathbb{R}^{384}, \quad \|\mathbf{e}_c\|_2 = 1
$$

At query time, the same encoder produces a query embedding $\mathbf{e}_q$, and cosine similarity with all chunk embeddings is computed via matrix multiplication:

$$
\text{sim}(q, c) = \mathbf{e}_q^\top \mathbf{e}_c = \cos(\theta_{q,c})
$$

### Retrieval Methods

#### 1. BM25 (Lexical Retrieval)

Okapi BM25 ranks chunks by term-frequency / inverse-document-frequency with document-length normalization:

$$
\text{score}_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

where:
- $f(t,d)$ = term frequency of $t$ in document $d$
- $|d|$ = document length (token count)
- $\text{avgdl}$ = average document length in corpus
- $k_1 = 1.5$ (term frequency saturation)
- $b = 0.75$ (length normalization)
- $\text{IDF}(t) = \log \frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1.0$
- $N$ = total documents, $\text{df}(t)$ = document frequency of term $t$

**Persian tokenization**: Unicode range `[\u0600-\u06FF\u0750-\u077F\u200C\u200D\d]+` with stopword removal and length > 1 filter. Arabic/Persian character normalization (ي→ی, ك→ک, ZWNJ handling) applied before tokenization.

#### 2. Dense Semantic Retrieval

Chunk embeddings are precomputed offline and stored as an L2-normalized matrix $\mathbf{E} \in \mathbb{R}^{N \times 384}$ where $N$ = number of chunks. Query embedding $\mathbf{e}_q$ is computed online. Cosine similarity is computed via:

$$
\mathbf{s} = \mathbf{E} \mathbf{e}_q \in \mathbb{R}^N, \quad \text{where } \mathbf{e}_q = \text{normalize}(\text{MiniLM}(q))
$$

Top-$k$ chunks are selected by $\arg\max$ over $\mathbf{s}$. Complexity: $O(N \cdot d)$ for query ($d=384$), dominated by BLAS matmul (microseconds on CPU).

#### 3. Reciprocal Rank Fusion (RRF)

Three ranked lists (BM25, Dense) are fused without score calibration:

$$
\text{RRF}(d) = \sum_{i=1}^{L} \frac{1}{k + \text{rank}_i(d)}
$$

where:
- $L$ = number of retrievers (2: BM25 + Dense)
- $\text{rank}_i(d)$ = 1-based rank of document $d$ in retriever $i$'s results
- $k = 60$ (empirical constant mitigating rank differences across systems)

RRF is **rank-based** (not score-based), making it robust to different score distributions. The fused score determines final ranking.

---

### Evaluation Methods & Mathematical Formulas

The benchmark evaluates retrieval quality over a test set of $Q$ queries, each with ground-truth relevant chunk IDs $\mathcal{R}_q \subset \mathcal{C}$ (typically $|\mathcal{R}_q| = 1$ for QA pairs).

#### Notation

- $Q$ = set of test queries
- $\mathcal{C}$ = set of all chunk IDs
- $\mathcal{R}_q$ = relevant chunk IDs for query $q$
- $\mathcal{A}_q = [a_1, a_2, \dots, a_K]$ = retrieved chunk IDs at rank $1..K$ (top-$K$)
- $\text{rel}_q(c) = \begin{cases} 1 & c \in \mathcal{R}_q \\ 0 & \text{otherwise} \end{cases}$ = binary relevance
- $K$ = cutoff (typically 5)

---

#### 1. Hit Rate @ K (Recall@K)

$$
\text{Hit@}K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\left[ \mathcal{A}_q \cap \mathcal{R}_q \neq \varnothing \right]
$$

Measures whether **at least one** relevant chunk appears in top-$K$. Binary per-query.

---

#### 2. Top-1 Accuracy

$$
\text{Top-1} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{1}\left[ a_1 \in \mathcal{R}_q \right]
$$

Strict metric: relevant chunk must be **ranked #1**.

---

#### 3. Mean Reciprocal Rank (MRR)

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
$$

where $\text{rank}_q = \min \{ i : a_i \in \mathcal{R}_q \}$, or $\infty$ if no hit (contributes 0).

Rewards **early** retrieval of relevant chunks. Range: $[0, 1]$.

---

#### 4. Mean Average Precision @ K (MAP@K)

For query $q$, precision at rank $i$:

$$
P_q(i) = \frac{1}{i} \sum_{j=1}^{i} \text{rel}_q(a_j)
$$

Average Precision for $q$ (binary relevance):

$$
\text{AP}_q = \frac{1}{|\mathcal{R}_q|} \sum_{i=1}^{K} P_q(i) \cdot \text{rel}_q(a_i)
$$

Then:

$$
\text{MAP@}K = \frac{1}{|Q|} \sum_{q \in Q} \text{AP}_q
$$

Measures **precision across all relevant positions** up to $K$. Higher when relevant items appear early and consistently.

---

#### 5. Normalized Discounted Cumulative Gain @ K (NDCG@K)

With binary relevance $\text{rel}_q(c) \in \{0, 1\}$:

$$
\text{DCG@}K_q = \sum_{i=1}^{K} \frac{\text{rel}_q(a_i)}{\log_2(i+1)}
$$

Ideal DCG (IDCG) assumes all relevant items ranked first:

$$
\text{IDCG@}K_q = \sum_{i=1}^{\min(K, |\mathcal{R}_q|)} \frac{1}{\log_2(i+1)}
$$

Then:

$$
\text{NDCG@}K = \frac{1}{|Q|} \sum_{q \in Q} \frac{\text{DCG@}K_q}{\text{IDCG@}K_q}
$$

Rewards **ranking relevant items higher**; logarithmic discount penalizes lower ranks. Range: $[0, 1]$.

---

#### 6. Precision @ K

$$
\text{Precision@}K = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathcal{A}_q \cap \mathcal{R}_q|}{K}
$$

Proportion of retrieved items that are relevant. With $|\mathcal{R}_q|=1$, max is $1/K$.

---

#### 7. Recall @ K

$$
\text{Recall@}K = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathcal{A}_q \cap \mathcal{R}_q|}{|\mathcal{R}_q|}
$$

Proportion of relevant items retrieved. With $|\mathcal{R}_q|=1$, equals Hit@K.

---

### Query Format Transformations

The benchmark generates 6 query variants per ground-truth question to test robustness:

| Format | Transformation | Target Similarity Band |
|--------|---------------|------------------------|
| `verbatim` | Identity (strip trailing ؟) | 0.80–1.00 |
| `paraphrase` | Synonym swap (3) + middle shuffle (2) + ask wrapper | 0.45–0.79 |
| `reworded` | Drop 40% tokens + synonym swap + shuffle (3) + ask wrapper + filler | 0.05–0.44 |
| `keyword_only` | Extract keywords field → split on `[،,;؛\n]` → first 6 tokens | 0.00–0.30 |
| `typo` | Apply Persian typo map (ي→ى, ك→ک, ZWNJ drop) | 0.40–0.85 |
| `conversational` | Formal→informal (می‌شود→میشه, می‌توانم→می‌تونم) + prefix + suffix | 0.05–0.45 |

Similarity measured by **token-set Jaccard** between variant and ground-truth question after Persian normalization (ي/ى→ی, ك→ک, ZWNJ→space, alef variants→ا).

---

### Corpus Fingerprinting (Cache Invalidation)

Dense embeddings cache is invalidated when corpus changes. Fingerprint:

$$
\text{fp}(\{t_i\}_{i=1}^N) = \text{SHA256}\left( \big\|_{i=1}^N \left( \text{len}(t_i) \parallel t_i \right) \right)
$$

where $\parallel$ denotes concatenation. Cache hit iff stored fingerprint == current fingerprint AND chunk count matches.

---

### Latency Breakdown

| Stage | v2 (BM25+TF-IDF) | v3 (BM25+Dense) |
|-------|------------------|-----------------|
| BM25 index build | ~27s (first query) | ~27s (first query) |
| Dense index build | N/A | ~120s (first run, cached to .npz) |
| Query: BM25 search | ~10ms | ~10ms |
| Query: Dense encode | N/A | ~70ms |
| Query: Dense matmul | N/A | <5ms |
| Query: TF-IDF cosine | ~1-2s (O(N)) | **Removed** |
| **Total per query (warm)** | **~2.8s** | **~1.9s** |

---

## License

Private — ICS Credit Scoring
