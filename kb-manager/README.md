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

## License

Private — ICS Credit Scoring
