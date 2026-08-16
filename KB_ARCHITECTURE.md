# ICS Knowledge Base Architecture & Processing Plan

## Table of Contents

1. [Current KB Analysis](#1-current-kb-analysis)
2. [Data Taxonomy & Content Types](#2-data-taxonomy--content-types)
3. [pgvector Database Schema](#3-pgvector-database-schema)
4. [Persian-Specific Preprocessing Pipeline](#4-persian-specific-preprocessing-pipeline)
5. [Chunking Strategy](#5-chunking-strategy)
6. [Embedding Strategy](#6-embedding-strategy)
7. [Hybrid Retrieval Architecture](#7-hybrid-retrieval-architecture)
8. [Data Monitoring & Quality](#8-data-monitoring--quality)
9. [Versioning System](#9-versioning-system)
10. [CI/CD Pipeline](#10-cicd-pipeline)
11. [KB Replacement & Update Flow](#11-kb-replacement--update-flow)
12. [Editing Interface](#12-editing-interface)
13. [Implementation Roadmap](#13-implementation-roadmap)

---

## 1. Current KB Analysis

### 1.1 Source Structure

The KB is a zip archive (`31Tir1405(1).zip`) containing **~160 files** across **18 top-level directories**, organized by domain in Persian:

```
extracted/
├── آیین نامه‌ها/                          # Regulations (10 PDFs)
├── اشکالات سامانه اعتباریتو/              # System Issues (2 files)
├── اطلاعات شرکت/                          # Company Info (8 xlsx + archive)
├── تخمین گر/                              # Estimator (1 docx)
├── توضیحات تشریحی ریزن کدها/              # Reason Code Explanations
│   ├── حقیقی/                             #   Individual
│   ├── حقوقی/                             #   Corporate
│   └── چک/                                #   Cheque
├── توضیحات گزارش/                         # Report Descriptions
├── رسیدگی به اعتراضات/                    # Dispute Handling
├── لیست API محصول/                         # Product API List
├── لینک‌های مهم/                           # Important Links
├── محتوای پابلیک مدل‌ها/                  # Public Model Content
│   ├── حقیقی/                             #   Individual
│   ├── حقوقی/                             #   Corporate
│   └── چک/                                #   Cheque
├── محدودیت‌ها و نکات مهم/                  # Limitations & Notes
├── مفاهیم پایه اعتبارسنجی/                # Base Credit Concepts
├── مقالات کاربردی/                         # Applied Articles
├── واژه های معادل یا متفاوت/              # Equivalent Terms
├── پرسش و پاسخ تامین کنندگان/             # Data Provider Q&A
├── پیشنهاد تسهیلات/                       # Loan Suggestions
├── دریافت API/                            # API Access
├── سوال و جواب‌های CRM/                   # CRM Q&A
│   ├── حقیقی و حقوقی/                     #   Individual & Corporate
│   ├── حقوقی/                             #   Corporate
│   └── چک/                                #   Cheque
└── سوالات پیشنهادی.xlsx                   # Suggested Questions
```

### 1.2 File Type Distribution

| Type | Count | Description |
|------|-------|-------------|
| `.xlsx` | ~90 | Structured data: Q&A, reason codes, models, articles |
| `.pdf` | ~12 | Regulations, legal documents |
| `.docx` | ~50 | Articles, notes, descriptions |
| `.docx` (temp) | ~5 | Temp/backup files (skip) |

### 1.3 Key Data Schemas Found

**Schema A: Reason Codes (RAG Core)**
Columns: `reason_code, model_name, model_id, brief_explanation, detailed_explanation, reason_text, improvement_suggestions, bin_score, bin_impact, feature_score, feature_impact, bin_details, keywords, feature_name, data_source`

**Schema B: CRM Q&A (RAG Core)**
Columns: `Question, Model, BriefAnswer, Answer, Keyword`

**Schema C: Articles/Documentation (RAG Core)**
Columns: `DocumentName, Title, SectionTitle, Content, Type, Version, Author(s), Heading, Keywords, Summary`

**Schema D: Public Model Content (Reference)**
Same as Schema C — detailed model explanations.

---

## 2. Data Taxonomy & Content Types

### 2.1 Content Classification

| Category | Content | Priority | Update Freq | Shelf Life |
|----------|---------|----------|-------------|------------|
| **Regulatory** | آیین نامه‌ها, قوانین | Critical | Quarterly | Long |
| **Operational** | CRM Q&A, API Docs | High | Weekly | Medium |
| **Analytical** | Reason Codes, Models | High | Monthly | Medium |
| **Educational** | Articles, Concepts | Medium | Monthly | Long |
| **Reference** | Company Info, Links | Low | Yearly | Long |

### 2.2 Domain Tags

```yaml
domains:
  individual:    # حقیقی - Individual/personal credit
  corporate:     # حقوقی - Corporate credit
  cheque:        # چک - Cheque-related
  general:       # عمومی - Cross-domain
  api:           # API - Technical integration
  regulatory:    # قانونی - Legal/regulatory
```

### 2.3 Content Lifecycle States

```
draft → review → approved → indexed → active → stale → archived → deleted
```

---

## 3. pgvector Database Schema

### 3.1 Core Schema (PostgreSQL + pgvector)

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- hybrid search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- DOCUMENTS TABLE (source of truth)
-- ============================================================
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_path     TEXT NOT NULL,                    -- original file path
    source_hash     TEXT NOT NULL,                    -- SHA-256 of source file
    content_hash    TEXT NOT NULL,                    -- SHA-256 of extracted text
    title           TEXT NOT NULL,
    domain          TEXT NOT NULL,                    -- individual/corporate/cheque/general
    category        TEXT NOT NULL,                    -- regulation/article/qa/reason_code/model
    file_type       TEXT NOT NULL,                    -- pdf/xlsx/docx
    language        TEXT DEFAULT 'fa',
    version         INTEGER DEFAULT 1,
    status          TEXT DEFAULT 'draft',             -- draft/review/approved/indexed/active/stale/archived
    shelf_life_days INTEGER,                          -- auto-staleness: NULL=never, 90=quarterly, etc.
    metadata        JSONB DEFAULT '{}',               -- arbitrary extra fields
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    indexed_at      TIMESTAMPTZ,
    embedding_model TEXT,                             -- which model produced embeddings
    embedding_dim   INTEGER                           -- 1536/1024/etc
);

CREATE INDEX idx_documents_source_hash ON documents(source_hash);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
CREATE INDEX idx_documents_domain ON documents(domain);
CREATE INDEX idx_documents_category ON documents(category);
CREATE INDEX idx_documents_status ON documents(status);

-- ============================================================
-- CHUNKS TABLE (retrieval unit)
-- ============================================================
CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,                 -- position within document
    chunk_type      TEXT NOT NULL,                    -- semantic/header/body/qa_pair/reason_detail
    content         TEXT NOT NULL,                    -- raw text of chunk
    content_tsv     TSVECTOR                          -- generated for hybrid search
        GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    heading_path    TEXT,                             -- "Model > Section > Subsection"
    keywords        TEXT[],                           -- extracted keywords
    token_count     INTEGER NOT NULL,

    -- Embedding (support multiple models via separate rows or columns)
    embedding       vector(1024),                     -- BGE-M3 at 1024d, or text-embedding-3-small at 1536
    embedding_model TEXT NOT NULL,                    -- model identifier
    embedding_version TEXT,                           -- model version hash

    -- Quality signals
    quality_score   FLOAT,                            -- 0-1, from preprocessing pipeline
    is_verified     BOOLEAN DEFAULT FALSE,            -- human-reviewed

    -- Metadata
    metadata        JSONB DEFAULT '{}',               -- page_num, section, etc.
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(document_id, ordinal)
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_embedding_model ON chunks(embedding_model);
CREATE INDEX idx_chunks_heading ON chunks USING gin(heading_path gin_trgm_ops);

-- HNSW index for vector search
CREATE INDEX idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN index for full-text hybrid search
CREATE INDEX idx_chunks_content_tsv ON chunks USING gin(content_tsv);
CREATE INDEX idx_chunks_content_trgm ON chunks USING gin(content gin_trgm_ops);

-- ============================================================
-- DOCUMENT VERSIONS TABLE (version tracking)
-- ============================================================
CREATE TABLE document_versions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version         INTEGER NOT NULL,
    content_hash    TEXT NOT NULL,
    source_hash     TEXT NOT NULL,
    change_summary  TEXT,                             -- what changed
    changed_by      TEXT,                             -- who changed it
    embedding_model TEXT,                             -- model used for this version
    chunk_count     INTEGER,                          -- how many chunks
    status          TEXT DEFAULT 'archived',          -- active/archived/rolled_back
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(document_id, version)
);

-- ============================================================
-- CHUNK MAPPINGS TABLE (for multi-model support)
-- ============================================================
CREATE TABLE chunk_embeddings (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id        UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding       vector(1024) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(chunk_id, embedding_model)
);

-- ============================================================
-- INGESTION JOBS TABLE (pipeline tracking)
-- ============================================================
CREATE TABLE ingestion_jobs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type        TEXT NOT NULL,                    -- full_rebuild/incremental/single_document
    status          TEXT DEFAULT 'pending',           -- pending/running/completed/failed
    source_version  TEXT,                             -- git commit or zip hash
    documents_total INTEGER DEFAULT 0,
    documents_ok    INTEGER DEFAULT 0,
    documents_failed INTEGER DEFAULT 0,
    chunks_total    INTEGER DEFAULT 0,
    embedding_model TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_log       TEXT,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- EVALUATION QUERIES TABLE (golden dataset)
-- ============================================================
CREATE TABLE eval_queries (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text      TEXT NOT NULL,
    expected_chunk_ids UUID[],                        -- which chunks should be retrieved
    expected_answer TEXT,                             -- ideal answer
    domain          TEXT,
    difficulty      TEXT DEFAULT 'medium',            -- easy/medium/hard
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_run_at     TIMESTAMPTZ,
    last_recall     FLOAT
);

-- ============================================================
-- RETRIEVAL LOGS TABLE (observability)
-- ============================================================
CREATE TABLE retrieval_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text      TEXT NOT NULL,
    query_embedding vector(1024),
    retrieved_chunk_ids UUID[],
    scores          FLOAT[],                         -- similarity scores
    combined_scores FLOAT[],                         -- after fusion
    latency_ms      INTEGER,
    embedding_model TEXT,
    reranker_used   BOOLEAN DEFAULT FALSE,
    answer_text     TEXT,
    faithfulness    FLOAT,                           -- LLM-graded
    relevancy       FLOAT,                           -- LLM-graded
    user_feedback   TEXT,                            -- thumbs_up/thumbs_down/null
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- COST TRACKING TABLE (budget monitoring)
-- ============================================================
CREATE TABLE cost_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation       TEXT NOT NULL,                    -- embed/rerank/generate
    model           TEXT NOT NULL,
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    latency_ms      INTEGER,
    cost_usd        FLOAT,
    feature_tag     TEXT,                             -- which feature triggered this
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### 3.2 Row-Level Security (Optional Multi-Tenant)

```sql
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY chunks_tenant_isolation ON chunks
    USING (document_id IN (
        SELECT id FROM documents WHERE domain = current_setting('app.current_domain')
    ));
```

### 3.3 Hybrid Search Function

```sql
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text TEXT,
    query_embedding vector(1024),
    domain_filter TEXT DEFAULT NULL,
    category_filter TEXT DEFAULT NULL,
    embedding_model_filter TEXT DEFAULT NULL,
    match_count INTEGER DEFAULT 20,
    vector_weight FLOAT DEFAULT 0.7,
    keyword_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    heading_path TEXT,
    document_title TEXT,
    vector_score FLOAT,
    keyword_score FLOAT,
    combined_score FLOAT,
    metadata JSONB
)
LANGUAGE sql STABLE
AS $$
    WITH vector_results AS (
        SELECT
            c.id,
            c.content,
            c.heading_path,
            c.metadata,
            1 - (c.embedding <=> query_embedding) AS v_score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
          AND (domain_filter IS NULL OR d.domain = domain_filter)
          AND (category_filter IS NULL OR d.category = category_filter)
          AND (embedding_model_filter IS NULL OR c.embedding_model = embedding_model_filter)
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count * 3
    ),
    keyword_results AS (
        SELECT
            c.id,
            ts_rank(c.content_tsv, plainto_tsquery('simple', query_text)) AS k_score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.content_tsv @@ plainto_tsquery('simple', query_text)
          AND (domain_filter IS NULL OR d.domain = domain_filter)
          AND (category_filter IS NULL OR d.category = category_filter)
        ORDER BY k_score DESC
        LIMIT match_count * 3
    ),
    fused AS (
        SELECT
            v.id,
            v.content,
            v.heading_path,
            v.metadata,
            v.v_score AS vector_score,
            COALESCE(k.k_score, 0) AS keyword_score,
            (vector_weight * v.v_score + keyword_weight * COALESCE(k.k_score, 0)) AS combined_score
        FROM vector_results v
        LEFT JOIN keyword_results k ON v.id = k.id
    )
    SELECT
        f.id AS chunk_id,
        f.content,
        f.heading_path,
        d.title AS document_title,
        f.vector_score,
        f.keyword_score,
        f.combined_score,
        f.metadata
    FROM fused f
    JOIN chunks c ON f.id = c.id
    JOIN documents d ON c.document_id = d.id
    ORDER BY f.combined_score DESC
    LIMIT match_count;
$$;
```

---

## 4. Persian-Specific Preprocessing Pipeline

### 4.1 Tool Stack

```yaml
persian_nlp_stack:
  primary:
    - name: "Hazm"
      version: "0.10+"
      purpose: "Normalization, tokenization, POS tagging, chunking, dependency parsing"
      accuracy: "POS 98.8%, Chunking 93.4%"
    - name: "Shekar"
      version: "latest"
      purpose: "Production-grade normalization, NER, spell checking, keyword extraction"
      features: "Academy of Persian Language compliant rules"
    - name: "ParsBERT"
      model: "HooshvareLab/bert-fa-zwnj-base"
      purpose: "Contextual embeddings for semantic understanding"

  supplementary:
    - name: "Parsivar"
      purpose: "Stemming, spell checking, dependency parsing"
    - name: "DadmaTools V2"
      purpose: "Adapter-based NLP pipeline, sentiment analysis"
    - name: "OpenAI text-embedding-3-small"
      dimensions: 1536
      purpose: "Primary embedding model (managed)"
    - name: "BGE-M3"
      dimensions: 1024
      purpose: "Alternative embedding model (local, multilingual)"
```

### 4.2 Preprocessing Pipeline (Step-by-Step)

```python
# pipeline/persian_preprocessor.py

from hazm import Normalizer, sent_tokenize, word_tokenize, Stemmer, POSTagger
from shekar import Normalizer as ShekarNormalizer, SpellChecker, KeywordExtractor
import hashlib
import re

class PersianKBPreprocessor:
    """
    Multi-stage preprocessing pipeline for Persian credit scoring KB.
    """

    def __init__(self):
        self.hazm_normalizer = Normalizer(
            remove_extra_spaces=True,
            normalize_unicode=True,
            normalize_punctuation=True,
            correct_spacing=True
        )
        self.shekar_normalizer = ShekarNormalizer()
        self.spell_checker = SpellChecker()
        self.stemmer = Stemmer()
        self.keyword_extractor = KeywordExtractor()

    def preprocess(self, text: str, content_type: str = "general") -> dict:
        """
        Full preprocessing pipeline. Returns normalized text + metadata.
        """
        result = {
            "original": text,
            "stages": {}
        }

        # Stage 1: Basic cleanup
        text = self._basic_cleanup(text)
        result["stages"]["cleanup"] = text

        # Stage 2: Persian normalization (Hazm)
        text = self.hazm_normalizer.normalize(text)
        result["stages"]["hazm_normalize"] = text

        # Stage 3: Persian normalization (Shekar - Academy compliant)
        text = self.shekar_normalizer.normalize(text)
        result["stages"]["shekar_normalize"] = text

        # Stage 4: ZWNJ correction (critical for Persian)
        text = self._fix_zwnj(text)
        result["stages"]["zwnj_fix"] = text

        # Stage 5: Number normalization (Persian ↔ Arabic numerals)
        text = self._normalize_numbers(text)
        result["stages"]["number_normalize"] = text

        # Stage 6: Spell checking (light, for obvious errors only)
        # Only apply for article-type content, not for technical codes
        if content_type in ("article", "model_content"):
            text = self._light_spell_check(text)
            result["stages"]["spell_check"] = text

        # Stage 7: Extract keywords
        keywords = self.keyword_extractor.extract(text)
        result["keywords"] = keywords

        # Stage 8: Sentence tokenization for validation
        sentences = sent_tokenize(text)
        result["sentence_count"] = len(sentences)

        result["final"] = text
        return result

    def _basic_cleanup(self, text: str) -> str:
        """Remove HTML artifacts, emails, URLs, mentions."""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+', '[URL]', text)
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        # Remove temp file markers
        text = re.sub(r'~\$\S+', '', text)
        # Collapse multiple spaces/newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _fix_zwnj(self, text: str) -> str:
        """Fix Zero-Width Non-Joiner issues in Persian."""
        # Common Persian suffixes that need ZWNJ
        zwnj_suffixes = [
            'ها', 'های', 'ای', 'ام', 'ات', 'اش',
            'یی', 'ایی', 'تر', 'ترین', 'گر', 'گری'
        ]
        # Fix missing ZWNJ before suffixes
        for suffix in zwnj_suffixes:
            text = re.sub(
                rf'(\w)(\s+{suffix})',
                rf'\1\u200c{suffix}',
                text
            )
        # Remove extra ZWNJs
        text = re.sub(r'\u200c{2,}', '\u200c', text)
        # Remove ZWNJ at start/end of words
        text = re.sub(r'\u200c(\s)', r'\1', text)
        text = re.sub(r'(\s)\u200c', r'\1', text)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """Normalize Persian/Arabic numerals and punctuation."""
        persian_digits = '۰۱۲۳۴۵۶۷۸۹'
        arabic_digits = '٠١٢٣٤٥٦٧٨٩'
        english_digits = '0123456789'

        # Arabic → Persian
        for a, p in zip(arabic_digits, persian_digits):
            text = text.replace(a, p)

        # Keep English digits for technical terms (API, model IDs, etc.)
        # Normalize Persian punctuation
        replacements = {
            '،': ',',    # Persian comma → standard
            '؛': ';',
            '؟': '?',
            '۔': '.',
        }
        # Keep Persian punctuation for content, normalize only for search
        return text

    def _light_spell_check(self, text: str) -> str:
        """Light spell checking - only fix obvious errors, don't rewrite."""
        words = text.split()
        corrected = []
        for word in words:
            if len(word) > 3 and not re.match(r'[\d۰-۹]+', word):
                suggestion = self.spell_checker.suggest(word)
                if suggestion and suggestion != word:
                    corrected.append(suggestion)
                else:
                    corrected.append(word)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def extract_section_hierarchy(self, text: str) -> list:
        """
        Extract heading hierarchy from structured documents.
        Returns list of (level, heading_text) tuples.
        """
        headings = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Detect markdown-style headings
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                heading_text = line.lstrip('#').strip()
                headings.append((level, heading_text))
            # Detect numbered sections (common in Persian legal docs)
            elif re.match(r'^[۰-۹\d]+[\.\)‌]', line):
                headings.append((2, line))
        return headings
```

### 4.3 Content-Type-Specific Preprocessing

```python
class ContentSpecificPreprocessor:
    """
    Different preprocessing strategies per content type.
    """

    def preprocess_reason_code(self, row: dict) -> dict:
        """
        Schema A: Reason code explanations.
        Combine brief + detailed explanation, keep structured metadata.
        """
        combined_text = f"""
دلیل امتیاز: {row.get('brief_explanation', '')}
توضیح کامل: {row.get('detailed_explanation', '')}
متن دلیل: {row.get('reason_text', '')}
پیشنهاد بهبود: {row.get('improvement_suggestions', '')}
""".strip()

        return {
            "content": combined_text,
            "heading_path": f"{row.get('model_name', '')} > {row.get('feature_name', '')} > {row.get('reason_code', '')}",
            "keywords": self._parse_keywords(row.get('keywords', '')),
            "chunk_type": "reason_detail",
            "metadata": {
                "reason_code": row.get('reason_code'),
                "model_id": row.get('model_id'),
                "feature_name": row.get('feature_name'),
                "data_source": row.get('data_source'),
                "bin_details": row.get('bin_details'),
                "bin_score": row.get('bin_score'),
                "feature_score": row.get('feature_score'),
            }
        }

    def preprocess_crm_qa(self, row: dict) -> dict:
        """
        Schema B: CRM Q&A pairs.
        Each Q&A is a self-contained chunk.
        """
        combined_text = f"""
سوال: {row.get('Question', '')}
پاسخ: {row.get('Answer', '')}
""".strip()

        return {
            "content": combined_text,
            "heading_path": f"问答 > {row.get('Model', '')}",
            "keywords": self._parse_keywords(row.get('Keyword', '')),
            "chunk_type": "qa_pair",
            "metadata": {
                "model": row.get('Model'),
                "brief_answer": row.get('BriefAnswer'),
            }
        }

    def preprocess_article(self, row: dict, section_text: str) -> dict:
        """
        Schema C: Articles/documentation.
        Section-based chunking with heading prefix.
        """
        # Prepend heading to improve retrieval
        heading = row.get('SectionTitle', row.get('Heading', ''))
        combined_text = f"{heading}\n\n{section_text}" if heading else section_text

        return {
            "content": combined_text,
            "heading_path": f"{row.get('DocumentName', '')} > {row.get('Title', '')} > {heading}",
            "keywords": self._parse_keywords(row.get('Keywords', '')),
            "chunk_type": "semantic",
            "metadata": {
                "document_name": row.get('DocumentName'),
                "title": row.get('Title'),
                "author": row.get('Author(s)'),
                "type": row.get('Type'),
                "summary": row.get('Summary'),
            }
        }

    def preprocess_regulation(self, pdf_text: str, page_num: int) -> dict:
        """
        PDF regulation documents.
        Split by article/section boundaries.
        """
        return {
            "content": pdf_text,
            "heading_path": f"قانون > صفحه {page_num}",
            "keywords": [],
            "chunk_type": "body",
            "metadata": {
                "page_number": page_num,
                "content_type": "regulation",
            }
        }

    def _parse_keywords(self, raw: str) -> list:
        """Parse Persian keywords from various formats."""
        if not raw:
            return []
        # Handle formats: "[kw1, kw2, kw3]" or "kw1، kw2، kw3"
        raw = raw.strip('[]')
        keywords = re.split(r'[,،]', raw)
        return [k.strip() for k in keywords if k.strip()]
```

---

## 5. Chunking Strategy

### 5.1 Chunking Rules

```yaml
chunking_config:
  default:
    strategy: "structure_aware"       # NOT fixed-size
    max_tokens: 512
    min_tokens: 100
    overlap_tokens: 50               # 10% of max, minimal
    split_on: "paragraph_boundaries"
    heading_prefix: true             # prepend section heading to each chunk

  content_type_overrides:
    reason_code:
      strategy: "single_entity"      # one reason code = one chunk
      max_tokens: 400

    qa_pair:
      strategy: "single_entity"      # one Q&A = one chunk
      max_tokens: 600

    regulation:
      strategy: "article_boundary"   # split on ماده (article)
      max_tokens: 800

    article:
      strategy: "section_heading"    # split on ### headings
      max_tokens: 512

    public_model:
      strategy: "section_heading"
      max_tokens: 512
```

### 5.2 Chunking Implementation

```python
class SmartChunker:
    """
    Structure-aware chunking for Persian KB content.
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, chunk_type: str = "auto") -> list[dict]:
        """
        Split text into chunks based on structure, not fixed-size windows.
        """
        if chunk_type == "single_entity":
            return [{"content": text, "ordinal": 0, "type": chunk_type}]

        # Step 1: Split on structural boundaries
        sections = self._split_on_structure(text)

        # Step 2: Further split oversized sections
        chunks = []
        for section in sections:
            if self._token_count(section) > self.max_tokens:
                sub_chunks = self._split_by_sentences(section)
                chunks.extend(sub_chunks)
            else:
                chunks.append(section)

        # Step 3: Apply minimal overlap
        chunks = self._add_overlap(chunks)

        return [
            {"content": c, "ordinal": i, "token_count": self._token_count(c)}
            for i, c in enumerate(chunks)
        ]

    def _split_on_structure(self, text: str) -> list[str]:
        """Split on Persian structural markers."""
        # Markdown headings
        # Persian numbered articles: ماده ۱, ماده ۲
        # Double newlines as fallback
        sections = []
        current = []

        for line in text.split('\n'):
            if self._is_section_break(line):
                if current:
                    sections.append('\n'.join(current))
                    current = []
            current.append(line)

        if current:
            sections.append('\n'.join(current))

        return [s.strip() for s in sections if s.strip()]

    def _is_section_break(self, line: str) -> bool:
        """Detect section boundaries in Persian text."""
        line = line.strip()
        if not line:
            return False
        # Markdown heading
        if line.startswith('#'):
            return True
        # Persian article: ماده ۱, ماده ۲
        if re.match(r'^ماده\s*[۰-۹\d]+', line):
            return True
        # Numbered section
        if re.match(r'^(فصل|بخش|بخشنامه)\s', line):
            return True
        return False

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split long text at sentence boundaries."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last sentence for overlap
                if self.overlap_tokens > 0:
                    overlap_sents = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s.split()) > self.overlap_tokens:
                            break
                        overlap_sents.insert(0, s)
                        overlap_len += len(s.split())
                    current_chunk = overlap_sents
                    current_len = overlap_len
                else:
                    current_chunk = []
                    current_len = 0
            current_chunk.append(sent)
            current_len += sent_len

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _token_count(self, text: str) -> int:
        """Approximate token count for Persian text."""
        # Persian words average ~2-3 tokens in most models
        return len(text.split()) * 2  # conservative estimate

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Already handled in _split_by_sentences."""
        return chunks
```

---

## 6. Embedding Strategy

### 6.1 Model Selection

```yaml
embedding_models:
  primary:
    name: "BGE-M3"
    provider: "BAAI"
    dimensions: 1024
    max_tokens: 8192
    language_support: "multilingual (excellent Persian)"
    cost: "self-hosted, free"
    rationale: "Best multilingual performance, local deployment, 1024d saves storage"

  fallback:
    name: "text-embedding-3-small"
    provider: "OpenAI"
    dimensions: 1536
    max_tokens: 8191
    cost: "$0.02/1M tokens"
    rationale: "High quality, managed, fallback if self-hosted unavailable"

  reranker:
    name: "BAAI/bge-reranker-v2-m3"
    provider: "BAAI"
    cost: "self-hosted"
    rationale: "Best multilingual reranker, ~80ms on CPU for 50 candidates"
```

### 6.2 Embedding Pipeline

```python
class EmbeddingPipeline:
    """
    Idempotent embedding pipeline with batch support.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", dimensions: int = 1024):
        self.model_name = model_name
        self.dimensions = dimensions
        self.model = self._load_model()

    def embed_documents(self, chunks: list[dict], batch_size: int = 64) -> list[dict]:
        """
        Embed chunks in batches. Idempotent: skips already-embedded chunks.
        """
        results = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            # Filter out already-embedded chunks
            to_embed = [c for c in batch if c.get('embedding') is None]

            if to_embed:
                texts = [c['content'] for c in to_embed]
                embeddings = self.model.encode(texts, normalize_embeddings=True)

                for chunk, emb in zip(to_embed, embeddings):
                    chunk['embedding'] = emb.tolist()
                    chunk['embedding_model'] = self.model_name

            results.extend(batch)

        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query with normalization."""
        embedding = self.model.encode([query], normalize_embeddings=True)
        return embedding[0].tolist()
```

### 6.3 Content Hashing (for incremental updates)

```python
import hashlib

def compute_content_hash(text: str) -> str:
    """SHA-256 hash of normalized content for change detection."""
    normalized = text.strip().lower()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def compute_file_hash(file_path: str) -> str:
    """SHA-256 hash of file bytes."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
```

---

## 7. Hybrid Retrieval Architecture

### 7.1 Retrieval Pipeline

```
User Query (Persian)
       │
       ▼
┌─────────────────────┐
│  Query Preprocessing │  ← Persian normalization, ZWNJ fix
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Query Embedding     │  ← BGE-M3 encode
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌──────────────┐
│ Vector   │ │ Keyword/BM25 │
│ Search   │ │ Search       │
│ (HNSW)   │ │ (pg_trgm)   │
└────┬─────┘ └──────┬───────┘
     │              │
     └──────┬───────┘
            ▼
   ┌────────────────┐
   │ Reciprocal Rank │  ← RRF fusion: 0.7 * vector + 0.3 * keyword
   │ Fusion (RRF)    │
   └────────┬───────┘
            │
            ▼
   ┌────────────────┐
   │ Cross-Encoder   │  ← bge-reranker-v2-m3 on top 20 candidates
   │ Reranking       │
   └────────┬───────┘
            │
            ▼
   ┌────────────────┐
   │ Top-5 Chunks    │  ← passed to LLM with prompt
   │ + Metadata      │
   └────────────────┘
```

### 7.2 Retrieval Implementation

```python
class HybridRetriever:
    """
    Production hybrid retrieval with reranking.
    """

    def __init__(self, db_pool, embedding_model, reranker_model):
        self.db = db_pool
        self.embedder = embedding_model
        self.reranker = reranker_model

    async def search(
        self,
        query: str,
        domain: str = None,
        category: str = None,
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rerank: bool = True
    ) -> list[dict]:

        # 1. Preprocess query
        query = self._preprocess_query(query)

        # 2. Embed query
        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True
        )[0].tolist()

        # 3. Hybrid search via SQL function
        candidates = await self.db.fetch(
            """
            SELECT * FROM hybrid_search(
                $1, $2, $3, $4, $5, 20, $6, $7
            )
            """,
            query, query_embedding, domain, category,
            self.embedder.model_name, vector_weight, keyword_weight
        )

        # 4. Rerank with cross-encoder
        if rerank and len(candidates) > 5:
            candidates = await self._rerank(query, candidates)

        # 5. Return top-k
        return candidates[:top_k]

    async def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Cross-encoder reranking."""
        pairs = [(query, c['content']) for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate['rerank_score'] = float(score)

        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

    def _preprocess_query(self, query: str) -> str:
        """Apply Persian normalization to query."""
        normalizer = PersianKBPreprocessor()
        result = normalizer.preprocess(query)
        return result['final']
```

---

## 8. Data Monitoring & Quality

### 8.1 Monitoring Metrics

```yaml
monitoring:
  retrieval_quality:
    - metric: "recall@10"
      threshold: 0.85
      check_frequency: "weekly"
      alert: "slack, email"

    - metric: "mrr@10"
      threshold: 0.60
      check_frequency: "weekly"

    - metric: "answer_faithfulness"
      threshold: 0.80
      check_frequency: "on_sample"

    - metric: "answer_relevancy"
      threshold: 0.75
      check_frequency: "on_sample"

  system_health:
    - metric: "retrieval_latency_p95"
      threshold: "200ms"
      check_frequency: "continuous"

    - metric: "end_to_end_latency_p95"
      threshold: "1500ms"
      check_frequency: "continuous"

    - metric: "embedding_cost_per_1k_queries"
      threshold: "$0.50"
      check_frequency: "daily"

  data_freshness:
    - metric: "stale_fraction"
      threshold: 0.05
      check_frequency: "daily"
      alert: "critical"

    - metric: "mean_document_age_days"
      threshold: "varies by shelf_life"
      check_frequency: "daily"

    - metric: "orphan_count"
      threshold: 0
      check_frequency: "daily"
```

### 8.2 Monitoring Implementation

```sql
-- Scheduled monitoring query (run via pg_cron or external scheduler)
-- Staleness check
SELECT
    d.category,
    d.domain,
    COUNT(*) as total_docs,
    COUNT(*) FILTER (WHERE d.indexed_at < NOW() - (d.shelf_life_days || ' days')::INTERVAL) as stale_docs,
    ROUND(
        COUNT(*) FILTER (WHERE d.indexed_at < NOW() - (d.shelf_life_days || ' days')::INTERVAL)::FLOAT
        / NULLIF(COUNT(*), 0) * 100, 2
    ) as stale_percentage
FROM documents d
WHERE d.shelf_life_days IS NOT NULL
  AND d.status = 'active'
GROUP BY d.category, d.domain;

-- Orphan check (docs deleted from source but still in index)
SELECT d.id, d.source_path, d.title
FROM documents d
WHERE NOT EXISTS (
    SELECT 1 FROM chunks c WHERE c.document_id = d.id
);

-- Retrieval drift detection (compare last 7 days vs baseline)
WITH recent AS (
    SELECT
        AVG(relevancy) as avg_relevancy,
        AVG(faithfulness) as avg_faithfulness,
        COUNT(*) as query_count
    FROM retrieval_logs
    WHERE created_at > NOW() - INTERVAL '7 days'
),
baseline AS (
    SELECT
        AVG(relevancy) as avg_relevancy,
        AVG(faithfulness) as avg_faithfulness
    FROM retrieval_logs
    WHERE created_at BETWEEN NOW() - INTERVAL '30 days' AND NOW() - INTERVAL '7 days'
)
SELECT
    r.avg_relevancy - b.avg_relevancy as relevancy_drift,
    r.avg_faithfulness - b.avg_faithfulness as faithfulness_drift,
    r.query_count
FROM recent r, baseline b;
```

### 8.3 Quality Checks in Pipeline

```python
class QualityGate:
    """
    Automated quality checks before indexing.
    """

    def validate_chunk(self, chunk: dict) -> list[str]:
        """Validate a single chunk before indexing."""
        issues = []

        # 1. Content quality
        if len(chunk['content'].strip()) < 50:
            issues.append("CHUNK_TOO_SHORT")

        if len(chunk['content']) > 2000:
            issues.append("CHUNK_TOO_LONG")

        # 2. Encoding quality
        garbled_chars = re.findall(r'[\ufffd\x00-\x08]', chunk['content'])
        if garbled_chars:
            issues.append("GARBLED_ENCODING")

        # 3. Persian quality
        persian_ratio = len(re.findall(r'[\u0600-\u06FF]', chunk['content'])) / max(len(chunk['content']), 1)
        if persian_ratio < 0.3 and chunk.get('language') == 'fa':
            issues.append("LOW_PERSIAN_RATIO")

        # 4. Embedding quality
        if chunk.get('embedding') is None:
            issues.append("MISSING_EMBEDDING")

        if chunk.get('embedding') and len(chunk['embedding']) == 0:
            issues.append("EMPTY_EMBEDDING")

        # 5. Metadata completeness
        if not chunk.get('heading_path'):
            issues.append("MISSING_HEADING")

        return issues

    def validate_document(self, doc: dict, chunks: list[dict]) -> dict:
        """Validate a complete document before indexing."""
        return {
            "document_id": doc['id'],
            "total_chunks": len(chunks),
            "valid_chunks": sum(1 for c in chunks if not self.validate_chunk(c)),
            "issues": [c for c in chunks if self.validate_chunk(c)],
            "pass": all(not self.validate_chunk(c) for c in chunks),
        }
```

---

## 9. Versioning System

### 9.1 Version Strategy

```yaml
versioning:
  documents:
    strategy: "content_hash_based"
    storage: "document_versions table"
    trigger: "any content change"
    retention: "keep all versions"

  embeddings:
    strategy: "model_version_tagged"
    storage: "embedding_model + embedding_version columns"
    trigger: "model change or content change"
    retention: "keep during migration window"

  chunking_config:
    strategy: "git_tracked"
    storage: "chunking_config.yaml in repo"
    trigger: "config change"
    retention: "git history"

  prompts:
    strategy: "git_tracked"
    storage: "prompts/ directory in repo"
    trigger: "prompt change"
    retention: "git history"

  evaluation_dataset:
    strategy: "versioned_with_DVC"
    storage: "DVC-tracked files"
    trigger: "dataset change"
    retention: "all versions"
```

### 9.2 Version Metadata in Chunks

```python
def create_versioned_chunk(chunk: dict, config: dict) -> dict:
    """
    Add version metadata to every chunk.
    """
    chunk['metadata']['version_info'] = {
        'ingestion_code_version': config['git_commit_hash'],
        'embedding_model': config['embedding_model'],
        'embedding_version': config['embedding_model_version'],
        'chunking_config_hash': config['chunking_config_hash'],
        'source_document_hash': chunk['source_content_hash'],
        'ingestion_timestamp': datetime.utcnow().isoformat(),
    }
    return chunk
```

---

## 10. CI/CD Pipeline

### 10.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Source   │───▶│ Validate │───▶│ Preprocess│───▶│  Chunk   │ │
│  │  Ingest  │    │  & Lint  │    │  (Persian)│    │ & Embed  │ │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘ │
│                                                         │       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │       │
│  │  Deploy  │◀───│  Eval    │◀───│  Index   │◀────────┘       │
│  │  (Blue/  │    │  Gate    │    │ (pgvector)│                  │
│  │  Green)  │    │          │    │          │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Monitor (ongoing)                      │  │
│  │  - Retrieval quality  - Data freshness  - Cost tracking  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 GitHub Actions Workflow

```yaml
# .github/workflows/kb-update.yml
name: Knowledge Base Update Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'kb-source/**'
  workflow_dispatch:
    inputs:
      rebuild_type:
        description: 'full_rebuild or incremental'
        required: true
        default: 'incremental'
        type: choice
        options:
          - incremental
          - full_rebuild

env:
  REBUILD_TYPE: ${{ github.event.inputs.rebuild_type || 'incremental' }}

jobs:
  # ============================================================
  # STAGE 1: Validate Source Files
  # ============================================================
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate source files
        run: |
          python scripts/validate_sources.py \
            --source-dir kb-source/ \
            --schema schemas/source_schema.json

      - name: Check for duplicates
        run: |
          python scripts/check_duplicates.py \
            --source-dir kb-source/

  # ============================================================
  # STAGE 2: Preprocess (Persian-specific)
  # ============================================================
  preprocess:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python -m spacy download xx_sent_ud_sm

      - name: Run preprocessing
        run: |
          python scripts/preprocess.py \
            --source-dir kb-source/ \
            --output-dir processed/ \
            --config configs/preprocessing.yaml

      - name: Upload processed artifacts
        uses: actions/upload-artifact@v4
        with:
          name: processed-kb
          path: processed/

  # ============================================================
  # STAGE 3: Chunk & Embed
  # ============================================================
  chunk-embed:
    needs: preprocess
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download processed artifacts
        uses: actions/download-artifact@v4
        with:
          name: processed-kb
          path: processed/

      - name: Chunk and embed
        run: |
          python scripts/chunk_embed.py \
            --input-dir processed/ \
            --output-dir embeddings/ \
            --model BAAI/bge-m3 \
            --chunk-config configs/chunking.yaml

      - name: Upload embedding artifacts
        uses: actions/upload-artifact@v4
        with:
          name: embeddings
          path: embeddings/

  # ============================================================
  # STAGE 4: Quality Gate (Evaluation)
  # ============================================================
  eval-gate:
    needs: chunk-embed
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Download embeddings
        uses: actions/download-artifact@v4
        with:
          name: embeddings
          path: embeddings/

      - name: Run evaluation
        run: |
          python scripts/evaluate.py \
            --embeddings-dir embeddings/ \
            --eval-dataset eval/golden_dataset.json \
            --thresholds configs/eval_thresholds.yaml \
            --output eval_results.json

      - name: Check evaluation gates
        run: |
          python scripts/check_gates.py \
            --results eval_results.json \
            --min-recall 0.85 \
            --min-faithfulness 0.80 \
            --min-relevancy 0.75

      - name: Upload eval results
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: eval_results.json

  # ============================================================
  # STAGE 5: Index to pgvector
  # ============================================================
  index:
    needs: eval-gate
    runs-on: ubuntu-latest
    if: success()
    steps:
      - uses: actions/checkout@v4

      - name: Download embeddings
        uses: actions/download-artifact@v4
        with:
          name: embeddings
          path: embeddings/

      - name: Index to database
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          python scripts/index_to_pgvector.py \
            --embeddings-dir embeddings/ \
            --rebuild-type $REBUILD_TYPE \
            --version ${{ github.sha }}

  # ============================================================
  # STAGE 6: Post-Index Validation
  # ============================================================
  post-validate:
    needs: index
    runs-on: ubuntu-latest
    steps:
      - name: Validate indexed data
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: |
          python scripts/post_index_validate.py \
            --check-orphan-chunks \
            --check-embedding-consistency \
            --check-staleness

  # ============================================================
  # STAGE 7: Notify & Report
  # ============================================================
  notify:
    needs: [index, post-validate]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Send notification
        if: always()
        run: |
          python scripts/notify.py \
            --status ${{ needs.index.result }} \
            --channel slack \
            --message "KB update ($REBUILD_TYPE) completed: ${{ needs.index.result }}"
```

### 10.3 Evaluation Gate Implementation

```python
# scripts/evaluate.py

import json
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

def run_evaluation(embeddings_dir: str, eval_dataset: str, thresholds: dict):
    """
    Run RAGAS evaluation against golden dataset.
    """
    # Load evaluation dataset
    with open(eval_dataset) as f:
        test_cases = json.load(f)

    # Load retriever
    retriever = HybridRetriever(...)

    results = []
    for case in test_cases:
        # Retrieve
        retrieved = retriever.search(case['query'], top_k=5)

        # Generate answer (using LLM)
        answer = generate_answer(case['query'], retrieved)

        results.append({
            "query": case['query'],
            "answer": answer,
            "contexts": [r['content'] for r in retrieved],
            "ground_truth": case['expected_answer'],
        })

    # Evaluate with RAGAS
    dataset = Dataset.from_list(results)
    eval_results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    )

    return eval_results

def check_gates(results: dict, thresholds: dict) -> bool:
    """Check if evaluation meets quality gates."""
    passed = True

    for metric, threshold in thresholds.items():
        score = results.get(metric, 0)
        if score < threshold:
            print(f"FAIL: {metric} = {score:.3f} < {threshold}")
            passed = False
        else:
            print(f"PASS: {metric} = {score:.3f} >= {threshold}")

    return passed
```

---

## 11. KB Replacement & Update Flow

### 11.1 Update Types

```yaml
update_types:
  full_rebuild:
    trigger: "New zip archive from business team"
    description: "Complete KB replacement"
    steps:
      1. Archive current KB (snapshot)
      2. Extract new zip
      3. Diff against current (identify adds/changes/deletes)
      4. Preprocess all content
      5. Chunk & embed
      6. Run evaluation gates
      7. Blue-green switch
      8. Verify & monitor
    estimated_time: "2-4 hours"
    risk: "high - affects all queries"

  incremental_update:
    trigger: "Individual file changes"
    description: "Update specific documents"
    steps:
      1. Detect changed files (content hash diff)
      2. Preprocess changed files only
      3. Re-chunk & re-embed changed content
      4. Update index (UPSERT)
      5. Run spot-check evaluation
      6. Monitor retrieval quality
    estimated_time: "10-30 minutes"
    risk: "medium - affects related queries"

  content_addition:
    trigger: "New file added to KB"
    description: "Add new content without modifying existing"
    steps:
      1. Validate new file format
      2. Preprocess
      3. Chunk & embed
      4. Insert into index
      5. Verify searchability
    estimated_time: "5-15 minutes"
    risk: "low - no existing content affected"

  content_deletion:
    trigger: "File removed from KB"
    description: "Remove content from index"
    steps:
      1. Identify document in database
      2. Soft-delete (mark as 'archived')
      3. Remove chunks after grace period
      4. Verify no orphan references
    estimated_time: "1-5 minutes"
    risk: "low"
```

### 11.2 Full KB Replacement Flow

```python
class KBReplacementManager:
    """
    Manages complete KB replacement with zero-downtime.
    """

    async def full_replacement(self, new_zip_path: str):
        """
        Complete KB replacement flow.
        """
        # Step 1: Snapshot current state
        snapshot_id = await self._create_snapshot()
        print(f"Snapshot created: {snapshot_id}")

        try:
            # Step 2: Extract new content
            new_files = self._extract_zip(new_zip_path)

            # Step 3: Diff against current
            diff = await self._compute_diff(new_files)
            print(f"Changes: {diff['added']} added, {diff['changed']} changed, {diff['removed']} removed")

            # Step 4: Preprocess all content
            processed = []
            for file in new_files:
                content = self._read_file(file)
                chunks = self._preprocess_and_chunk(content, file)
                processed.extend(chunks)

            # Step 5: Embed (full rebuild)
            embeddings = await self._embed_all(processed)

            # Step 6: Validate with quality gates
            eval_results = await self._run_evaluation(embeddings)
            if not self._check_gates(eval_results):
                raise Exception("Quality gates failed - reverting")

            # Step 7: Index to NEW schema version (blue-green)
            new_schema_version = self._create_new_schema_version()
            await self._index_to_schema(embeddings, new_schema_version)

            # Step 8: Switch traffic to new version
            await self._switch_traffic(new_schema_version)

            # Step 9: Monitor for 1 hour
            await self._monitor_post_switch(duration_hours=1)

            # Step 10: Cleanup old version
            await self._cleanup_old_version(snapshot_id)

            print("KB replacement completed successfully")

        except Exception as e:
            print(f"KB replacement failed: {e}")
            await self._rollback(snapshot_id)
            raise

    async def _create_snapshot(self) -> str:
        """Snapshot current database state for rollback."""
        snapshot_id = f"snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        await self.db.execute(f"""
            CREATE TABLE {snapshot_id}_docs AS
            SELECT * FROM documents WHERE status = 'active';

            CREATE TABLE {snapshot_id}_chunks AS
            SELECT * FROM chunks WHERE document_id IN (
                SELECT id FROM documents WHERE status = 'active'
            );
        """)

        return snapshot_id

    async def _rollback(self, snapshot_id: str):
        """Rollback to snapshot on failure."""
        await self.db.execute(f"""
            -- Mark current active docs as failed
            UPDATE documents SET status = 'failed'
            WHERE status = 'active' AND version = (
                SELECT MAX(version) FROM documents
            );

            -- Restore from snapshot
            INSERT INTO documents SELECT * FROM {snapshot_id}_docs;
            INSERT INTO chunks SELECT * FROM {snapshot_id}_chunks;
        """)
```

### 11.3 Incremental Update Flow

```python
class IncrementalUpdater:
    """
    Handles incremental KB updates.
    """

    async def detect_changes(self, source_dir: str) -> dict:
        """
        Detect which files have changed since last ingestion.
        """
        changes = {
            'added': [],
            'modified': [],
            'removed': []
        }

        current_files = self._scan_directory(source_dir)
        indexed_files = await self._get_indexed_files()

        current_hashes = {f['path']: f['hash'] for f in current_files}
        indexed_hashes = {f['source_path']: f['source_hash'] for f in indexed_files}

        for path, hash_val in current_hashes.items():
            if path not in indexed_hashes:
                changes['added'].append(path)
            elif indexed_hashes[path] != hash_val:
                changes['modified'].append(path)

        for path in indexed_hashes:
            if path not in current_hashes:
                changes['removed'].append(path)

        return changes

    async def update_incremental(self, changes: dict):
        """
        Apply incremental changes.
        """
        # Process additions
        for path in changes['added']:
            await self._add_document(path)

        # Process modifications
        for path in changes['modified']:
            await self._update_document(path)

        # Process removals
        for path in changes['removed']:
            await self._remove_document(path)

        # Verify
        await self._verify_index_consistency()

    async def _update_document(self, path: str):
        """Update a single document: re-chunk, re-embed, upsert."""
        # Read new content
        content = self._read_file(path)
        new_hash = compute_content_hash(content)

        # Get old version
        old_doc = await self._get_document(path)

        # Skip if content unchanged
        if old_doc and old_doc['content_hash'] == new_hash:
            return

        # Preprocess & chunk
        chunks = self._preprocess_and_chunk(content, path)

        # Embed
        embeddings = await self._embed_chunks(chunks)

        # Transactional update
        async with self.db.transaction():
            # Update document metadata
            await self.db.execute("""
                UPDATE documents
                SET content_hash = $1,
                    version = version + 1,
                    updated_at = NOW(),
                    indexed_at = NOW()
                WHERE source_path = $2
                RETURNING id
            """, new_hash, path)

            # Delete old chunks
            await self.db.execute("""
                DELETE FROM chunks WHERE document_id = $1
            """, old_doc['id'])

            # Insert new chunks
            for i, chunk in enumerate(embeddings):
                await self.db.execute("""
                    INSERT INTO chunks (
                        document_id, ordinal, chunk_type, content,
                        heading_path, keywords, token_count,
                        embedding, embedding_model, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """, old_doc['id'], i, chunk['chunk_type'],
                    chunk['content'], chunk['heading_path'],
                    chunk['keywords'], chunk['token_count'],
                    chunk['embedding'], chunk['embedding_model'],
                    json.dumps(chunk['metadata']))
```

---

## 12. Editing Interface

### 12.1 Web-Based KB Editor

```yaml
editor_features:
  document_management:
    - Upload new files (xlsx, pdf, docx)
    - Edit document metadata (domain, category, shelf_life)
    - Preview extracted content
    - Compare versions side-by-side

  chunk_management:
    - View chunk list with scores
    - Edit chunk content inline
    - Merge/split chunks
    - Preview retrieval results for specific chunks
    - Mark chunks as verified/rejected

  quality_monitoring:
    - Dashboard with retrieval metrics
    - Stale content alerts
    - Failed chunks list
    - Cost tracking

  version_control:
    - View version history
    - Rollback to previous version
    - Diff between versions
    - Tag releases

  evaluation:
    - Manage golden dataset
    - Run evaluation on-demand
    - View evaluation history
    - Compare evaluation results across versions
```

### 12.2 API Endpoints

```yaml
api_endpoints:
  documents:
    - GET    /api/documents              # List documents
    - GET    /api/documents/{id}         # Get document detail
    - POST   /api/documents              # Upload new document
    - PUT    /api/documents/{id}         # Update document metadata
    - DELETE /api/documents/{id}         # Soft-delete document
    - POST   /api/documents/{id}/reindex # Re-index single document

  chunks:
    - GET    /api/documents/{id}/chunks  # List chunks for document
    - PUT    /api/chunks/{id}            # Edit chunk content
    - POST   /api/chunks/{id}/verify     # Mark as verified
    - POST   /api/chunks/merge           # Merge multiple chunks
    - POST   /api/chunks/split           # Split a chunk

  search:
    - POST   /api/search                 # Hybrid search
    - POST   /api/search/preview         # Preview search results

  evaluation:
    - GET    /api/eval/queries           # List evaluation queries
    - POST   /api/eval/queries           # Add evaluation query
    - POST   /api/eval/run               # Run evaluation
    - GET    /api/eval/results           # Get evaluation results

  monitoring:
    - GET    /api/monitor/health         # System health
    - GET    /api/monitor/staleness      # Staleness report
    - GET    /api/monitor/costs          # Cost report
    - GET    /api/monitor/metrics        # Quality metrics

  pipeline:
    - POST   /api/pipeline/build         # Trigger full rebuild
    - POST   /api/pipeline/incremental   # Trigger incremental update
    - GET    /api/pipeline/status/{id}   # Check pipeline status
    - POST   /api/pipeline/rollback      # Rollback to snapshot
```

---

## 13. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

```
Week 1:
├── Set up PostgreSQL + pgvector
├── Create database schema
├── Implement Persian preprocessing pipeline
├── Test with sample files (reason codes, CRM Q&A)
└── Basic content hash + versioning

Week 2:
├── Implement structure-aware chunker
├── Set up BGE-M3 embedding pipeline
├── Build hybrid search function
├── Basic evaluation with 50 golden queries
└── Manual testing with 10 representative queries
```

### Phase 2: Pipeline (Weeks 3-4)

```
Week 3:
├── Build full ingestion pipeline
├── Process all Excel files (reason codes, Q&A, articles)
├── Process PDF regulations
├── Process DOCX articles
├── Index everything to pgvector
└── Run evaluation, tune parameters

Week 4:
├── Build incremental update pipeline
├── Implement content change detection
├── Set up GitHub Actions CI/CD
├── Build quality gates
└── Test with simulated file changes
```

### Phase 3: Monitoring & Interface (Weeks 5-6)

```
Week 5:
├── Build monitoring dashboards (Grafana)
├── Implement staleness detection
├── Build cost tracking
├── Implement retrieval logging
└── Set up alerting (Slack/email)

Week 6:
├── Build web editing interface (FastAPI + React)
├── Document management UI
├── Chunk editing UI
├── Evaluation management
└── Version history & rollback
```

### Phase 4: Production Hardening (Weeks 7-8)

```
Week 7:
├── Load testing (k6)
├── Security review
├── RLS implementation (if multi-tenant)
├── Backup & disaster recovery
└── Performance tuning (HNSW params, connection pooling)

Week 8:
├── Full regression testing
├── Documentation
├── Training for business users
├── Production deployment
└── Post-launch monitoring (1 week)
```

---

## Appendix A: Technology Stack

```yaml
infrastructure:
  database: "PostgreSQL 16 + pgvector 0.8"
  vector_index: "HNSW (m=16, ef_construction=64)"
  search: "Hybrid (vector + pg_trgm BM25)"

nlp_tools:
  persian_normalization: "Hazm + Shekar"
  spell_checking: "Shekar SpellChecker"
  keyword_extraction: "Shekar KeywordExtractor"
  tokenization: "Hazm sent_tokenize + word_tokenize"

embedding:
  primary: "BGE-M3 (1024d, self-hosted)"
  fallback: "OpenAI text-embedding-3-small (1536d)"
  reranker: "BAAI/bge-reranker-v2-m3"

backend:
  api: "FastAPI (Python 3.11+)"
  orm: "SQLAlchemy + asyncpg"
  task_queue: "Celery + Redis"

frontend:
  editor: "React + TypeScript"
  dashboard: "Grafana"

ci_cd:
  pipeline: "GitHub Actions"
  evaluation: "RAGAS"
  experiment_tracking: "MLflow"

monitoring:
  metrics: "Prometheus + Grafana"
  logging: "Structured JSON logs"
  alerting: "Slack webhooks"
```

## Appendix B: File Processing Priority

```yaml
processing_order:
  phase_1_high_value:
    - "توضیحات تشریحی ریزن کدها/**/*.xlsx"    # Reason codes (core RAG)
    - "سوال و جواب‌های CRM/**/*.xlsx"          # CRM Q&A (core RAG)
    - "مقالات کاربردی/**/*.xlsx"               # Articles (RAG)
    - "محتوای پابلیک مدل‌ها/**/*.xlsx"        # Public models (RAG)

  phase_2_reference:
    - "مقالات کاربردی/**/*.docx"               # Article DOCXs
    - "آیین نامه‌ها/**/*.pdf"                  # Regulations
    - "مفاهیم پایه اعتبارسنجی/**/*.xlsx"     # Base concepts

  phase_3_supplementary:
    - "اطلاعات شرکت/**/*.xlsx"                # Company info
    - "لینک‌های مهم/**/*.xlsx"                # Important links
    - "واژه های معادل یا متفاوت/**/*.xlsx"   # Equivalent terms
    - "پرسش و پاسخ تامین کنندگان/**/*.xlsx"  # Provider Q&A
```

## Appendix C: Golden Dataset Structure

```json
{
  "version": "1.0",
  "created_at": "2026-08-16",
  "queries": [
    {
      "id": "q001",
      "query": "امتیاز اعتباری چیست؟",
      "expected_answer": "امتیاز اعتباری عددی بین ۲۵۰ تا ۹۰۰ است...",
      "expected_chunks": ["chunk_id_1", "chunk_id_2"],
      "domain": "general",
      "difficulty": "easy",
      "tags": ["credit_score", "basic_concept"]
    },
    {
      "id": "q002",
      "query": "چک برگشتی چه تأثیری بر امتیاز شرکت دارد؟",
      "expected_answer": "اگر شرکت چک برگشتی رفع سوءاثر نشده داشته باشد...",
      "expected_chunks": ["chunk_id_10", "chunk_id_11"],
      "domain": "corporate",
      "difficulty": "medium",
      "tags": ["cheque", "corporate", "reason_code"]
    }
  ]
}
```
