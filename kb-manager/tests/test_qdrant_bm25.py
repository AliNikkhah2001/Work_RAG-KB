#!/usr/bin/env python3
"""
test_qdrant_bm25.py - Comprehensive E2E Test Suite and Verification Script
for BM25 Sparse Vectors + Qdrant Native RRF Hybrid Search.

Covers:
- Acceptance Criteria Verification (AC1, AC2, AC3)
- Tier 1: Feature Coverage (Insertion, Query, Index Alignment, Schema)
- Tier 2: Boundary & Corner Cases (Empty text, single token, OOV, repeated words, ZWNJ)
- Tier 3: Cross-Feature Combinations (Dense vs Sparse balance, varying top_k, batching)
- Tier 4: Real-World Persian Banking / Credit Loan Scenarios
"""

import math
import os
import random
import sys
import unittest
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

# Attempt imports of the target modules under test
try:
    from bm25_sparse import PersianBM25Encoder
    BM25_AVAILABLE = True
    BM25_IMPORT_ERROR = None
except ImportError as err:
    PersianBM25Encoder = None
    BM25_AVAILABLE = False
    BM25_IMPORT_ERROR = str(err)

try:
    import qdrant_hybrid
    from qdrant_hybrid import (
        COLLECTION_NAME,
        get_qdrant_client,
        insert_chunks,
        recreate_collection,
        search_hybrid,
    )
    QDRANT_HYBRID_AVAILABLE = True
    QDRANT_HYBRID_IMPORT_ERROR = None
    SPARSE_VECTOR_NAME = getattr(qdrant_hybrid, "SPARSE_VECTOR_NAME", "bm25")
except ImportError as err:
    qdrant_hybrid = None
    get_qdrant_client = None
    recreate_collection = None
    insert_chunks = None
    search_hybrid = None
    COLLECTION_NAME = "credit_rag_chunks"
    SPARSE_VECTOR_NAME = "bm25"
    QDRANT_HYBRID_AVAILABLE = False
    QDRANT_HYBRID_IMPORT_ERROR = str(err)

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import (
        Distance,
        FusionQuery,
        PointStruct,
        Prefetch,
        ScoredPoint,
        SparseVector,
        SparseVectorParams,
        VectorParams,
    )
    QDRANT_CLIENT_AVAILABLE = True
except ImportError as err:
    QdrantClient = None
    models = None
    QDRANT_CLIENT_AVAILABLE = False


# ==============================================================================
# Helper Utilities & Fixtures
# ==============================================================================

def make_deterministic_dense_vector(dim: int = 1024, seed: int = 42) -> List[float]:
    """Generate a unit-normalized deterministic dense vector."""
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [round(x / norm, 6) for x in raw]


def make_similar_dense_vector(base_vector: List[float], noise_level: float = 0.05, seed: int = 99) -> List[float]:
    """Generate a dense vector with high cosine similarity to base_vector."""
    rng = random.Random(seed)
    dim = len(base_vector)
    noise = [rng.gauss(0, 1) for _ in range(dim)]
    noise_norm = math.sqrt(sum(x * x for x in noise)) or 1.0
    perturbed = [b + noise_level * (n / noise_norm) for b, n in zip(base_vector, noise)]
    pert_norm = math.sqrt(sum(x * x for x in perturbed)) or 1.0
    return [round(x / pert_norm, 6) for x in perturbed]


def make_orthogonal_dense_vector(base_vector: List[float], seed: int = 101) -> List[float]:
    """Generate a dense vector orthogonal (zero cosine similarity) to base_vector."""
    rng = random.Random(seed)
    dim = len(base_vector)
    random_vec = [rng.gauss(0, 1) for _ in range(dim)]
    dot = sum(b * r for b, r in zip(base_vector, random_vec))
    orth = [r - dot * b for b, r in zip(base_vector, random_vec)]
    norm = math.sqrt(sum(x * x for x in orth)) or 1.0
    return [round(x / norm, 6) for x in orth]


# Realistic Persian banking knowledge base chunks for testing
REALISTIC_BANKING_CHUNKS: List[Dict[str, Any]] = [
    {
        "id": "chunk_marriage_loan_001",
        "doc_id": "doc_marriage_loan",
        "doc_title": "دستورالعمل تسهیلات قرض‌الحسنه ازدواج",
        "heading_path": "تسهیلات / قرض‌الحسنه / ازدواج",
        "folder_hierarchy": ["تسهیلات", "قرض‌الحسنه"],
        "ordinal": 1,
        "text": (
            "شرایط و مدارک لازم جهت دریافت وام قرض‌الحسنه ازدواج در سال ۱۴۰۳: متقاضیان "
            "باید در سامانه تسهیلات قرض‌الحسنه ازدواج بانک مرکزی ثبت‌نام نموده و مدارک هویتی "
            "شامل شناسنامه و کارت ملی زوجین را به همراه یک ضامن معتبر با اعتبارسنجی بانکی ارائه دهند."
        ),
    },
    {
        "id": "chunk_credit_scoring_002",
        "doc_id": "doc_credit_scoring",
        "doc_title": "راهنمای استعلام گزارش اعتبارسنجی",
        "heading_path": "اعتبارسنجی / سامانه ملی / نمره اعتباری",
        "folder_hierarchy": ["اعتبارسنجی", "سامانه ملی"],
        "ordinal": 2,
        "text": (
            "دستورالعمل نحوه دریافت گزارش اعتبارسنجی مشتریان حقیقی و حقوقی از سامانه ملی "
            "اعتبارسنجی ایران. نمره اعتباری بر اساس سابقه پرداخت اقساط تسهیلات، وضعیت چک‌های برگشتی "
            "و تعهدات بانکی محاسبه می‌گردد و رتبه اعتباری بالاتر شانس تایید وام را افزایش می‌دهد."
        ),
    },
    {
        "id": "chunk_housing_loan_003",
        "doc_id": "doc_housing_loan",
        "doc_title": "تسهیلات خرید و ساخت مسکن",
        "heading_path": "مسکن / خرید مسکن / اقساط و سود",
        "folder_hierarchy": ["مسکن", "تسهیلات خرید"],
        "ordinal": 3,
        "text": (
            "سقف تسهیلات خرید مسکن از محل اوراق ممتاز بانک مسکن و نحوه محاسبه اقساط ماهیانه "
            "و سود دوران مشارکت مدنی. مدت بازپرداخت اقساط حداکثر ۱۲ سال بوده و سود تسهیلات "
            "بر اساس نرخ مصوب شورای پول و اعتبار تعیین می‌شود."
        ),
    },
    {
        "id": "chunk_deposit_account_004",
        "doc_id": "doc_deposit_account",
        "doc_title": "حساب‌های سپرده سرمایه‌گذاری مدت‌دار",
        "heading_path": "سپرده‌ها / سرمایه‌گذاری / نرخ سود",
        "folder_hierarchy": ["سپرده‌ها", "سرمایه‌گذاری"],
        "ordinal": 4,
        "text": (
            "شرایط افتتاح حساب سپرده سرمایه‌گذاری کوتاه‌مدت و بلندمدت بانکی، نرخ سود علی‌الحساب "
            "و حداقل موجودی لازم برای اشخاص حقیقی. سود سپرده به صورت ماهانه به حساب جاری متصل "
            "واریز شده و مشمول معافیت مالیاتی قانونی است."
        ),
    },
    {
        "id": "chunk_bounced_cheque_005",
        "doc_id": "doc_bounced_cheque",
        "doc_title": "قوانین رفع سوء اثر از چک‌های برگشتی",
        "heading_path": "چک صیادی / رفع سوء اثر / حساب جاری",
        "folder_hierarchy": ["خدمات ریالی", "چک صیادی"],
        "ordinal": 5,
        "text": (
            "مراحل قانونی و نحوه رفع سوء اثر از چک‌های برگشتی صیادی در سامانه یکپارچه بانک مرکزی: "
            "واریز مبلغ کسری به حساب جاری و مسدود کردن آن، ارائه لاشه چک به شعبه بانک افتتاح‌کننده "
            "حساب و یا دریافت رضایت‌نامه رسمی محضری از دارنده چک."
        ),
    },
    {
        "id": "chunk_microloan_006",
        "doc_id": "doc_microloan",
        "doc_title": "تسهیلات خرد بدون ضامن",
        "heading_path": "تسهیلات خرد / بدون ضامن / سفته الکترونیک",
        "folder_hierarchy": ["تسهیلات", "تسهیلات خرد"],
        "ordinal": 6,
        "text": (
            "شرایط پرداخت تسهیلات خرد بانکی بدون ضامن برای شاغلان و بازنشستگان دولتی و بخش خصوصی. "
            "مبنای اعطای وام رتبه اعتباری A و B در سامانه اعتبارسنجی است و صرفاً با ارائه سفته الکترونیکی "
            "و کسر از حقوق بدون نیاز به ضامن پرداخت می‌گردد."
        ),
    },
]


# ==============================================================================
# Base Test Case Class with Dependency Assertions
# ==============================================================================

class BaseBM25QdrantTestCase(unittest.TestCase):
    """Base class providing assertions and test lifecycle setup."""

    def setUp(self) -> None:
        if not QDRANT_CLIENT_AVAILABLE:
            self.skipTest("qdrant-client package is not available in environment.")
        if not BM25_AVAILABLE:
            self.fail(f"bm25_sparse module not implemented or failed to import: {BM25_IMPORT_ERROR}")
        if not QDRANT_HYBRID_AVAILABLE:
            self.fail(f"qdrant_hybrid module not implemented or failed to import: {QDRANT_HYBRID_IMPORT_ERROR}")

        # Unique in-memory collection per test for isolation
        self.collection_name = f"test_col_{uuid.uuid4().hex[:8]}"
        self.client = get_qdrant_client(location=":memory:")
        self.encoder = PersianBM25Encoder()

    def tearDown(self) -> None:
        if hasattr(self, "client") and self.client is not None:
            try:
                if self.client.collection_exists(self.collection_name):
                    self.client.delete_collection(self.collection_name)
            except Exception:
                pass

    def assert_sparse_vector_valid(self, indices: List[int], values: List[float], allow_empty: bool = False) -> None:
        """Verify strict invariants of a Qdrant SparseVector representation."""
        self.assertIsInstance(indices, list, "indices must be a List")
        self.assertIsInstance(values, list, "values must be a List")
        self.assertEqual(len(indices), len(values), "Length of indices and values must match")

        if not allow_empty:
            self.assertGreater(len(indices), 0, "indices must not be empty for non-empty text")

        for idx in indices:
            self.assertIsInstance(idx, int, f"Index {idx} is not an int")
            self.assertGreaterEqual(idx, 0, f"Index {idx} must be non-negative (>= 0)")
            self.assertLess(idx, 2**32, f"Index {idx} must fit in uint32 (< 2**32)")

        # Unique indices invariant: Qdrant sparse vectors reject or corrupt duplicate indices
        self.assertEqual(len(indices), len(set(indices)), "Sparse vector indices must be strictly unique!")

        for val in values:
            self.assertIsInstance(val, (float, int), f"Value {val} is not a numeric type")
            self.assertGreater(val, 0.0, f"Sparse weight value {val} must be strictly positive")


# ==============================================================================
# Acceptance Criteria Verification (ORIGINAL_REQUEST.md AC1 - AC3)
# ==============================================================================

class TestAcceptanceCriteriaVerification(BaseBM25QdrantTestCase):
    """
    Direct verification of Acceptance Criteria specified in ORIGINAL_REQUEST.md:
    AC1: Standalone insertion of Persian text into local Qdrant (:memory:) with dense & sparse vectors.
    AC2: Execution of hybrid search using Prefetch + FusionQuery(fusion="rrf") without typing/indexing errors.
    AC3: Exact match of BM25 query sparse integer indices to chunk insertion integer indices.
    """

    def test_ac1_in_memory_hybrid_insertion(self) -> None:
        """AC1: Successfully insert sample Persian text with dense and sparse vectors under same point ID."""
        dense_dim = 1024
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        doc_text = "شرایط دریافت وام قرض‌الحسنه بانک ملی ایران اعلام شد."
        indices, values = self.encoder.encode_document(doc_text)
        self.assert_sparse_vector_valid(indices, values)

        dense_vec = make_deterministic_dense_vector(dim=dense_dim, seed=1)
        point_id = "550e8400-e29b-41d4-a716-446655440001"

        chunks = [
            {
                "id": point_id,
                "text": doc_text,
                "dense_vector": dense_vec,
                "sparse_indices": indices,
                "sparse_values": values,
                "doc_id": "doc_1",
                "doc_title": "وام بانک ملی",
            }
        ]

        # Insert chunks
        insert_chunks(self.client, chunks, collection_name=self.collection_name)

        # Retrieve and verify point exists with both vectors
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True,
            with_payload=True,
        )
        self.assertEqual(len(points), 1, "Inserted point was not retrieved from Qdrant")
        point = points[0]

        # Verify point ID
        self.assertEqual(str(point.id), point_id)

        # Verify dense and sparse vectors are stored under the same point
        self.assertIsInstance(point.vector, dict, "Point vector should be a dict of named vectors")
        self.assertIn("dense", point.vector, "Dense vector missing from point")
        self.assertIn(SPARSE_VECTOR_NAME, point.vector, f"Sparse vector '{SPARSE_VECTOR_NAME}' missing from point")

        # Verify payload
        self.assertEqual(point.payload["text"], doc_text)
        self.assertEqual(point.payload["doc_id"], "doc_1")

    def test_ac2_native_rrf_hybrid_search(self) -> None:
        """AC2: Execute hybrid search with Prefetch + FusionQuery(fusion='rrf') without errors."""
        dense_dim = 1024
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        doc1_text = "دستورالعمل استعلام گزارش اعتبارسنجی و نمره اعتباری مشتریان بانکی."
        doc2_text = "مدارک مورد نیاز جهت افتتاح حساب سپرده سرمایه‌گذاری کوتاه‌مدت."

        idx1, val1 = self.encoder.encode_document(doc1_text)
        idx2, val2 = self.encoder.encode_document(doc2_text)

        d1 = make_deterministic_dense_vector(dim=dense_dim, seed=10)
        d2 = make_deterministic_dense_vector(dim=dense_dim, seed=20)

        chunks = [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "text": doc1_text,
                "dense_vector": d1,
                "sparse_indices": idx1,
                "sparse_values": val1,
                "doc_title": "اعتبارسنجی",
            },
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "text": doc2_text,
                "dense_vector": d2,
                "sparse_indices": idx2,
                "sparse_values": val2,
                "doc_title": "سپرده",
            },
        ]
        insert_chunks(self.client, chunks, collection_name=self.collection_name)

        # Search with Persian query targeting doc1
        query_text = "استعلام گزارش اعتبارسنجی بانکی"
        q_idx, q_val = self.encoder.encode_query(query_text)
        q_dense = make_similar_dense_vector(d1, noise_level=0.01, seed=55)

        results = search_hybrid(
            client=self.client,
            query_dense=q_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=2,
            collection_name=self.collection_name,
        )

        self.assertIsInstance(results, list, "search_hybrid must return a list of ScoredPoint")
        self.assertGreater(len(results), 0, "search_hybrid returned empty results")
        for point in results:
            self.assertTrue(hasattr(point, "score"), "ScoredPoint missing score")
            self.assertTrue(hasattr(point, "payload"), "ScoredPoint missing payload")
            self.assertIsInstance(point.score, float, "Score must be a float")

        # Top result must match doc1
        self.assertEqual(results[0].id, "11111111-1111-1111-1111-111111111111")
        self.assertEqual(results[0].payload["doc_title"], "اعتبارسنجی")

    def test_ac3_query_token_index_alignment(self) -> None:
        """AC3: BM25 query sparse vector correctly maps to the exact same integer indices used during chunk insertion."""
        doc_text = "وام قرض‌الحسنه مسکن برای جوانان"
        query_text = "دریافت وام قرض‌الحسنه"

        doc_indices, doc_values = self.encoder.encode_document(doc_text)
        query_indices, query_values = self.encoder.encode_query(query_text)

        self.assert_sparse_vector_valid(doc_indices, doc_values)
        self.assert_sparse_vector_valid(query_indices, query_values)

        # Both contain the terms 'وام' and 'قرض‌الحسنه'
        # Query individual terms to obtain their canonical token indices
        term_vam_idx, _ = self.encoder.encode_query("وام")
        term_gharz_idx, _ = self.encoder.encode_query("قرض‌الحسنه")

        self.assertEqual(len(term_vam_idx), 1, "Single token 'وام' should produce 1 index")
        vam_index = term_vam_idx[0]

        # The term 'وام' index must appear in BOTH doc_indices and query_indices
        self.assertIn(vam_index, doc_indices, "Index for 'وام' missing from document sparse indices")
        self.assertIn(vam_index, query_indices, "Index for 'وام' missing from query sparse indices")

        # For all common tokens between doc and query, the indices must be identical
        doc_idx_set = set(doc_indices)
        query_idx_set = set(query_indices)
        overlap = doc_idx_set.intersection(query_idx_set)

        self.assertGreater(len(overlap), 0, "Overlapping tokens between doc and query must share identical indices")
        self.assertIn(vam_index, overlap, "Index for 'وام' must be in the overlap set")


# ==============================================================================
# Tier 1: Feature Coverage (Insertion, Query, Index Alignment, Schema)
# ==============================================================================

class TestTier1FeatureCoverage(BaseBM25QdrantTestCase):
    """
    Tier 1: Feature coverage for normalizer, encoder, schema creation, chunk insertion,
    and hybrid query interface.
    """

    def test_persian_character_normalization(self) -> None:
        """Verify Arabic-to-Persian character mapping (ك -> ک, ي -> ی, diacritics)."""
        # Arabic kaf and yeh vs Persian kaf and yeh
        arabic_text = "بانك ملّي ايران"      # Contains \u0643 (ك), \u064a (ي), tashdeed (\u0651)
        persian_text = "بانک ملی ایران"     # Standard Persian \u06a9 (ک), \u06cc (ی)

        idx_arabic, _ = self.encoder.encode_document(arabic_text)
        idx_persian, _ = self.encoder.encode_document(persian_text)

        self.assertEqual(
            set(idx_arabic),
            set(idx_persian),
            "Arabic variants (ك, ي, diacritics) must normalize to identical Persian indices",
        )

    def test_encoder_output_types_and_ranges(self) -> None:
        """Verify encode_document and encode_query satisfy index range and value types."""
        text = "دستورالعمل اعتبارسنجی تسهیلات خرد بانک مرکزی"

        doc_idx, doc_val = self.encoder.encode_document(text)
        query_idx, query_val = self.encoder.encode_query(text)

        self.assert_sparse_vector_valid(doc_idx, doc_val)
        self.assert_sparse_vector_valid(query_idx, query_val)

        # All indices must fit within uint32: 0 <= idx < 2**32
        for idx in doc_idx + query_idx:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 2**32)

        # Values must be positive floats
        for val in doc_val + query_val:
            self.assertIsInstance(val, float)
            self.assertGreater(val, 0.0)

    def test_deterministic_stateless_index_mapping(self) -> None:
        """Verify two independent encoder instances yield identical indices."""
        encoder_a = PersianBM25Encoder()
        encoder_b = PersianBM25Encoder()

        phrase = "سپرده سرمایه‌گذاری کوتاه‌مدت بانک پاسارگاد"
        idx_a, val_a = encoder_a.encode_document(phrase)
        idx_b, val_b = encoder_b.encode_document(phrase)

        self.assertEqual(idx_a, idx_b, "Two separate encoder instances must yield identical indices")
        self.assertEqual(val_a, val_b, "Two separate encoder instances must yield identical weights")

    def test_qdrant_collection_schema_bm25(self) -> None:
        """Verify recreate_collection configures 'dense' and 'bm25' named vectors."""
        dense_dim = 512
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        col_info = self.client.get_collection(self.collection_name)

        # Check vectors config
        vectors_config = col_info.config.params.vectors
        self.assertIn("dense", vectors_config, "Collection missing 'dense' vector configuration")
        self.assertEqual(vectors_config["dense"].size, dense_dim)
        self.assertEqual(vectors_config["dense"].distance, Distance.COSINE)

        # Check sparse vectors config
        sparse_config = col_info.config.params.sparse_vectors
        self.assertIsNotNone(sparse_config, "Collection missing sparse_vectors config")
        self.assertIn("bm25", sparse_config, "Sparse vector 'bm25' missing from collection config")
        self.assertNotIn("splade", sparse_config, "Legacy 'splade' vector must not exist in collection config")

    def test_chunk_insertion_with_uuid_and_non_uuid_ids(self) -> None:
        """Verify insert_chunks supports standard hex UUIDs and arbitrary string IDs."""
        dense_dim = 1024
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        hex_uuid = "a1b2c3d4-e5f6-4a1b-8c2d-3e4f5a6b7c8d"
        non_hex_id = "credit_report_chunk_999"
        numeric_id = "987654321"

        chunks = [
            {
                "id": hex_uuid,
                "text": "متن چانک اول",
                "dense_vector": make_deterministic_dense_vector(dense_dim, seed=1),
                "sparse_indices": [10, 20],
                "sparse_values": [1.5, 2.5],
            },
            {
                "id": non_hex_id,
                "text": "متن چانک دوم",
                "dense_vector": make_deterministic_dense_vector(dense_dim, seed=2),
                "bm25_indices": [30, 40],
                "bm25_values": [0.8, 1.2],
            },
            {
                "id": numeric_id,
                "text": "متن چانک سوم",
                "dense_vector": make_deterministic_dense_vector(dense_dim, seed=3),
                "sparse_indices": [50],
                "sparse_values": [3.0],
            },
        ]

        # Insertion must succeed without ValueError: badly formed hexadecimal UUID string
        insert_chunks(self.client, chunks, collection_name=self.collection_name)

        count_res = self.client.count(self.collection_name)
        self.assertEqual(count_res.count, 3, "All 3 chunks must be inserted successfully")

    def test_hybrid_search_payload_preservation(self) -> None:
        """Verify search_hybrid returns ScoredPoints with all payload fields intact."""
        dense_dim = 1024
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        chunk_data = {
            "id": "e4b3c2a1-0000-0000-0000-000000000001",
            "text": "قوانین صدور چک الکترونیکی و ثبت در سامانه پیچک",
            "dense_vector": make_deterministic_dense_vector(dense_dim, seed=7),
            "sparse_indices": [100, 200],
            "sparse_values": [1.0, 2.0],
            "doc_id": "doc_cheque_rule",
            "doc_title": "قوانین چک الکترونیک",
            "heading_path": "چک / الکترونیک / پیچک",
            "folder_hierarchy": ["قوانین", "چک"],
            "ordinal": 5,
        }
        insert_chunks(self.client, [chunk_data], collection_name=self.collection_name)

        results = search_hybrid(
            client=self.client,
            query_dense=make_deterministic_dense_vector(dense_dim, seed=7),
            query_sparse_indices=[100],
            query_sparse_values=[1.0],
            top_k=1,
            collection_name=self.collection_name,
        )

        self.assertEqual(len(results), 1)
        payload = results[0].payload
        self.assertEqual(payload["chunk_id"], "e4b3c2a1-0000-0000-0000-000000000001")
        self.assertEqual(payload["text"], chunk_data["text"])
        self.assertEqual(payload["doc_id"], "doc_cheque_rule")
        self.assertEqual(payload["doc_title"], "قوانین چک الکترونیک")
        self.assertEqual(payload["heading_path"], "چک / الکترونیک / پیچک")
        self.assertEqual(payload["folder_hierarchy"], ["قوانین", "چک"])
        self.assertEqual(payload["ordinal"], 5)


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================

class TestTier2BoundaryAndCornerCases(BaseBM25QdrantTestCase):
    """
    Tier 2: Boundary and corner cases:
    - Empty strings & whitespace-only inputs
    - Single token documents & queries
    - Out-of-vocabulary terms
    - Repeated words and term frequency scaling
    - Half-spaces / Zero-Width Non-Joiner (ZWNJ \u200c)
    - Persian numbers, punctuation, mixed English
    """

    def test_empty_string_document_and_query(self) -> None:
        """Verify empty strings produce empty sparse vectors without exceptions."""
        doc_idx, doc_val = self.encoder.encode_document("")
        query_idx, query_val = self.encoder.encode_query("")

        self.assertEqual(doc_idx, [], "Empty document must produce empty indices")
        self.assertEqual(doc_val, [], "Empty document must produce empty values")
        self.assertEqual(query_idx, [], "Empty query must produce empty indices")
        self.assertEqual(query_val, [], "Empty query must produce empty values")

    def test_whitespace_only_text(self) -> None:
        """Verify whitespace/newlines/tabs produce empty sparse vectors."""
        whitespace_text = "   \t\n  \r\n   "
        doc_idx, doc_val = self.encoder.encode_document(whitespace_text)
        query_idx, query_val = self.encoder.encode_query(whitespace_text)

        self.assertEqual(doc_idx, [])
        self.assertEqual(doc_val, [])
        self.assertEqual(query_idx, [])
        self.assertEqual(query_val, [])

    def test_empty_sparse_vector_insertion_and_search(self) -> None:
        """Verify Qdrant handles points and queries with empty sparse vectors."""
        dense_dim = 1024
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        chunk = {
            "id": "00000000-0000-0000-0000-000000000001",
            "text": "...",
            "dense_vector": make_deterministic_dense_vector(dense_dim, seed=9),
            "sparse_indices": [],
            "sparse_values": [],
        }

        # Inserting empty sparse vector must succeed
        insert_chunks(self.client, [chunk], collection_name=self.collection_name)

        # Searching with empty sparse vector must succeed (falling back gracefully)
        results = search_hybrid(
            client=self.client,
            query_dense=make_deterministic_dense_vector(dense_dim, seed=9),
            query_sparse_indices=[],
            query_sparse_values=[],
            top_k=1,
            collection_name=self.collection_name,
        )
        self.assertIsInstance(results, list)

    def test_single_token_document_and_query(self) -> None:
        """Verify single token produces exactly 1 index and matching query index."""
        token = "اعتبارسنجی"
        doc_idx, doc_val = self.encoder.encode_document(token)
        query_idx, query_val = self.encoder.encode_query(token)

        self.assertEqual(len(doc_idx), 1)
        self.assertEqual(len(doc_val), 1)
        self.assertEqual(len(query_idx), 1)
        self.assertEqual(len(query_val), 1)

        self.assertEqual(doc_idx[0], query_idx[0], "Single token doc and query indices must match")
        self.assertGreater(doc_val[0], 0.0)
        self.assertGreater(query_val[0], 0.0)

    def test_out_of_vocabulary_query(self) -> None:
        """Verify queries with completely unindexed tokens execute cleanly."""
        dense_dim = 1024
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        doc_text = "تسهیلات خرید خودرو و وام قرض‌الحسنه بانک ملی"
        doc_idx, doc_val = self.encoder.encode_document(doc_text)
        d_vec = make_deterministic_dense_vector(dense_dim, seed=4)

        insert_chunks(
            self.client,
            [
                {
                    "id": "12345678-1234-1234-1234-123456789abc",
                    "text": doc_text,
                    "dense_vector": d_vec,
                    "sparse_indices": doc_idx,
                    "sparse_values": doc_val,
                }
            ],
            collection_name=self.collection_name,
        )

        # Query with terms that have zero overlap with doc
        oov_query = "کهکشان راه شیری فضاپیمای آپولو تلسکوپ جیمز وب"
        q_idx, q_val = self.encoder.encode_query(oov_query)

        # Execute hybrid search with orthogonal dense vector and OOV sparse vector
        results = search_hybrid(
            client=self.client,
            query_dense=make_orthogonal_dense_vector(d_vec, seed=88),
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=5,
            collection_name=self.collection_name,
        )

        # Must execute without exception
        self.assertIsInstance(results, list)

    def test_repeated_tokens_and_tf_weighting(self) -> None:
        """
        Verify:
        1. Repeated tokens do NOT produce duplicate indices in the sparse vector.
        2. Higher term frequency produces higher or equal weight compared to single occurrence.
        """
        text_single = "ضامن وام"
        text_repeated = "ضامن ضامن ضامن ضامن وام"

        idx_single, val_single = self.encoder.encode_document(text_single)
        idx_repeated, val_repeated = self.encoder.encode_document(text_repeated)

        self.assert_sparse_vector_valid(idx_single, val_single)
        self.assert_sparse_vector_valid(idx_repeated, val_repeated)

        # Unique indices invariant
        self.assertEqual(len(idx_repeated), len(set(idx_repeated)), "Indices must be unique even with repeated tokens!")

        # Find index for 'ضامن'
        token_zamen_idx, _ = self.encoder.encode_query("ضامن")
        zamen_idx = token_zamen_idx[0]

        pos_single = idx_single.index(zamen_idx)
        pos_repeated = idx_repeated.index(zamen_idx)

        val_s = val_single[pos_single]
        val_r = val_repeated[pos_repeated]

        self.assertGreaterEqual(
            val_r,
            val_s,
            f"Repeated token TF weight ({val_r}) should be >= single occurrence weight ({val_s})",
        )

    def test_persian_zwnj_half_spaces(self) -> None:
        """Verify words with Zero-Width Non-Joiner (\\u200c) are handled robustly."""
        # Compound words with ZWNJ: قرض‌الحسنه, سرمایه‌گذاری, بازپرداخت
        zwnj_text = "وام قرض‌الحسنه و سرمایه‌گذاری کوتاه‌مدت با بازپرداخت اقساط"
        indices, values = self.encoder.encode_document(zwnj_text)

        self.assert_sparse_vector_valid(indices, values)

        # Querying with exact ZWNJ compound word must match
        q_idx, _ = self.encoder.encode_query("قرض‌الحسنه")
        self.assertGreater(len(q_idx), 0)
        self.assertTrue(any(idx in indices for idx in q_idx), "ZWNJ compound term index must match document index")

    def test_persian_digits_and_numbers(self) -> None:
        """Verify Persian digits (۰۱۲۳۴۵۶۷۸۹) are processed cleanly."""
        text_with_digits = "تسهیلات ۲۰۰ میلیون تومانی در سال ۱۴۰۳ با سود ۱۸ درصد"
        indices, values = self.encoder.encode_document(text_with_digits)

        self.assert_sparse_vector_valid(indices, values)
        q_idx, _ = self.encoder.encode_query("۱۴۰۳")
        self.assertGreater(len(q_idx), 0)
        self.assertTrue(any(idx in indices for idx in q_idx))

    def test_mixed_persian_english_and_punctuation(self) -> None:
        """Verify punctuation removal and handling of mixed Persian/English characters."""
        text = "طرح VIP بانک ملی ایران: تسهیلات آنلاین (Online Loan)؛ بدون ضامن، با سود ۴٪!"
        indices, values = self.encoder.encode_document(text)

        self.assert_sparse_vector_valid(indices, values)

        # Check English tokens
        q_idx_vip, _ = self.encoder.encode_query("VIP")
        if len(q_idx_vip) > 0:
            self.assertTrue(any(idx in indices for idx in q_idx_vip))


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================

class TestTier3CrossFeatureCombinations(BaseBM25QdrantTestCase):
    """
    Tier 3: Cross-feature combinations:
    - Varying top_k limits (top_k=1, top_k > total points)
    - Dense-dominant vs sparse-dominant ranking balance in RRF
    - Multiple collection recreation and re-insertion cycles
    - Multi-chunk batch insertion integrity
    """

    def test_varying_top_k_limits(self) -> None:
        """Verify search_hybrid correctly bounds result count by min(top_k, total_points)."""
        dense_dim = 512
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        # Insert 5 distinct chunks
        chunks = []
        for i in range(5):
            t = f"متن سند بانکی شماره {i} برای تست محدودیت تعداد نتایج"
            idx, val = self.encoder.encode_document(t)
            chunks.append(
                {
                    "id": f"00000000-0000-0000-0000-00000000000{i}",
                    "text": t,
                    "dense_vector": make_deterministic_dense_vector(dense_dim, seed=i + 1),
                    "sparse_indices": idx,
                    "sparse_values": val,
                }
            )
        insert_chunks(self.client, chunks, collection_name=self.collection_name)

        q_idx, q_val = self.encoder.encode_query("سند بانکی")
        q_dense = make_deterministic_dense_vector(dense_dim, seed=1)

        # top_k = 1 -> exactly 1
        res_1 = search_hybrid(self.client, q_dense, q_idx, q_val, top_k=1, collection_name=self.collection_name)
        self.assertEqual(len(res_1), 1)

        # top_k = 3 -> exactly 3
        res_3 = search_hybrid(self.client, q_dense, q_idx, q_val, top_k=3, collection_name=self.collection_name)
        self.assertEqual(len(res_3), 3)

        # top_k = 5 -> exactly 5
        res_5 = search_hybrid(self.client, q_dense, q_idx, q_val, top_k=5, collection_name=self.collection_name)
        self.assertEqual(len(res_5), 5)

        # top_k = 25 (exceeding total points) -> exactly 5
        res_25 = search_hybrid(self.client, q_dense, q_idx, q_val, top_k=25, collection_name=self.collection_name)
        self.assertEqual(len(res_25), 5)

    def test_dense_dominant_vs_sparse_dominant_ranking(self) -> None:
        """
        Verify RRF fusion combines dense and sparse signals:
        - Doc Sparse: High BM25 keyword overlap, low dense similarity.
        - Doc Dense: Low keyword overlap, high dense similarity.
        Both should appear in fused RRF results with positive scores.
        """
        dense_dim = 1024
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        query_text = "وام ازدواج"
        query_dense = make_deterministic_dense_vector(dense_dim, seed=100)
        q_idx, q_val = self.encoder.encode_query(query_text)

        # Doc A: Exact keyword match ('وام ازدواج'), but orthogonal dense vector
        doc_a_text = "شرایط دریافت وام ازدواج بانک مرکزی"
        idx_a, val_a = self.encoder.encode_document(doc_a_text)
        dense_a = make_orthogonal_dense_vector(query_dense, seed=10)

        # Doc B: No keyword match, but high dense similarity
        doc_b_text = "تسهیلات پیوند زناشویی جوانان کشور"
        idx_b, val_b = self.encoder.encode_document(doc_b_text)
        dense_b = make_similar_dense_vector(query_dense, noise_level=0.01, seed=20)

        chunks = [
            {
                "id": "10000000-0000-0000-0000-000000000001",
                "text": doc_a_text,
                "dense_vector": dense_a,
                "sparse_indices": idx_a,
                "sparse_values": val_a,
                "type": "sparse_dominant",
            },
            {
                "id": "20000000-0000-0000-0000-000000000002",
                "text": doc_b_text,
                "dense_vector": dense_b,
                "sparse_indices": idx_b,
                "sparse_values": val_b,
                "type": "dense_dominant",
            },
        ]
        insert_chunks(self.client, chunks, collection_name=self.collection_name)

        results = search_hybrid(
            client=self.client,
            query_dense=query_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=2,
            collection_name=self.collection_name,
        )

        self.assertEqual(len(results), 2, "Both sparse and dense matches must be returned by RRF")
        retrieved_ids = {p.id for p in results}
        self.assertIn("10000000-0000-0000-0000-000000000001", retrieved_ids)
        self.assertIn("20000000-0000-0000-0000-000000000002", retrieved_ids)

        # Scores must be positive and non-zero
        for p in results:
            self.assertGreater(p.score, 0.0)

    def test_recreate_collection_idempotency_and_isolation(self) -> None:
        """Verify recreating a collection empties previous points cleanly."""
        dense_dim = 256
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)

        chunks_1 = [
            {
                "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "text": "سند اول",
                "dense_vector": make_deterministic_dense_vector(dense_dim, seed=1),
                "sparse_indices": [1],
                "sparse_values": [1.0],
            }
        ]
        insert_chunks(self.client, chunks_1, collection_name=self.collection_name)
        self.assertEqual(self.client.count(self.collection_name).count, 1)

        # Recreate collection
        recreate_collection(self.client, dense_dim=dense_dim, collection_name=self.collection_name)
        self.assertEqual(self.client.count(self.collection_name).count, 0)

        chunks_2 = [
            {
                "id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "text": "سند دوم",
                "dense_vector": make_deterministic_dense_vector(dense_dim, seed=2),
                "sparse_indices": [2],
                "sparse_values": [1.0],
            },
            {
                "id": "cccccccc-cccc-cccc-cccc-cccccccccccc",
                "text": "سند سوم",
                "dense_vector": make_deterministic_dense_vector(dense_dim, seed=3),
                "sparse_indices": [3],
                "sparse_values": [1.0],
            },
        ]
        insert_chunks(self.client, chunks_2, collection_name=self.collection_name)
        self.assertEqual(self.client.count(self.collection_name).count, 2)


# ==============================================================================
# Tier 4: Real-World Persian Banking / Credit Loan Scenarios
# ==============================================================================

class TestTier4RealWorldPersianBankingScenarios(BaseBM25QdrantTestCase):
    """
    Tier 4: End-to-end evaluation with realistic Persian banking & credit scoring data:
    - Multi-chunk knowledge base indexing
    - Domain-specific queries (marriage loan, credit bureau scoring, mortgage installments,
      bounced cheques, collateral-free loans)
    - Verifying rank 1 retrieval accuracy, index alignment, and RRF score ordering.
    """

    def setUp(self) -> None:
        super().setUp()
        self.dense_dim = 1024
        recreate_collection(self.client, dense_dim=self.dense_dim, collection_name=self.collection_name)

        # Index the knowledge base chunks with real BM25 sparse vectors
        self.chunk_records: Dict[str, Dict[str, Any]] = {}
        chunks_to_insert: List[Dict[str, Any]] = []

        for idx_num, item in enumerate(REALISTIC_BANKING_CHUNKS, start=1):
            text = item["text"]
            sparse_idx, sparse_val = self.encoder.encode_document(text)
            dense_vec = make_deterministic_dense_vector(dim=self.dense_dim, seed=idx_num * 100)

            chunk_payload = dict(item)
            chunk_payload["dense_vector"] = dense_vec
            chunk_payload["bm25_indices"] = sparse_idx
            chunk_payload["bm25_values"] = sparse_val

            chunks_to_insert.append(chunk_payload)
            self.chunk_records[item["id"]] = chunk_payload

        insert_chunks(self.client, chunks_to_insert, collection_name=self.collection_name)

    def test_marriage_loan_query_ranking(self) -> None:
        """Target query for Marriage Loan must retrieve chunk_marriage_loan_001 at rank 1."""
        target_id = "chunk_marriage_loan_001"
        query_text = "شرایط و مدارک وام قرض‌الحسنه ازدواج ضامن معتبر"

        q_idx, q_val = self.encoder.encode_query(query_text)
        target_dense = self.chunk_records[target_id]["dense_vector"]
        q_dense = make_similar_dense_vector(target_dense, noise_level=0.02, seed=11)

        results = search_hybrid(
            client=self.client,
            query_dense=q_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=3,
            collection_name=self.collection_name,
        )

        self.assertGreater(len(results), 0)
        # Top result must match target chunk
        top_hit = results[0]
        self.assertEqual(top_hit.payload["chunk_id"], target_id)
        self.assertEqual(top_hit.payload["doc_title"], "دستورالعمل تسهیلات قرض‌الحسنه ازدواج")

    def test_credit_bureau_scoring_query_ranking(self) -> None:
        """Target query for Credit Bureau Scoring must retrieve chunk_credit_scoring_002 at rank 1."""
        target_id = "chunk_credit_scoring_002"
        query_text = "استعلام نمره اعتباری و رتبه اعتبارسنجی در سامانه ملی"

        q_idx, q_val = self.encoder.encode_query(query_text)
        target_dense = self.chunk_records[target_id]["dense_vector"]
        q_dense = make_similar_dense_vector(target_dense, noise_level=0.02, seed=22)

        results = search_hybrid(
            client=self.client,
            query_dense=q_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=3,
            collection_name=self.collection_name,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].payload["chunk_id"], target_id)
        self.assertEqual(results[0].payload["doc_title"], "راهنمای استعلام گزارش اعتبارسنجی")

    def test_housing_loan_query_ranking(self) -> None:
        """Target query for Housing Loan must retrieve chunk_housing_loan_003 at rank 1."""
        target_id = "chunk_housing_loan_003"
        query_text = "محاسبه اقساط ماهانه و سود تسهیلات خرید مسکن اوراق ممتاز"

        q_idx, q_val = self.encoder.encode_query(query_text)
        target_dense = self.chunk_records[target_id]["dense_vector"]
        q_dense = make_similar_dense_vector(target_dense, noise_level=0.02, seed=33)

        results = search_hybrid(
            client=self.client,
            query_dense=q_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=3,
            collection_name=self.collection_name,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].payload["chunk_id"], target_id)
        self.assertEqual(results[0].payload["doc_title"], "تسهیلات خرید و ساخت مسکن")

    def test_bounced_cheque_query_ranking(self) -> None:
        """Target query for Bounced Cheque Clearing must retrieve chunk_bounced_cheque_005 at rank 1."""
        target_id = "chunk_bounced_cheque_005"
        query_text = "مراحل رفع سوء اثر از چک برگشتی صیادی و مسدودی حساب جاری"

        q_idx, q_val = self.encoder.encode_query(query_text)
        target_dense = self.chunk_records[target_id]["dense_vector"]
        q_dense = make_similar_dense_vector(target_dense, noise_level=0.02, seed=44)

        results = search_hybrid(
            client=self.client,
            query_dense=q_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=3,
            collection_name=self.collection_name,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].payload["chunk_id"], target_id)
        self.assertEqual(results[0].payload["doc_title"], "قوانین رفع سوء اثر از چک‌های برگشتی")

    def test_microloan_no_guarantor_query_ranking(self) -> None:
        """Target query for Collateral-Free Microloan must retrieve chunk_microloan_006 at rank 1."""
        target_id = "chunk_microloan_006"
        query_text = "تسهیلات خرد بانکی بدون ضامن با سفته الکترونیک و رتبه اعتباری"

        q_idx, q_val = self.encoder.encode_query(query_text)
        target_dense = self.chunk_records[target_id]["dense_vector"]
        q_dense = make_similar_dense_vector(target_dense, noise_level=0.02, seed=55)

        results = search_hybrid(
            client=self.client,
            query_dense=q_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=3,
            collection_name=self.collection_name,
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].payload["chunk_id"], target_id)
        self.assertEqual(results[0].payload["doc_title"], "تسهیلات خرد بدون ضامن")

    def test_rrf_scores_are_sorted_descending(self) -> None:
        """Verify that RRF hybrid scores are strictly ordered descending."""
        query_text = "وام و تسهیلات بانکی اعتبارسنجی"
        q_idx, q_val = self.encoder.encode_query(query_text)
        q_dense = make_deterministic_dense_vector(self.dense_dim, seed=777)

        results = search_hybrid(
            client=self.client,
            query_dense=q_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=5,
            collection_name=self.collection_name,
        )

        self.assertGreater(len(results), 1)
        for i in range(len(results) - 1):
            score_curr = results[i].score
            score_next = results[i + 1].score
            self.assertGreaterEqual(
                score_curr,
                score_next,
                f"Result scores not in descending order: {score_curr} < {score_next}",
            )


# ==============================================================================
# Standalone Acceptance Verification Runner
# ==============================================================================

def run_acceptance_demo() -> bool:
    """
    Executes a direct demonstration of the acceptance criteria:
    - In-memory collection creation
    - Chunk vectorization & insertion
    - Hybrid RRF search
    - Index alignment verification
    Prints human-readable diagnostics.
    """
    print("\n" + "=" * 70)
    print("  BM25 QDRANT MIGRATION: STANDALONE ACCEPTANCE VERIFICATION")
    print("=" * 70)

    if not QDRANT_CLIENT_AVAILABLE:
        print("[-] FAILED: qdrant-client package is not installed.")
        return False

    if not BM25_AVAILABLE:
        print(f"[-] FAILED: bm25_sparse module could not be imported: {BM25_IMPORT_ERROR}")
        return False

    if not QDRANT_HYBRID_AVAILABLE:
        print(f"[-] FAILED: qdrant_hybrid module could not be imported: {QDRANT_HYBRID_IMPORT_ERROR}")
        return False

    try:
        # Step 1: Initialize client in :memory: mode
        print("[1/5] Initializing QdrantClient in :memory: mode...")
        client = get_qdrant_client(location=":memory:")
        demo_col = f"demo_col_{uuid.uuid4().hex[:6]}"
        dense_dim = 1024
        print(f"      -> Created in-memory client. Target collection: '{demo_col}'")

        # Step 2: Recreate collection with dense and bm25 sparse vectors
        print("[2/5] Recreating collection schema (dense: 1024, sparse: 'bm25')...")
        recreate_collection(client, dense_dim=dense_dim, collection_name=demo_col)
        col_info = client.get_collection(demo_col)
        assert "dense" in col_info.config.params.vectors, "dense vector config missing"
        assert "bm25" in col_info.config.params.sparse_vectors, "bm25 sparse vector config missing"
        print("      -> Collection created successfully with native BM25 sparse index.")

        # Step 3: Vectorize and insert chunks
        print("[3/5] Vectorizing and inserting Persian banking chunks...")
        encoder = PersianBM25Encoder()

        sample_chunks = [
            {
                "id": "c1000000-0000-0000-0000-000000000001",
                "text": "شرایط دریافت وام قرض‌الحسنه ازدواج و مدارک ضامنین در سال ۱۴۰۳",
                "doc_title": "وام قرض‌الحسنه ازدواج",
            },
            {
                "id": "c2000000-0000-0000-0000-000000000002",
                "text": "دستورالعمل نحوه دریافت گزارش اعتبارسنجی و رتبه اعتباری در سامانه ملی",
                "doc_title": "استعلام گزارش اعتبارسنجی",
            },
            {
                "id": "c3000000-0000-0000-0000-000000000003",
                "text": "سقف تسهیلات خرید مسکن و نحوه محاسبه اقساط ماهیانه و سود دوران مشارکت",
                "doc_title": "تسهیلات مسکن",
            },
        ]

        chunks_to_insert = []
        for i, sc in enumerate(sample_chunks, start=1):
            idx, val = encoder.encode_document(sc["text"])
            d_vec = make_deterministic_dense_vector(dim=dense_dim, seed=i * 50)
            chunks_to_insert.append(
                {
                    "id": sc["id"],
                    "text": sc["text"],
                    "dense_vector": d_vec,
                    "bm25_indices": idx,
                    "bm25_values": val,
                    "doc_title": sc["doc_title"],
                }
            )

        insert_chunks(client, chunks_to_insert, collection_name=demo_col)
        print(f"      -> Successfully inserted {len(chunks_to_insert)} points with dual vectors.")

        # Step 4: Verify index alignment
        print("[4/5] Verifying query vs document token index alignment...")
        query_text = "وام قرض‌الحسنه ازدواج ضامن"
        q_idx, q_val = encoder.encode_query(query_text)

        # Doc 1 indices
        doc1_idx = chunks_to_insert[0]["bm25_indices"]
        common_indices = set(q_idx).intersection(set(doc1_idx))
        print(f"      Query tokens: {query_text}")
        print(f"      Query sparse indices: {q_idx}")
        print(f"      Doc 1 sparse indices: {doc1_idx[:6]}... (total {len(doc1_idx)})")
        print(f"      Matching common indices: {common_indices}")
        assert len(common_indices) >= 2, "Expected at least 2 common token indices between query and Doc 1"
        print("      -> Index alignment verified: Query tokens map to identical indices as chunk tokens.")

        # Step 5: Execute hybrid RRF search
        print("[5/5] Executing native RRF hybrid search (Prefetch + FusionQuery)...")
        q_dense = make_similar_dense_vector(chunks_to_insert[0]["dense_vector"], noise_level=0.01, seed=77)
        results = search_hybrid(
            client=client,
            query_dense=q_dense,
            query_sparse_indices=q_idx,
            query_sparse_values=q_val,
            top_k=3,
            collection_name=demo_col,
        )

        print(f"      -> Hybrid search returned {len(results)} ranked points:")
        for rank, pt in enumerate(results, start=1):
            print(f"         Rank {rank}: [Score: {pt.score:.6f}] {pt.payload.get('doc_title')} - {pt.payload.get('text')[:50]}...")

        assert len(results) > 0, "No results returned"
        assert results[0].payload["doc_title"] == "وام قرض‌الحسنه ازدواج", "Top result should be Doc 1"
        print("\n[+] SUCCESS: All acceptance criteria verified successfully!\n")
        return True

    except Exception as exc:
        print(f"\n[-] ERROR during acceptance verification: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    """Main entrypoint for standalone test execution."""
    print("Executing BM25 Qdrant Test Suite...")

    # First run the acceptance verification demo
    acceptance_ok = run_acceptance_demo()

    print("Running Unittest Suite (Tiers 1 - 4 & Acceptance)...")
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestAcceptanceCriteriaVerification))
    suite.addTests(loader.loadTestsFromTestCase(TestTier1FeatureCoverage))
    suite.addTests(loader.loadTestsFromTestCase(TestTier2BoundaryAndCornerCases))
    suite.addTests(loader.loadTestsFromTestCase(TestTier3CrossFeatureCombinations))
    suite.addTests(loader.loadTestsFromTestCase(TestTier4RealWorldPersianBankingScenarios))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)

    print("\n" + "=" * 70)
    print("  TEST EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed:         {total_tests - failures - errors - skipped}")
    print(f"Failures:       {failures}")
    print(f"Errors:         {errors}")
    print(f"Skipped:        {skipped}")
    print(f"Acceptance Demo:{'PASS' if acceptance_ok else 'FAIL'}")
    print("=" * 70)

    if not result.wasSuccessful() or not acceptance_ok:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
