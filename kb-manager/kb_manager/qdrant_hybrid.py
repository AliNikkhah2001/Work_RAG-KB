"""
qdrant_hybrid.py - Hybrid Qdrant client integration with native BM25 sparse vectors and RRF fusion.
"""

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

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

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "credit_rag_chunks"
SPARSE_VECTOR_NAME = "bm25"


def get_qdrant_client(location: Optional[str] = None) -> QdrantClient:
    """Return Qdrant client, supporting in-memory mode or remote host."""
    loc = location or os.getenv("QDRANT_LOCATION")
    if loc:
        return QdrantClient(location=loc)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def recreate_collection(
    client: QdrantClient,
    dense_dim: int = 1024,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Recreate Qdrant collection with both dense and BM25 sparse vectors."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams()
        },
    )


def insert_chunks(
    client: QdrantClient,
    chunks: List[Dict[str, Any]],
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    Insert chunks with dense and BM25 sparse vectors under the same point ID.

    chunks format:
    [
        {
            "id": "chunk_uuid_or_str",
            "text": "chunk text",
            "doc_id": "...",
            "doc_title": "...",
            "heading_path": "...",
            "folder_hierarchy": [...],
            "dense_vector": [float, ...],
            "bm25_indices": [int, ...],  # or "sparse_indices"
            "bm25_values": [float, ...]   # or "sparse_values"
        }
    ]
    """
    if not chunks:
        return

    points = []
    for chunk in chunks:
        raw_id = chunk["id"]
        try:
            point_id = str(uuid.UUID(str(raw_id)))
        except (ValueError, AttributeError, TypeError):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))

        indices = (
            chunk.get("bm25_indices")
            if "bm25_indices" in chunk
            else chunk.get("sparse_indices", [])
        )
        values = (
            chunk.get("bm25_values")
            if "bm25_values" in chunk
            else chunk.get("sparse_values", [])
        )

        payload = {
            "chunk_id": str(raw_id),
            "text": chunk.get("text", ""),
            "doc_id": chunk.get("doc_id"),
            "doc_title": chunk.get("doc_title"),
            "heading_path": chunk.get("heading_path"),
            "folder_hierarchy": chunk.get("folder_hierarchy", []),
            "ordinal": chunk.get("ordinal", 0),
        }
        for k, v in chunk.items():
            if k not in (
                "id",
                "text",
                "doc_id",
                "doc_title",
                "heading_path",
                "folder_hierarchy",
                "ordinal",
                "dense_vector",
                "sparse_indices",
                "sparse_values",
                "bm25_indices",
                "bm25_values",
            ):
                payload[k] = v

        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense": chunk["dense_vector"],
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=indices,
                        values=values,
                    ),
                },
                payload=payload,
            )
        )

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=100,
    )


def search_hybrid(
    client: QdrantClient,
    query_dense: List[float],
    query_sparse_indices: List[int],
    query_sparse_values: List[float],
    top_k: int = 10,
    collection_name: str = COLLECTION_NAME,
) -> List[ScoredPoint]:
    """Perform hybrid search using Reciprocal Rank Fusion (RRF) natively in Qdrant."""
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_dense,
                using="dense",
                limit=top_k * 3,
            ),
            Prefetch(
                query=SparseVector(
                    indices=query_sparse_indices,
                    values=query_sparse_values,
                ),
                using=SPARSE_VECTOR_NAME,
                limit=top_k * 3,
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
        with_payload=True,
    )
    return results.points


def search_hybrid_text(
    client: QdrantClient,
    query_text: str,
    query_dense: List[float],
    encoder: Optional[Any] = None,
    top_k: int = 10,
    collection_name: str = COLLECTION_NAME,
) -> List[ScoredPoint]:
    """
    Helper function to encode query text using PersianBM25Encoder and execute native RRF hybrid search.
    """
    if encoder is None:
        from bm25_sparse import PersianBM25Encoder

        encoder = PersianBM25Encoder()
    query_sparse_indices, query_sparse_values = encoder.encode_query(query_text)
    return search_hybrid(
        client=client,
        query_dense=query_dense,
        query_sparse_indices=query_sparse_indices,
        query_sparse_values=query_sparse_values,
        top_k=top_k,
        collection_name=collection_name,
    )
