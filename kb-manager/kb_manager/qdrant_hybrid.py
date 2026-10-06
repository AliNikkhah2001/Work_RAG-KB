import os
import uuid
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
    Prefetch,
    QueryRequest,
    FusionQuery,
)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "credit_rag_chunks"

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def recreate_collection(client: QdrantClient, dense_dim: int = 1024):
    """Recreate Qdrant collection with both dense and sparse vectors."""
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=dense_dim,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "splade": SparseVectorParams()
        }
    )

def insert_chunks(client: QdrantClient, chunks: List[Dict[str, Any]]):
    """
    chunks format:
    [
        {
            "id": "chunk_uuid",
            "text": "chunk text",
            "doc_id": "...",
            "doc_title": "...",
            "heading_path": "...",
            "folder_hierarchy": [...],
            "dense_vector": [float, ...],
            "sparse_indices": [int, ...],
            "sparse_values": [float, ...]
        }
    ]
    """
    points = []
    for chunk in chunks:
        points.append(PointStruct(
            id=str(uuid.UUID(chunk["id"])),
            vector={
                "dense": chunk["dense_vector"],
                "splade": SparseVector(
                    indices=chunk["sparse_indices"],
                    values=chunk["sparse_values"]
                )
            },
            payload={
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "doc_id": chunk.get("doc_id"),
                "doc_title": chunk.get("doc_title"),
                "heading_path": chunk.get("heading_path"),
                "folder_hierarchy": chunk.get("folder_hierarchy", []),
                "ordinal": chunk.get("ordinal", 0)
            }
        ))
        
    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=100
    )

def search_hybrid(client: QdrantClient, query_dense: List[float], query_sparse_indices: List[int], query_sparse_values: List[float], top_k: int = 10):
    """Perform hybrid search using Reciprocal Rank Fusion (RRF) natively in Qdrant."""
    results = client.query_points(
        collection_name=COLLECTION_NAME,
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
                using="splade",
                limit=top_k * 3,
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k,
        with_payload=True
    )
    return results.points
