import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
import numpy as np
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redis.asyncio import Redis

logger = logging.getLogger("semantic_cache")

# Schema for Semantic Caching
CACHE_SCHEMA = {
    "index": {
        "name": "query_cache",
        "prefix": "cache",
        "storage_type": "hash"
    },
    "fields": [
        {"name": "query", "type": "text"},
        {"name": "response_json", "type": "text"},
        {
            "name": "query_vector",
            "type": "vector",
            "attrs": {
                "dims": 1024,
                "distance_metric": "cosine",
                "algorithm": "flat",
                "datatype": "float32"
            }
        }
    ]
}

class SemanticCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.client = Redis.from_url(redis_url)
        self.schema = IndexSchema.from_dict(CACHE_SCHEMA)
        self.index = SearchIndex(self.schema, self.client)
        self._initialized = False

    async def initialize(self):
        try:
            await self.index.create(overwrite=False)
            self._initialized = True
            logger.info("Semantic cache initialized.")
        except Exception as e:
            logger.warning(f"Semantic cache initialization failed: {e}")

    async def check(self, query: str, query_vector: List[float], threshold: float = 0.95) -> Optional[List[Dict[str, Any]]]:
        if not self._initialized:
            return None
            
        from redisvl.query import VectorQuery
        
        vector_query = VectorQuery(
            vector=query_vector,
            vector_field_name="query_vector",
            return_fields=["response_json"],
            num_results=1
        )
        
        try:
            results = await self.index.query(vector_query)
            if results and len(results) > 0:
                # Calculate cosine similarity from distance
                # Cosine distance = 1 - Cosine Similarity. We want similarity > 0.95 (distance < 0.05)
                dist = float(results[0]["vector_distance"])
                if dist < (1.0 - threshold):
                    logger.info(f"Cache HIT! distance: {dist:.4f}")
                    return json.loads(results[0]["response_json"])
        except Exception as e:
            logger.error(f"Semantic cache search error: {e}")
            
        return None

    async def insert(self, query: str, query_vector: List[float], response_chunks: List[Dict[str, Any]]):
        if not self._initialized:
            return
            
        try:
            data = {
                "query": query,
                "query_vector": np.array(query_vector, dtype=np.float32).tobytes(),
                "response_json": json.dumps(response_chunks, ensure_ascii=False)
            }
            # Write to index
            await self.index.load([data])
        except Exception as e:
            logger.error(f"Semantic cache insert error: {e}")

# Global instance
_cache = None

def get_semantic_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        import os
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _cache = SemanticCache(redis_url=redis_url)
    return _cache
