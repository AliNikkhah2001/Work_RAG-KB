"""Bounded TTL score cache for the cross-encoder reranker.

The reranker memoizes (query, passage) -> relevance-score pairs so repeated
searches skip GPU re-scoring of already-seen pairs. Keys are
``(sha1(query), sha1(passage))`` tuples; values are float scores.

Semantics contract (see ``kb_manager.reranker.CrossEncoderReranker``):

* Scores are deterministic for identical (query, passage) input, so serving
  a cached value can never change rerank ordering.
* The TTL defaults to 600s to stay consistent with the search-route result
  cache (``KB_REDIS_TTL``); expiry only falls back to recomputation.
* The maxsize (default 8192) bounds memory via LRU eviction of the
  least-recently-used entries.

This module is intentionally stdlib-only so it can be imported at module
load time without touching torch/transformers.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional

__all__ = ["TTLCache"]


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL, keyed by arbitrary hashables.

    Not a full Mapping: only ``get``/``put``/``clear``/``__len__`` plus hit
    and miss counters (for diagnostics). Expiry is checked lazily on
    ``get``; entries therefore may briefly outlive their TTL in memory but
    are never *served* after expiry.
    """

    def __init__(self, maxsize: int = 8192, ttl: float = 600.0) -> None:
        self._maxsize = max(1, int(maxsize))
        self._ttl = float(ttl)
        # key -> (expires_at_monotonic, value)
        self._data: "OrderedDict[Any, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        """Return the cached value for ``key`` or None on miss/expiry."""
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at <= now:
                # Expired: drop it so it does not count against maxsize.
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)  # LRU refresh
            self.hits += 1
            return value

    def put(self, key: Any, value: Any) -> None:
        """Insert/refresh ``key`` and evict the oldest entry over maxsize."""
        with self._lock:
            self._data[key] = (time.monotonic() + self._ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __repr__(self) -> str:
        return (
            f"TTLCache(size={len(self)}, maxsize={self._maxsize}, "
            f"ttl={self._ttl}s, hits={self.hits}, misses={self.misses})"
        )
