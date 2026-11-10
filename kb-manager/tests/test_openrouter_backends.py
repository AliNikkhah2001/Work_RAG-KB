"""Unit tests for OpenRouter embed + rerank backends (mocked HTTP)."""

from __future__ import annotations

import numpy as np
import pytest


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def test_openrouter_embedder_batches(monkeypatch):
    from kb_manager.embedder.openrouter import OpenRouterEmbedder

    calls: list[dict] = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            calls.append({"url": url, "json": json})
            n = len(json["input"])
            return _FakeResp(
                {
                    "data": [
                        {"index": i, "embedding": [float(i), 1.0, 0.0]}
                        for i in range(n)
                    ]
                }
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr("httpx.Client", _Client)

    emb = OpenRouterEmbedder(model_name="openai/text-embedding-3-small", dimensions=3, batch_size=2)
    vecs = emb.embed_texts(["a", "b", "c"])
    assert len(vecs) == 3
    assert len(calls) == 2
    assert calls[0]["url"].endswith("/embeddings")
    arr = emb.as_numpy_fn()(["x"])
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 1


def test_openrouter_reranker_orders_by_score(monkeypatch):
    from kb_manager.reranker import OpenRouterReranker

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            assert url.endswith("/rerank")
            return _FakeResp(
                {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ]
                }
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr("httpx.Client", _Client)

    rr = OpenRouterReranker(model_name="cohere/rerank-v3.5")
    out = rr.rerank(
        "q",
        [
            {"chunk_id": "a", "content_preview": "doc a", "hybrid_score": 0.5},
            {"chunk_id": "b", "content_preview": "doc b", "hybrid_score": 0.4},
        ],
        top_k=2,
    )
    assert [c["chunk_id"] for c in out] == ["b", "a"]
    assert out[0]["rerank_score"] == pytest.approx(0.9)


def test_get_reranker_openrouter_backend(monkeypatch):
    monkeypatch.setenv("KB_RERANKER_BACKEND", "openrouter")
    monkeypatch.setenv("KB_RERANKER_MODEL", "cohere/rerank-5-fast")
    from kb_manager.reranker import get_reranker, OpenRouterReranker

    rr = get_reranker()
    assert isinstance(rr, OpenRouterReranker)
    assert rr._model_name == "cohere/rerank-5-fast"
