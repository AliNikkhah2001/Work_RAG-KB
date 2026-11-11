"""HTTP client for Work_RAG-Evaluation_Suite — sole evaluation backend."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_EVAL_URL = os.environ.get("EVAL_SUITE_URL", "http://127.0.0.1:8300")


class EvalSuiteClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or DEFAULT_EVAL_URL).rstrip("/")
        self.timeout = timeout

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            r = await client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    async def list_datasets(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            r = await client.get(f"{self.base_url}/v1/datasets")
            r.raise_for_status()
            return list((r.json() or {}).get("datasets") or [])

    async def run_hitk(
        self,
        *,
        dataset: str = "retrieval/eval_clean_remapped.json",
        top_k: int = 5,
        limit: int | None = None,
        remap_from: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"dataset": dataset, "top_k": top_k}
        if limit:
            body["limit"] = limit
        if remap_from:
            body["remap_from"] = remap_from
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            r = await client.post(f"{self.base_url}/v1/hitk/run", json=body)
            r.raise_for_status()
            return r.json()

    async def run_multiturn(
        self,
        *,
        conversations: str = "multiturn/conversations_v1.yaml",
        only: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"conversations": conversations}
        if only:
            body["only"] = only
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            r = await client.post(f"{self.base_url}/v1/multiturn/run", json=body)
            r.raise_for_status()
            return r.json()

    async def job(self, job_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            r = await client.get(f"{self.base_url}/v1/jobs/{job_id}")
            r.raise_for_status()
            return r.json()

    async def results(self, suite: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            r = await client.get(f"{self.base_url}/v1/results/{suite}")
            r.raise_for_status()
            return r.json()
