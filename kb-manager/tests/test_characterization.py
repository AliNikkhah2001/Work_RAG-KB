"""Phase 0 characterization — pins current behavior before any fix.

Requires `pytest -m "not models"` to stay offline; mocked MiniLM vectors.
If these assertions fail after a Phase 1+ change, the benchmark is intentionally invalidated (expected).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from kb_manager.chunker.semantic import SemanticChunker
from kb_manager.web.routes.search import _tokenize, BM25


def test_dataset_checksum_frozen():
    p = pathlib.Path("data/test_questions.json")
    sha = pathlib.Path("data/test_questions.sha256").read_text(encoding="utf-8").split()[0]
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == sha, "dataset changed without version bump — see REMEDIATION_PLAN.md Phase 0"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data) == 120
    counts = {f: sum(1 for x in data if x.get("format") == f) for f in ["verbatim","paraphrase","reworded","keyword_only","typo","conversational"]}
    assert all(v == 20 for v in counts.values())


def test_tokenize_zwnj_normalized():
    # ZWNJ currently mapped to space before regex — char class still contains \u200c but dead
    tokens = _tokenize("می‌خواهم کتاب را بخوانم")
    # should not contain ZWNJ, should split on it
    assert all("\u200c" not in t for t in tokens)


def test_bm25_keyword_boost_length_dependent():
    # Documents differing only by keyword dup currently change dl — will become field-weighted in Phase 4
    bm = BM25()
    bm.index([("d1", "گزارش اعتباری وام"), ("d2", "گزارش اعتباری وام گزارش اعتباری وام گزارش اعتباری وام گزارش اعتباری وام")])
    assert bm.doc_lens[1] > bm.doc_lens[0]


def test_semantic_chunker_dedup_state_persists():
    # Characterizes F36: _seen_questions persists across documents if same instance reused
    c = SemanticChunker(dedup_questions=True)
    sheets_a = [{"name":"s","schema":"crm_qa","headers":["question","answer"],"rows":[["سلام","پاسخ"]]}]
    sheets_b = [{"name":"s","schema":"crm_qa","headers":["question","answer"],"rows":[["سلام","پاسخ"]]}]
    a = [x for x in c.chunk("", metadata={"doc_type":"qa_pair","sheets":sheets_a}) if not x.metadata.get("is_parent")]
    b = [x for x in c.chunk("", metadata={"doc_type":"qa_pair","sheets":sheets_b}) if not x.metadata.get("is_parent")]
    # second call currently dedupes across documents — documents share instance state
    assert len(a) == 1 and len(b) == 0, "dedup state leaks across documents (expected pre-fix behavior)"


def test_reranker_dtype_default_is_float16_on_cpu_bug():
    # Characterizes F11: device=None currently picks float16
    from kb_manager.reranker import CrossEncoderReranker
    r = CrossEncoderReranker(model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=None)
    # dtype selection is internal; we assert the condition that causes the bug
    assert r._device is None  # triggers float16 path in _ensure_model
