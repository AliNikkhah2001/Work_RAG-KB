"""Evaluation package removed — tests live in Work_RAG-Evaluation_Suite."""

import pytest


@pytest.mark.skip(reason="kb_manager.evaluation removed; use components/evaluation")
def test_evaluation_moved():
    assert False
