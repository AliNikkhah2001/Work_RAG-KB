"""Unit checks for DVC snapshot helpers (no database)."""

from kb_manager.dvc_snapshot import _vector_dir_name, parse_embedding


def test_parse_embedding_pgvector_text():
    assert parse_embedding("[0.5, 1, -2]") == [0.5, 1.0, -2.0]
    assert parse_embedding(None) is None
    assert parse_embedding([1, 2]) == [1.0, 2.0]


def test_vector_dir_name_replaces_slash():
    assert _vector_dir_name("sentence-transformers/x", 384) == "sentence-transformers__x-384"
