"""Tests for versioned pack registry + schema detection + taxonomy filters."""

from __future__ import annotations

import json

import pytest

from kb_manager.registry import (
    PackNotFoundError,
    Registry,
    build_manifest,
    get_registry,
    reset_registry,
    save_manifest,
)
from kb_manager.registry.loader import PACKS_ROOT


@pytest.fixture(autouse=True)
def _reset():
    reset_registry()
    yield
    reset_registry()


def test_packs_root_exists():
    assert PACKS_ROOT.is_dir()
    assert (PACKS_ROOT / "schemas" / "v1.0.0.yaml").is_file()
    assert (PACKS_ROOT / "taxonomy" / "v1.0.0.yaml").is_file()
    assert (PACKS_ROOT / "canned" / "v1.0.0.yaml").is_file()


def test_list_and_load_schemas():
    reg = get_registry()
    versions = reg.list_versions("schemas")
    assert "1.0.0" in versions
    pack = reg.schemas_pack("1.0.0")
    assert pack["kind"] == "schemas"
    assert "reason_codes" in pack["schemas"]
    assert "crm_qa" in pack["schemas"]


def test_detect_schema_reason_codes():
    reg = get_registry()
    headers = ["reason_code", "reason_text", "brief_explanation", "keywords", "model_name"]
    assert reg.detect_schema(headers) == "reason_codes"


def test_detect_schema_crm_qa_persian():
    reg = get_registry()
    assert reg.detect_schema(["پرسش", "پاسخ"]) == "crm_qa"


def test_detect_schema_via_xlsx_parser():
    from kb_manager.parsers.xlsx_parser import SCHEMA_A_COLUMNS, _detect_schema

    assert SCHEMA_A_COLUMNS  # populated from pack
    assert _detect_schema(["پرسش", "پاسخ"]) == "crm_qa"
    assert _detect_schema(["reason_code", "reason_text", "brief_explanation", "keywords"]) == "reason_codes"


def test_taxonomy_subspace_map():
    reg = get_registry()
    smap = reg.subspace_map()
    assert smap["cheque"]
    assert "facilities_person" in smap
    boosts = reg.subspace_keyword_boost()
    assert boosts["cheque"] == 3.5
    assert reg.evidence_threshold("cheque") == 0.05


def test_match_chunk_filter_by_subspace():
    reg = get_registry()
    assert reg.match_chunk_to_filter(
        folder_hierarchy=["اشخاص حقیقی", "گزارش اعتبارسنجی چک"],
        domain=None,
        subspace="cheque",
    )
    assert not reg.match_chunk_to_filter(
        folder_hierarchy=["پایگاه دانش عمومی"],
        domain=None,
        subspace="cheque",
    )


def test_canned_and_prompts():
    reg = get_registry()
    canned = reg.canned_pack()
    assert "سلام" in canned["greetings"]["greeting"] or "دستیار" in canned["greetings"]["greeting"]
    prompts = reg.prompts_pack("orchestrator")
    keys = prompts["prompts"]
    for required in (
        "classify",
        "hybrid_classify",
        "clarify",
        "rewrite_system",
        "rewrite_user_with_history",
        "answer_system",
        "followup_system",
        "followup_user",
        "specify_judge",
        "suggest_subspace",
        "compact_summary_system",
        "compact_incremental_user",
        "greeting_system_hello",
        "greeting_system_farewell",
        "greeting_system_thanks",
        "clarify_fallback",
        "hybrid_judge_escalation",
    ):
        assert required in keys, required
    assert "{text}" in keys["classify"]
    assert reg.active_pins()["prompts_orchestrator"] == "1.0.1"


def test_active_pins_and_manifest(tmp_path, monkeypatch):
    from kb_manager.registry import manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "MANIFESTS_DIR", tmp_path)
    monkeypatch.setattr(manifest_mod, "ACTIVE_POINTER", tmp_path / "ACTIVE.json")
    m = build_manifest(doc_count=10, chunk_count=100, source_dir=str(tmp_path))
    path = save_manifest(m, activate=True)
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["packs"]["schemas"] == "1.0.0"
    assert data["active"] is True
    active = json.loads((tmp_path / "ACTIVE.json").read_text(encoding="utf-8"))
    assert active["manifest_id"] == m.manifest_id


def test_missing_pack_raises():
    reg = Registry()
    with pytest.raises(PackNotFoundError):
        reg.load("schemas", version="9.9.9")


def test_save_prompts_bump_updates_pin(tmp_path):
    """Editor save path: bump prompt pack + flip retrieval.active_packs pin."""
    import shutil

    src = PACKS_ROOT
    dst = tmp_path / "packs"
    shutil.copytree(src, dst)
    reg = Registry(root=dst)
    before = reg.prompts_pack("orchestrator")
    before["prompts"] = dict(before.get("prompts") or {})
    before["prompts"]["classify"] = 'ONLY_TEST "{text}"'
    result = reg.save(
        "prompts",
        before,
        scope="orchestrator",
        bump="patch",
        update_active_pin=True,
    )
    assert result["ok"] is True
    assert result["version"] == "1.0.1"
    assert result["pins"]["prompts_orchestrator"] == "1.0.1"
    loaded = reg.prompts_pack("orchestrator", "1.0.1")
    assert loaded["prompts"]["classify"].startswith("ONLY_TEST")
    retrieval = reg.retrieval_pack()
    assert retrieval["active_packs"]["prompts_orchestrator"] == "1.0.1"
