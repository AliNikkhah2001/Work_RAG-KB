"""Reason-code catalog + exact-match regression tests.

Covers:
- exact-stem spec matching (the substring bug that turned the cheque
  reason-code workbook into the cheque score table),
- ``build_reason_code_entity`` shape (kind / model / hierarchy / subspace_ids),
- duplicate-row dedup,
- reason-code query detection + cross-model ambiguity helpers,
- subspace filtering of reason catalogs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kb_manager.entity_lists import (
    AMBIGUOUS_REASON_CODES,
    build_reason_code_entity,
    entity_model,
    extract_reason_code,
    filter_entities_for_context,
    is_ambiguous_reason_code,
    is_reason_code_query,
    match_list_spec,
    match_reason_spec,
    match_score_spec,
)

_SUPP = Path(__file__).resolve().parents[2] / "kb-source" / "1405-08-20" / "ضمیمه پایگاه دانش"
CHEQUE_RC = str(_SUPP / "لیست/کدهای دلیل/چک/توضیحات تشریحی دلایل کاهش امتیاز چک و روش بهبود آن.xlsx")
CHEQUE_SCORE = str(_SUPP / "جدول امتیاز اعتبارسنجی چک.xlsx")
LOAN = str(_SUPP / "لیست/لیست و عناوین وام ها.xlsx")
PERSON_MAIN = str(
    _SUPP / "لیست/کدهای دلیل/اشخاص حقیقی/مدل اصلی/توضیحات تشریحی دلایل کاهش امتیاز اعتباری مدل اصلی و روش بهبود آن.xlsx"
)
PERSON_SUB = str(
    _SUPP / "لیست/کدهای دلیل/اشخاص حقیقی/مدل جایگزین/توضیحات تشریحی دلایل کاهش امتیاز اعتباری مدل جایگزین و روش بهبود آن.xlsx"
)
LEGAL = str(
    _SUPP / "لیست/کدهای دلیل/اشخاص حقوقی/توضیحات تشریحی دلایل کاهش امتیاز اعتباری شرکتها.xlsx"
)


class TestExactSpecMatching:
    def test_reason_code_file_is_not_a_score_table(self):
        # Regression: «امتیاز چک» substring once matched this workbook.
        assert match_score_spec(CHEQUE_RC) is None
        assert match_reason_spec(CHEQUE_RC) is not None

    def test_score_table_still_matches(self):
        spec = match_score_spec(CHEQUE_SCORE)
        assert spec is not None and spec["list_name"] == "جدول امتیاز اعتبارسنجی چک"

    def test_loan_catalog_still_matches(self):
        spec = match_list_spec(LOAN)
        assert spec is not None and spec["kind"] == "loan_catalog"

    @pytest.mark.parametrize(
        "path,model",
        [
            (CHEQUE_RC, "cheque"),
            (PERSON_MAIN, "person_main"),
            (PERSON_SUB, "person_substitute"),
            (LEGAL, "legal"),
        ],
    )
    def test_reason_specs_resolve_per_model(self, path, model):
        spec = match_reason_spec(path)
        assert spec is not None and spec["model"] == model


class TestReasonCodeQuery:
    @pytest.mark.parametrize(
        "q,expected",
        [
            ("کد دلیل HAD1 چیست؟", True),
            ("HAD1", True),
            ("DAG1-1", True),
            ("دلایل کاهش امتیاز چک چیست؟", True),
            ("reason code HAD1", True),
            ("چرا امتیاز چک من کم شده؟", False),
            ("امتیاز ۶۱۰ حقیقی چه رنگی است؟", False),
            ("سلام", False),
        ],
    )
    def test_is_reason_code_query(self, q, expected):
        assert is_reason_code_query(q) is expected

    @pytest.mark.parametrize(
        "q,code",
        [("کد دلیل HAD1 چیست؟", "HAD1"), ("DAG1-1", "DAG1-1"), ("سلام", None)],
    )
    def test_extract_reason_code(self, q, code):
        assert extract_reason_code(q) == code

    def test_ambiguity_helpers(self):
        assert AMBIGUOUS_REASON_CODES["AGE2"] == ["person_main", "person_substitute"]
        assert is_ambiguous_reason_code("AGE2") is not None
        assert is_ambiguous_reason_code("NGL1", model="person_main") is not None
        assert is_ambiguous_reason_code("AGE2", model="cheque") is not None
        assert is_ambiguous_reason_code("HAD1") is None


class TestReasonCatalogShape:
    @pytest.mark.parametrize(
        "path,list_name,model,subspace",
        [
            (CHEQUE_RC, "کدهای دلیل — چک", "cheque", "cheque"),
            (PERSON_MAIN, "کدهای دلیل — اشخاص حقیقی (مدل اصلی)", "person_main", "facilities_person"),
            (PERSON_SUB, "کدهای دلیل — اشخاص حقیقی (مدل جایگزین)", "person_substitute", "facilities_person"),
            (LEGAL, "کدهای دلیل — اشخاص حقوقی", "legal", "facilities_legal"),
        ],
    )
    def test_build_entity_shape(self, path, list_name, model, subspace):
        # Skip when the raw corpus is not present next to the test checkout.
        import os

        if not os.path.exists(path):
            pytest.skip("reason-code corpus not available")
        ent = build_reason_code_entity(path, None)
        assert ent is not None
        meta = ent["content_json"]["_meta"]
        assert ent["list_name"] == list_name
        assert meta["kind"] == "reason_code_catalog"
        assert meta["model"] == model
        assert subspace in meta["subspace_ids"]
        assert meta["hierarchy"][0] == "کدهای دلیل"
        assert meta["count"] == len(ent["content_json"]["records"])
        assert entity_model(ent) == model

    def test_records_have_canonical_fields_and_unique_codes(self):
        import os

        if not os.path.exists(PERSON_MAIN):
            pytest.skip("reason-code corpus not available")
        ent = build_reason_code_entity(PERSON_MAIN, None)
        recs = ent["content_json"]["records"]
        codes = [r["reason_code"] for r in recs]
        assert len(codes) == len(set(codes)), "duplicate reason codes must be merged"
        for r in recs:
            assert set(r) == {
                "reason_code",
                "reason_text",
                "brief_explanation",
                "detailed_explanation",
                "improvement_suggestions",
                "keywords",
            }


class TestSubspaceFiltering:
    def _ent(self, model, subspace):
        return {
            "list_name": f"کدهای دلیل — {model}",
            "content": {
                "_meta": {
                    "kind": "reason_code_catalog",
                    "model": model,
                    "subspace_ids": [subspace],
                },
                "records": [],
            },
        }

    def test_reason_catalog_does_not_cross_subspace(self):
        ents = [
            self._ent("cheque", "cheque"),
            self._ent("person_main", "facilities_person"),
            self._ent("legal", "facilities_legal"),
        ]
        kept = filter_entities_for_context(ents, query="کد دلیل HAD1", subspace="facilities_person")
        models = {entity_model(e) for e in kept}
        assert models == {"person_main"}
