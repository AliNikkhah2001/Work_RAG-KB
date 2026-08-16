from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestXlsxParser:
    def test_xlsx_parser_reads_reason_codes(self, sample_xlsx: Path):
        from kb_manager.parsers.xlsx_parser import XlsxParser

        parser = XlsxParser()
        doc = parser.parse(str(sample_xlsx))

        assert doc.file_type == "xlsx"
        assert doc.sheets is not None
        assert len(doc.sheets) == 1
        sheet = doc.sheets[0]
        assert sheet["schema"] == "reason_codes"
        assert len(sheet["rows"]) == 3
        assert sheet["rows"][0][0] == "RC001"

    def test_xlsx_parser_reads_crm_qa(self, tmp_path: Path):
        from openpyxl import Workbook

        from kb_manager.parsers.xlsx_parser import XlsxParser

        wb = Workbook()
        ws = wb.active
        ws.append(["question", "model", "briefanswer", "answer", "keyword"])
        ws.append(["چیست؟", "GPT", "کوتاه", "بلند", "کلید"])
        path = tmp_path / "crm.xlsx"
        wb.save(str(path))

        parser = XlsxParser()
        doc = parser.parse(str(path))

        assert doc.sheets is not None
        assert doc.sheets[0]["schema"] == "crm_qa"

    def test_xlsx_parser_skips_temp_files(self, tmp_path: Path):
        from kb_manager.parsers.xlsx_parser import XlsxParser

        (tmp_path / "~$temp_file.xlsx").touch()
        (tmp_path / "real_file.xlsx").touch()

        parser = XlsxParser()
        assert parser.can_parse(str(tmp_path / "~$temp_file.xlsx")) is False
        assert parser.can_parse(str(tmp_path / "real_file.xlsx")) is True


class TestDocxParser:
    def test_docx_parser_reads_content(self, sample_docx: Path):
        from kb_manager.parsers.docx_parser import DocxParser

        parser = DocxParser()
        doc = parser.parse(str(sample_docx))

        assert doc.file_type == "docx"
        assert "آزمایشی" in doc.content
        assert doc.sections is not None
        assert len(doc.sections) > 0


class TestParserRegistry:
    def test_registry_detects_file_type(self, tmp_path: Path):
        from kb_manager.parsers.registry import get_parser, get_supported_extensions

        exts = get_supported_extensions()
        assert ".xlsx" in exts
        assert ".docx" in exts

        with pytest.raises(ValueError, match="Unsupported"):
            get_parser("test.unknown")
