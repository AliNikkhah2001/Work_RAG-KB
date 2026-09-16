# -*- coding: utf-8 -*-
"""Tests for single-column XLSX sheets, Persian label map, and JSON flatten."""
from __future__ import annotations


class TestSingleColAndFlatten:
    def test_single_col_parse(self, tmp_path):
        from openpyxl import Workbook

        from kb_manager.parsers.xlsx_parser import XlsxParser

        wb = Workbook()
        ws = wb.active
        ws.title = "Banks"
        ws.append(["BankName"])
        banks = [f"Bank {i}" for i in range(1, 21)]
        for b in banks:
            ws.append([b])
        path = tmp_path / "banks.xlsx"
        wb.save(str(path))

        parser = XlsxParser()
        doc = parser.parse(str(path))

        assert doc.sheets is not None
        assert len(doc.sheets) == 1
        sheet = doc.sheets[0]
        assert sheet["schema"] == "single_col_list"
        assert sheet["headers"] == ["BankName"]
        assert len(sheet["rows"]) == 20
        assert sheet["rows"][0][0] == "Bank 1"

    def test_single_col_parse_rows_direct(self):
        from kb_manager.parsers.xlsx_parser import XlsxParser

        parser = XlsxParser()
        rows = [["MyList"], ["a"], ["b"], [""]]
        result = parser._parse_sheet_rows("S1", rows, [])
        assert result is not None
        assert result["schema"] == "single_col_list"
        assert len(result["rows"]) == 2

    def test_empty_single_col_returns_none(self):
        from kb_manager.parsers.xlsx_parser import XlsxParser

        parser = XlsxParser()
        assert parser._parse_sheet_rows("S", [["OnlyHeader"]], []) is None
        assert parser._parse_sheet_rows("S", [["", ""]], []) is None

    def test_label_map_presence(self):
        from kb_manager.chunker.semantic import SemanticChunker

        expected = {
            "reason_text": "متن دلیل",
            "improvement_suggestions": "پیشنهاد بهبود",
            "Category": "دسته‌بندی",
            "DocumentName": "نام سند",
            "Title": "عنوان",
            "SectionTitle": "عنوان بخش",
            "Content": "محتوا",
            "Type": "نوع",
            "Version": "نسخه",
            "Author(s)": "نویسنده",
            "Heading": "سرفصل",
            "Summary": "خلاصه",
            "Name": "نام",
            "OBJECT": "موضوع",
        }
        for key, val in expected.items():
            assert SemanticChunker._FIELD_NAMES_FA.get(key) == val, f"missing {key!r}"

    def test_json_flatten(self):
        from kb_manager.parsers.xlsx_parser import XlsxParser

        cell = '{"name": "Ali", "age": 30}'
        out = XlsxParser._format_cell(cell)
        assert "{" not in out
        assert "}" not in out
        assert '"' not in out
        assert "name: Ali" in out
        assert "age: 30" in out
        assert len(out) <= 500

    def test_json_flatten_truncates_to_500(self):
        from kb_manager.parsers.xlsx_parser import XlsxParser

        big = "{" + ", ".join(f'"k{i}": "v{i}"' for i in range(200)) + "}"
        out = XlsxParser._format_cell(big)
        assert len(out) <= 500
        assert "{" not in out

    def test_json_flatten_end_to_end(self, tmp_path):
        from openpyxl import Workbook

        from kb_manager.parsers.xlsx_parser import XlsxParser

        wb = Workbook()
        ws = wb.active
        ws.append(["id", "payload"])
        ws.append(["1", '{"name": "Ali", "city": "Tehran"}'])
        path = tmp_path / "json.xlsx"
        wb.save(str(path))

        parser = XlsxParser()
        doc = parser.parse(str(path))
        assert doc.sheets is not None
        row = doc.sheets[0]["rows"][0]
        flat = row[1]
        assert "{" not in flat
        assert "name: Ali" in flat
