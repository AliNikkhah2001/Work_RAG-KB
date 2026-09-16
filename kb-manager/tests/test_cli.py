from __future__ import annotations

import subprocess
import sys

from kb_manager.config import PROJECT_ROOT


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "kb_manager.cli", "--version"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_cli_inspect(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["col1", "col2"])
    ws.append(["a", "b"])
    xlsx_path = tmp_path / "inspect.xlsx"
    wb.save(str(xlsx_path))

    result = subprocess.run(
        [sys.executable, "-m", "kb_manager.cli", "inspect", str(xlsx_path)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0
    assert "inspect.xlsx" in result.stdout or "col1" in result.stdout
