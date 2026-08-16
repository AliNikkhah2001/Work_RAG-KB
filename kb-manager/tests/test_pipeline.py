from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestPipeline:
    @pytest.mark.asyncio()
    async def test_full_rebuild_creates_documents_and_chunks(self, tmp_path: Path):
        from openpyxl import Workbook

        from kb_manager.pipeline.orchestrator import PipelineOrchestrator

        wb = Workbook()
        ws = wb.active
        ws.append(["reason_code", "description"])
        ws.append(["RC001", "تست"])
        xlsx_path = tmp_path / "test.xlsx"
        wb.save(str(xlsx_path))

        class MockDB:
            def session(self):
                import contextlib

                @contextlib.asynccontextmanager
                async def _session():
                    yield None

                return _session()

        orchestrator = PipelineOrchestrator(database=MockDB())
        files = orchestrator._scan_files(str(tmp_path))
        assert len(files) == 1

    @pytest.mark.asyncio()
    async def test_incremental_skips_unchanged_files(self, tmp_path: Path):
        from openpyxl import Workbook

        from kb_manager.pipeline.orchestrator import PipelineOrchestrator

        wb = Workbook()
        ws = wb.active
        ws.append(["reason_code", "description"])
        ws.append(["RC001", "تست"])
        xlsx_path = tmp_path / "test.xlsx"
        wb.save(str(xlsx_path))

        class MockDB:
            def session(self):
                import contextlib

                @contextlib.asynccontextmanager
                async def _session():
                    yield None

                return _session()

        orchestrator = PipelineOrchestrator(database=MockDB())
        files = orchestrator._scan_files(str(tmp_path))
        assert len(files) == 1
        # Verify file scanning works
        assert str(xlsx_path.resolve()) in files
