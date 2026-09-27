@echo off
REM Massive QA test: 400+ verbatim questions must all hit
cd /d "%~dp0"
echo === Massive QA harness ===
echo This runs every QA row in kb-source (Public/Company/Individual/Cheque etc.) verbatim
echo Expect 400+ queries, ~60 min (22s each)
python -m pytest tests/test_qa_massive.py::test_qa_massive_count -v
echo.
echo Running per-file verbatim checks (1 file = ~60 queries, ~20 min total)...
python -m pytest tests/test_qa_massive.py::test_qa_file_verbatim_recall -v --tb=line 2>&1 | more
echo.
echo For full log: python -m pytest tests/test_qa_massive.py -v 2>&1 | more
echo Must be 0 failures for 100%% verbatim
pause
