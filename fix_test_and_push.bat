@echo off
REM Fix failing test and push branches correctly (single-line commit messages)
cd /d "%~dp0"
echo === Fixing parser test ===
cd kb-manager
python -m pytest tests/test_parsers.py::TestXlsxParser::test_xlsx_parser_reads_crm_qa -v
if errorlevel 1 (
  echo TEST STILL FAILS - check xlsx_parser.py
  pause
  exit /b 1
)
echo TEST PASSED
cd ..

echo === Creating branches with single-line commits ===
git checkout -b feat/table-aware-chunker 2>nul || git checkout feat/table-aware-chunker
git add kb-manager/kb_manager/parsers/xlsx_parser.py kb-manager/kb_manager/chunker/semantic.py kb-manager/kb_manager/pipeline/orchestrator.py
git diff --cached --quiet || git commit -m "feat(parser): type-aware schemas glossary staff loan timeline plus row-wise chunking"
git push -u origin feat/table-aware-chunker
if errorlevel 1 echo push feat/table-aware-chunker failed

git checkout master 2>nul
git checkout -b feat/retrieval-tuning 2>nul || git checkout feat/retrieval-tuning
git add kb-manager/kb_manager/web/routes/search.py kb-manager/kb_manager/query_expansion.py kb-manager/kb_manager/preprocessor/clean.py kb-manager/build_iva_dataset.py kb-manager/kb_manager/preprocessor/persian.py
git diff --cached --quiet || git commit -m "fix(retrieval): IVA misses RERANKER 50-100 colloquial strip duplicate gold"
git push -u origin feat/retrieval-tuning

git checkout master 2>nul
git checkout -b feat/qa-massive-and-zip 2>nul || git checkout feat/qa-massive-and-zip
git add kb-manager/tests/test_qa_massive.py kb-manager/kb_manager/web/routes/zip_browser.py kb-manager/kb_manager/web/templates/zip_browser.html kb-manager/kb_manager/web/templates/zip_preview.html kb-manager/kb_manager/web/app.py docs/TABULAR_CHUNKING_RESEARCH.md
git diff --cached --quiet || git commit -m "feat(qa): massive QA harness and zip browser"
git push -u origin feat/qa-massive-and-zip

echo === Running full tests ===
cd kb-manager
python -m pytest tests/test_chunker.py tests/test_parsers.py tests/test_preprocessor.py -v --tb=short
if errorlevel 1 echo TESTS FAILED & pause & exit /b 1
cd ..

echo === Merging to master ===
git checkout master
git merge --no-ff feat/table-aware-chunker -m "merge: feat/table-aware-chunker into master"
git merge --no-ff feat/retrieval-tuning -m "merge: feat/retrieval-tuning into master"
git merge --no-ff feat/qa-massive-and-zip -m "merge: feat/qa-massive-and-zip into master"
git push origin master
if errorlevel 1 echo push master failed

echo === Tag stable ===
git tag -a v7.2-stable -m "stable v7.2 type-aware ZWNJ fix massive QA" 2>nul
git push origin v7.2-stable 2>nul
git checkout -b stable/v7.2 2>nul
git push -u origin stable/v7.2 2>nul

echo === DONE ===
git log --oneline -10
pause
