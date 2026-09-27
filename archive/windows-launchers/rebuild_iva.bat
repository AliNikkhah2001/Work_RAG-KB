@echo off
echo Rebuilding IVA gold set for current DB (2174 chunks, fixed persian)...
cd /d "%~dp0"
python build_iva_dataset.py
if errorlevel 1 (echo build_iva_dataset FAILED & pause & exit /b 1)
echo.
echo Re-running IVA benchmark on rebuilt gold...
python run_iva_eval.py
if errorlevel 1 (echo run_iva_eval FAILED & pause & exit /b 1)
echo.
echo Done - new iva_results.json should show doc_hit vs before (11/15)
echo Compare: type data\iva_results_BEFORE.json and data\iva_results.json
pause
