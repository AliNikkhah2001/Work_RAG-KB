@echo off
echo Rebuilding IVA gold with duplicate handling + re-running debug...
cd /d "%~dp0"
python build_iva_dataset.py
python debug_iva_retrieved.py
echo.
echo Re-running IVA benchmark...
python run_iva_eval.py
pause
