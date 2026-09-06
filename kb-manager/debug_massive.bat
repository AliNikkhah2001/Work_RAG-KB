@echo off
cd /d "%~dp0"
echo Listing DB documents...
python debug_docs.py
echo.
echo Running IVA debug (side-by-side) to check 15q...
python debug_iva_retrieved.py 2>&1 | more
echo.
echo Check: data/iva_debug.html and data/massive_results.json
pause
