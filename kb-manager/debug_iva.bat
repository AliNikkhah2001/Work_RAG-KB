@echo off
cd /d "%~dp0"
echo Building side-by-side debug HTML (expects rebuilt iva gold)...
python debug_iva_retrieved.py
if errorlevel 1 (echo debug FAILED & pause & exit /b 1)
echo.
echo Open: data\iva_debug.html  or  http://127.0.0.1:8000/data/iva_debug.html
echo Also try: http://127.0.0.1:8000/transparency  -> Inspect any doc
pause
