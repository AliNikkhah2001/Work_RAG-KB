@echo off
REM Run all benchmarks on fixed pipeline (no ZWNJ corruption) and save before/after
cd /d "%~dp0"
echo === Backing up BEFORE results ===
if exist data\benchmark_results.json copy /Y data\benchmark_results.json data\benchmark_results_BEFORE.json >nul
if exist data\iva_results.json copy /Y data\iva_results.json data\iva_results_BEFORE.json >nul
if exist data\ir_metrics.json copy /Y data\ir_metrics.json data\ir_metrics_BEFORE.json >nul
echo BEFORE backed up to *_BEFORE.json

echo.
echo === Running IVA 15 benchmark (fixed) ===
python run_iva_eval.py
if errorlevel 1 echo IVA FAILED & pause & exit /b 1

echo.
echo === Running 120q full benchmark (fixed) - this takes ~5-10 min ===
python run_benchmark.py test_questions.json --top-k 5
if errorlevel 1 echo full benchmark FAILED & pause & exit /b 1

echo.
echo === Comparing BEFORE vs AFTER ===
python compare_benchmarks.py
echo.
echo === Done - results in data/*_BEFORE.json and data/benchmark_results.json ===
pause
