@echo off
echo Repairing ZWNJ-corrupted chunks (re-ingest with fixed persian.py)...
cd /d "%~dp0"
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul
python repair_chunks.py
echo.
echo Restarting server...
python start_server_detached.py
timeout /t 10 /nobreak >nul
type server_err.log
echo.
echo Test: fetching transparency for staff file...
curl -s http://127.0.0.1:8000/transparency/ab2f5ac7-d39b-498c-9397-834ef213b2f3 | findstr "نام و نام خانوادگی" >nul && echo OK header || echo FAIL
pause
