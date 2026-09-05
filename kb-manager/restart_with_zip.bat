@echo off
echo Restarting KB server with zip browser + ZWNJ fix...
taskkill /F /IM python.exe 2>nul
timeout /t 3 /nobreak >nul
cd /d "%~dp0"
python start_server_detached.py
echo Started, waiting 12s for startup...
timeout /t 12 /nobreak >nul
type server_err.log
echo.
echo Testing endpoints...
curl -s http://127.0.0.1:8000/transparency | findstr "Pipeline Transparency" >nul && echo OK /transparency || echo FAIL /transparency
curl -s http://127.0.0.1:8000/transparency/zip | findstr "Zip Browser" >nul && echo OK /transparency/zip || echo FAIL /transparency/zip
pause
