@echo off
REM Restart KB Manager server to load Transparency routes
cd /d "%~dp0"
echo Killing old server on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
  echo Killing PID %%a
  taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo Starting new server (detached)...
python start_server_detached.py
echo.
echo Waiting 5s then checking...
timeout /t 5 /nobreak >nul
curl -s http://127.0.0.1:8000/transparency | findstr "Pipeline Transparency" >nul && echo OK: /transparency loaded || echo FAILED: still 404 - check server.log/server_err.log
curl -s http://127.0.0.1:8000/ | findstr "KB Manager" >nul && echo OK: / still works
pause
