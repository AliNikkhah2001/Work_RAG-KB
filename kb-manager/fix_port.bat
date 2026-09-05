@echo off
echo Killing ALL python processes on port 8000...
netstat -ano | findstr :8000
echo.
echo Killing all python.exe (frees 8000)...
taskkill /F /IM python.exe 2>nul
timeout /t 3 /nobreak >nul
netstat -ano | findstr :8000
if %errorlevel%==0 (
  echo STILL bound - trying again...
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %%a 2>nul
  timeout /t 2 /nobreak >nul
)
echo Starting server...
cd /d "%~dp0"
python start_server_detached.py
timeout /t 8 /nobreak >nul
echo --- server_err.log ---
type server_err.log
echo.
echo --- server.log tail ---
type server.log
echo.
echo --- testing /transparency ---
curl -s http://127.0.0.1:8000/transparency | findstr "Pipeline Transparency" >nul && echo OK: Transparency loaded || echo FAILED
curl -s http://127.0.0.1:8000/ | findstr "KB Manager" >nul && echo OK: Dashboard
pause
