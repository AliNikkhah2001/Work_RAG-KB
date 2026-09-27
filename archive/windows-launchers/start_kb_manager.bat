@echo off
cd /d "D:\Code\KB\kb-manager"
python run_server.py > kb-manager.log 2>&1
echo KB Manager started on port 8000
timeout /t 2 >nul