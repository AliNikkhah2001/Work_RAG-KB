@echo off
cd /d "D:\Code\KB\Work_Credit-RAG_Phase1\scripts"
python mock_gemma_manager.py > gemma.log 2>&1
echo Mock Gemma started on port 9000
timeout /t 2 >nul