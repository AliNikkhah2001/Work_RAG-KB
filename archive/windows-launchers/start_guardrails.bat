@echo off
cd /d "D:\Code\KB\Work_Credit-RAG_Phase1"
python start_guardrails.py > guardrails.log 2>&1
echo Guardrails started on port 8200
timeout /t 2 >nul