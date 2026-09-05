@echo off
cd /d "D:\Code\KB\Work_Credit-RAG_Phase1\components\orchestrator\src"
python -m uvicorn work_rag_orchestrator.api:create_app --factory --host 127.0.0.1 --port 8100 > orchestrator.log 2>&1
echo Orchestrator started on port 8100
timeout /t 2 >nul