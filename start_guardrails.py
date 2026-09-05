import sys
sys.path.insert(0, r'Work_Credit-RAG_Phase1')
import uvicorn
from components.guardrails.src.work_rag_guardrails.service import create_app

app = create_app()
uvicorn.run(app, host="127.0.0.1", port=8200, log_level="info")