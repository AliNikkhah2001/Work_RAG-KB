import sys
import os
sys.path.insert(0, os.path.abspath('venv/lib/python3.11/site-packages'))
import uvicorn
from kb_manager.web.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
