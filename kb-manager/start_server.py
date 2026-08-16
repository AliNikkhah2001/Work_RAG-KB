"""Start the KB Manager web server with error logging."""
import os
import sys
import logging

os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///C:/Users/10225/Downloads/KB/kb-manager/data/kb_test.db"
os.chdir(r"C:\Users\10225\Downloads\KB\kb-manager")

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

import uvicorn
uvicorn.run("kb_manager.web.app:app", host="127.0.0.1", port=8000, log_level="debug")
