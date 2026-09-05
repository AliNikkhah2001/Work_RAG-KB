"""Start the KB Manager web server with error logging."""
import os
import sys
import logging

if not os.getenv("KB_DB_URL"):
    os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///./data/kb_test.db"

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

import uvicorn
uvicorn.run("kb_manager.web.app:app", host="127.0.0.1", port=8000, log_level="debug")
