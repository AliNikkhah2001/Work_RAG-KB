import os
os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_test.db"
import uvicorn

if __name__ == "__main__":
    uvicorn.run("kb_manager.web.app:app", host="127.0.0.1", port=8000, log_level="info")