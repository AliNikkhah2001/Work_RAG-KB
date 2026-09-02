import os
os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///D:/Code/KB/kb-manager/data/kb_1405.db"
os.environ["KB_SOURCE_DIR"] = r"D:/Code/KB/kb-source/1405-05-31"
os.environ["KB_SYNONYM_ENABLED"] = "true"
import uvicorn

if __name__ == "__main__":
    uvicorn.run("kb_manager.web.app:app", host="127.0.0.1", port=8000, log_level="info")