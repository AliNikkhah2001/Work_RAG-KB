import os

# Respect existing env; default to portable SQLite path relative to repo.
if not os.getenv("KB_DB_URL"):
    os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///./data/kb_test.db"
import uvicorn

if __name__ == "__main__":
    uvicorn.run("kb_manager.web.app:app", host="127.0.0.1", port=8000, log_level="info")