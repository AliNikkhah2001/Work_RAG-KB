import os

# Respect existing env; default to portable SQLite path relative to repo.
if not os.getenv("KB_DB_URL"):
    os.environ["KB_DB_URL"] = "sqlite+aiosqlite:///./data/kb_test.db"
import uvicorn

if __name__ == "__main__":
    from kb_manager.config import load_config
    cfg = load_config()
    uvicorn.run("kb_manager.web.app:app", host=cfg.web_host, port=cfg.web_port, log_level="info")