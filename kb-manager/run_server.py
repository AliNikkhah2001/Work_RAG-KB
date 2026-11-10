"""Run the KB Manager web server — respects KB_* env vars, binds 0.0.0.0 by default."""

import os

# Only set defaults if not already provided via env / .env.
# When KB_DB_URL is set (compose/prod), do NOT force KB_DB_MODE=sqlite — that
# previously shadowed postgres URLs in _resolve_db_mode_and_url.
if not os.getenv("KB_DB_URL"):
    os.environ.setdefault("KB_DB_MODE", "sqlite")
    os.environ.setdefault("KB_SQLITE_PATH", "./data/kb_test.db")
os.environ.setdefault("KB_WEB_HOST", "0.0.0.0")
os.environ.setdefault("KB_WEB_PORT", "8000")

import uvicorn  # noqa: E402
from kb_manager.config import load_config  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    print(f"[KB] DB mode={cfg.db.mode} driver={cfg.db.driver} url={cfg.db.async_url}")
    print(f"[KB] Web {cfg.web_host}:{cfg.web_port}")
    uvicorn.run("kb_manager.web.app:app", host=cfg.web_host, port=cfg.web_port, log_level="info")