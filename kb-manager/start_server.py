"""Start the KB Manager web server with error logging (0.0.0.0, config-driven)."""

import logging
import sys

from kb_manager.config import load_config

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

import uvicorn  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    print(f"[KB] DB mode={cfg.db.mode} driver={cfg.db.driver} url={cfg.db.async_url}")
    print(f"[KB] Web {cfg.web_host}:{cfg.web_port}")
    uvicorn.run("kb_manager.web.app:app", host=cfg.web_host, port=cfg.web_port, log_level="debug")
