"""Shared web dependencies to avoid circular imports."""

from kb_manager.config import load_config
from kb_manager.models.database import Database
from fastapi.templating import Jinja2Templates
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

config = load_config()
db = Database(config.db)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))