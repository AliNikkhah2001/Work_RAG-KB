"""Shared Jinja2 templates instance.

Kept in a leaf module so route modules can import ``templates`` without
importing :mod:`kb_manager.web.app` (which imports the routers, creating
a circular import when routes are used outside the app, e.g. benchmarks).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
