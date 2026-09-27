"""KB admin auth — simple token check for critical routes."""

from __future__ import annotations

import os
from fastapi import Header, HTTPException
from typing import Optional

def require_admin_auth(x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token")):
    """Require KB_ADMIN_TOKEN if set; otherwise allow (dev)."""
    expected = os.getenv("KB_ADMIN_TOKEN")
    if not expected:
        return  # no token configured — allow (dev mode)
    if x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid X-Admin-Token")
