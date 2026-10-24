"""Offline-safe resolution of model names to local filesystem paths.

Why this exists
---------------
``transformers`` 4.57 tokenization loads call
``tokenization_utils_base._patch_mistral_regex`` → ``is_base_mistral()`` →
``huggingface_hub.model_info()`` — a *network* API call — for any repo-id
model name. Under ``HF_HUB_OFFLINE=1`` this hard-fails with
``OfflineModeIsEnabled`` instead of degrading gracefully, which silently
killed the cross-encoder reranker stage (the search route then falls back
to RRF-only ordering — see the 2026-10-12 "reranker offline fallback"
incident in the service log).

Loading from a local directory sets ``_is_local`` in transformers and
short-circuits that call entirely. This helper resolves a repo id to a
durable local directory, checked in a deterministic order so the dense
``.npz`` fingerprint (which includes the model path) stays stable:

1. ``name`` itself, when it is already a local directory (caller override);
2. the offline-provisioned mirror at
   ``Work_RAG-Server-Server-Setup/offline-prep/models/huggingface`` with
   ``/`` replaced by ``_`` in the repo id (durable, survives cache wipes);
3. the local HuggingFace hub cache snapshot (``HF_HOME``/``HF_HUB_CACHE``);
4. ``name`` unchanged — let the caller surface the original error.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Offline-provisioned model mirror shipped by Work_RAG-Server-Setup.
# Kept as a module-level list so tests can monkeypatch it.
OFFLINE_MODEL_ROOTS: list[str] = [
    "/splunk-data/v1/Work_RAG-Server-Setup/offline-prep/models/huggingface",
]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_ROOT_FALLBACK = _PROJECT_ROOT / "models"  # kb-manager-local mirror, if ever used


def resolve_local_model_path(name: str) -> str:
    """Return a local directory for ``name`` when one can be found.

    Non-directory or unresolvable names are returned unchanged so behaviour
    degrades to the pre-existing (network) load path.
    """
    if not name:
        return name
    if os.path.isdir(name):
        return name

    roots = list(OFFLINE_MODEL_ROOTS) + [str(_LOCAL_ROOT_FALLBACK)]
    flat_name = name.replace("/", "_")
    for root in roots:
        candidate = Path(root) / flat_name
        if candidate.is_dir():
            logger.debug("Resolved model %s -> local dir %s", name, candidate)
            return str(candidate)

    try:  # pragma: no cover - depends on machine cache state
        from huggingface_hub import snapshot_download

        path = snapshot_download(name, local_files_only=True)
        if path and os.path.isdir(path):
            logger.debug("Resolved model %s -> HF cache snapshot %s", name, path)
            return path
    except Exception:  # noqa: BLE001 - offline / missing cache / import error
        pass

    return name
