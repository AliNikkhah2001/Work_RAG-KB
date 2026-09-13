"""Dense bi-encoder training contracts (stage 4b).

Fine-tunes the dense embedding model (default
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``, see
``kb_manager/dense.py::DenseSemanticIndex``) on mined pairs, then rebuilds
the ``.npz`` embedding cache. Heavy compute lives with later agents; this
module only fixes the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DenseTrainingConfig",
    "train_dense",
    "export_dense",
    "rebuild_dense_index",
]


@dataclass
class DenseTrainingConfig:
    """Knobs for dense bi-encoder fine-tuning."""

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2e-5
    use_context: bool = True
    output_dir: str = "artifacts/retrieval_training/dense"


def train_dense(
    train_path: str,
    val_path: str,
    config: DenseTrainingConfig,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Fine-tune the bi-encoder on mined query/passage pairs.

    Args:
        train_path: Path to the training pairs file.
        val_path: Path to the validation pairs file.
        config: Training knobs (model, epochs, batch size, LR, ...).
        seed: Deterministic training seed.
        provenance: Run tag (code rev + db sha + dataset manifest).

    Returns:
        Dict with ``best_checkpoint``, ``metrics``, and ``manifest``.
    """
    raise NotImplementedError


def export_dense(
    checkpoint_path: str,
    output_dir: str,
    config: DenseTrainingConfig,
    seed: int,
    provenance: str,
) -> str:
    """Export the best checkpoint to a serving-ready directory.

    Args:
        checkpoint_path: Source checkpoint from :func:`train_dense`.
        output_dir: Destination directory for the exported model.
        config: Training knobs (records model identity / context flag).
        seed: Deterministic seed (recorded in the export manifest).
        provenance: Run tag recorded in the export manifest.

    Returns:
        Path of the exported model directory.
    """
    raise NotImplementedError


def rebuild_dense_index(
    model_dir: str,
    cache_path: str,
    config: DenseTrainingConfig,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Rebuild the ``.npz`` dense cache with the fine-tuned model.

    Args:
        model_dir: Exported model directory from :func:`export_dense`.
        cache_path: Destination ``.npz`` cache path (fingerprint-guarded,
            see ``kb_manager/dense.py::load_or_build``).
        config: Training knobs.
        seed: Deterministic seed (recorded in the rebuild manifest).
        provenance: Run tag recorded in the rebuild manifest.

    Returns:
        Dict with ``cache_path``, ``fingerprint``, and ``size``.
    """
    raise NotImplementedError
