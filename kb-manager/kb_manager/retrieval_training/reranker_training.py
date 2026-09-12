"""Cross-encoder reranker training contracts (stage 4a).

Fine-tunes the cross-encoder reranker (default
``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``, see
``kb_manager/reranker.py::get_reranker``) on mined
``(query, positive, negatives)`` triples. Heavy compute lives with later
agents; this module only fixes the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "RerankerTrainingConfig",
    "train_reranker",
    "export_reranker",
    "build_triples",
]


@dataclass
class RerankerTrainingConfig:
    """Knobs for reranker fine-tuning."""

    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    max_length: int = 512
    output_dir: str = "artifacts/retrieval_training/reranker"


def train_reranker(
    train_path: str,
    val_path: str,
    config: RerankerTrainingConfig,
    seed: int,
    provenance: str,
) -> dict[str, Any]:
    """Fine-tune the cross-encoder on the mined triples dataset.

    Args:
        train_path: Path to the training triples file.
        val_path: Path to the validation triples file.
        config: Training knobs (model, epochs, batch size, LR, ...).
        seed: Deterministic training seed.
        provenance: Run tag (code rev + db sha + dataset manifest).

    Returns:
        Dict with ``best_checkpoint``, ``metrics``, and ``manifest``.
    """
    raise NotImplementedError


def export_reranker(
    checkpoint_path: str,
    output_dir: str,
    config: RerankerTrainingConfig,
    seed: int,
    provenance: str,
) -> str:
    """Export the best checkpoint to a serving-ready directory.

    Args:
        checkpoint_path: Source checkpoint from :func:`train_reranker`.
        output_dir: Destination directory for the exported model.
        config: Training knobs (records model identity / max length).
        seed: Deterministic seed (recorded in the export manifest).
        provenance: Run tag recorded in the export manifest.

    Returns:
        Path of the exported model directory.
    """
    raise NotImplementedError


def build_triples(
    dataset_path: str,
    output_path: str,
    config: RerankerTrainingConfig,
    seed: int,
    provenance: str,
) -> dict[str, int]:
    """Convert the dataset splits into ``(query, positive, negatives)`` triples.

    Args:
        dataset_path: Path to the built dataset (see ``dataset`` module).
        output_path: Destination path for the triples file.
        config: Training knobs (only used for provenance snapshotting).
        seed: Deterministic shuffle seed.
        provenance: Run tag recorded alongside the triples.

    Returns:
        Counts dict, e.g. ``{"train": ..., "val": ...}``.
    """
    raise NotImplementedError
