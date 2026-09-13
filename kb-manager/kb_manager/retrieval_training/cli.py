"""Standalone scaffolding CLI for retrieval-training stages.

Run::

    python -m kb_manager.retrieval_training.cli --help
    python -m kb_manager.retrieval_training.cli mine-negatives --help

NOTE: every subcommand is a not-implemented contract stub: it prints the
stage contract and exits with status 2. Later agents wire each subcommand
to its stage module. Uses only the standard library so ``--help`` works
without heavy model dependencies.
"""

from __future__ import annotations

import argparse
import sys

EXIT_NOT_IMPLEMENTED = 2
DEFAULT_SEED = 42

__all__ = ["build_parser", "main"]


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Attach options shared by every stage subcommand."""
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for sampling/shuffling/training (default: %(default)s).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to a JSON/YAML config file with stage knobs (e.g. TierRatio mixture, "
        "pool sizes). Empty means stage defaults.",
    )
    parser.add_argument(
        "--provenance",
        type=str,
        default="",
        help="Free-text run tag stamped onto every emitted record "
        "(recommended: '<code_rev>/<db_sha>/<config_snapshot>').",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all stage subparsers."""
    parser = argparse.ArgumentParser(
        prog="kb_manager.retrieval_training.cli",
        description="Retrieval-training scaffolding: mine negatives, build datasets, "
        "train reranker/dense models, evaluate. All stages are contract stubs "
        "(exit 2) until later agents implement them.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<stage>")

    p_mine = sub.add_parser(
        "mine-negatives",
        help="Mine Tier1..Tier4 hard negatives (contract stub).",
        description="Stage 1 (mining): query the frozen pipeline "
        "(BM25 + dense + RRF + reranker) and emit MinedNegative records. "
        "Contract stub: prints the contract and exits 2.",
    )
    _add_common_options(p_mine)
    p_mine.add_argument("--queries", type=str, default="",
                        help="Path to input queries JSON (query_id/query/positive_chunk_ids).")
    p_mine.add_argument("--output", type=str, default="",
                        help="Destination path for mined MinedNegative JSONL.")
    p_mine.add_argument("--negatives-per-query", type=int, default=4,
                        help="Negatives to mine per query (default: %(default)s).")
    p_mine.add_argument("--candidate-pool-size", type=int, default=100,
                        help="Candidate pool depth per leg (default: %(default)s).")

    p_ds = sub.add_parser(
        "build-dataset",
        help="Build versioned train/val/test splits (contract stub).",
        description="Stage 3 (dataset): assemble validated negatives + failures into "
        "DatasetSplit sets with a RunManifest. Contract stub: prints the contract "
        "and exits 2.",
    )
    _add_common_options(p_ds)
    p_ds.add_argument("--mined", type=str, default="",
                      help="Path to validated MinedNegative JSONL.")
    p_ds.add_argument("--failures", type=str, default="",
                      help="Path to FailureRecord JSONL.")
    p_ds.add_argument("--output", type=str, default="",
                      help="Destination directory for splits + manifest.")
    p_ds.add_argument("--train-ratio", type=float, default=0.8,
                      help="Train share of queries (default: %(default)s).")
    p_ds.add_argument("--val-ratio", type=float, default=0.1,
                      help="Validation share of queries (default: %(default)s).")
    p_ds.add_argument("--test-ratio", type=float, default=0.1,
                      help="Test share of queries (default: %(default)s).")

    p_rr = sub.add_parser(
        "train-reranker",
        help="Fine-tune the cross-encoder reranker (contract stub).",
        description="Stage 4a (reranker_training): fine-tune the cross-encoder on "
        "mined triples. Contract stub: prints the contract and exits 2.",
    )
    _add_common_options(p_rr)
    p_rr.add_argument("--train", type=str, default="",
                      help="Path to training triples file.")
    p_rr.add_argument("--val", type=str, default="",
                      help="Path to validation triples file.")
    p_rr.add_argument("--model", type=str, default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                      help="Base cross-encoder model id (default: %(default)s).")
    p_rr.add_argument("--epochs", type=int, default=3,
                      help="Training epochs (default: %(default)s).")
    p_rr.add_argument("--output-dir", type=str, default="artifacts/retrieval_training/reranker",
                      help="Destination directory for checkpoints (default: %(default)s).")

    p_de = sub.add_parser(
        "train-dense",
        help="Fine-tune the dense bi-encoder (contract stub).",
        description="Stage 4b (dense_training): fine-tune the bi-encoder and rebuild "
        "the .npz dense cache. Contract stub: prints the contract and exits 2.",
    )
    _add_common_options(p_de)
    p_de.add_argument("--train", type=str, default="",
                      help="Path to training pairs file.")
    p_de.add_argument("--val", type=str, default="",
                      help="Path to validation pairs file.")
    p_de.add_argument("--model", type=str,
                      default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                      help="Base bi-encoder model id (default: %(default)s).")
    p_de.add_argument("--epochs", type=int, default=3,
                      help="Training epochs (default: %(default)s).")
    p_de.add_argument("--output-dir", type=str, default="artifacts/retrieval_training/dense",
                      help="Destination directory for checkpoints (default: %(default)s).")

    p_ev = sub.add_parser(
        "evaluate",
        help="Evaluate a checkpoint vs the frozen baseline (contract stub).",
        description="Stage 5 (evaluation): score a checkpoint with P/R/Hit/MRR/nDCG/MAP "
        "and diff against baseline_freeze.json. Contract stub: prints the contract "
        "and exits 2.",
    )
    _add_common_options(p_ev)
    p_ev.add_argument("--checkpoint", type=str, default="",
                      help="Exported model directory to evaluate.")
    p_ev.add_argument("--dataset", type=str, default="",
                      help="Eval dataset path (queries + expected_chunk_ids).")
    p_ev.add_argument("--baseline", type=str, default="artifacts/retrieval_training/baseline_freeze.json",
                      help="Frozen baseline JSON to diff against (default: %(default)s).")
    p_ev.add_argument("--top-k", type=int, default=10,
                      help="Retrieval depth for metrics (default: %(default)s).")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* and run the requested stage stub.

    Returns:
        Process exit code: always 2 (not implemented yet) for subcommands;
        argparse handles ``--help`` with exit 0.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    print(
        f"retrieval-training stage '{args.command}': contract stub, not implemented yet "
        f"(seed={args.seed}, config={(args.config or 'defaults')!r}, "
        f"provenance={(args.provenance or 'none')!r})."
    )
    return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    sys.exit(main())
