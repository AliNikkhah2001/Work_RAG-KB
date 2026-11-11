"""DEPRECATED — evaluation moved to Work_RAG-Evaluation_Suite.

Importing this package raises. Use EVAL_SUITE_URL HTTP API or
`components/evaluation` CLIs instead.
"""

raise ImportError(
    "kb_manager.evaluation was removed. "
    "Use Work_RAG-Evaluation_Suite (EVAL_SUITE_URL, default http://127.0.0.1:8300) "
    "or `python components/evaluation/scripts/run_hitk.py` / run_multiturn.py."
)
