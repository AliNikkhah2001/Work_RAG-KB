"""Corpus deduplication: normalized-question + MinHash LSH.

Target: 6208 -> ~5000 chunks (-19%). Two stages:
1) Exact normalized-question dedup (already in SemanticChunker, here as standalone)
2) Near-duplicate MinHash LSH on content shingles (content dedup for non-QA)
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Iterable


def normalize_question(q: str) -> str:
    """Same as SemanticChunker._normalize_question — single source collapsed."""
    from kb_manager.preprocessor.regex_persian import normalize_digits, strip_diacritics

    s = q.strip().lower()
    s = s.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9").replace("\u0671", "\u0627")
    s = s.replace("\u200c", " ").replace("؟", "").replace("?", "")
    s = strip_diacritics(s)
    s = normalize_digits(s, to="ascii")
    return re.sub(r"\s+", " ", s).strip()


def dedup_by_question(chunks: list[dict]) -> tuple[list[dict], int]:
    """Keep first occurrence per normalized question. Returns (kept, removed)."""
    seen: set[str] = set()
    kept: list[dict] = []
    removed = 0
    for ch in chunks:
        # extract question field if qa_pair else use content
        q = ch.get("metadata", {}).get("fields", {}).get("question") or ch.get("content", "")[:200]
        nq = normalize_question(q) if ch.get("chunk_type") == "qa_pair" else ""
        if nq:
            if nq in seen:
                removed += 1
                continue
            seen.add(nq)
        kept.append(ch)
    return kept, removed


# ---------------------------------------------------------------------------
# MinHash LSH (lightweight, no deps)
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 5) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _minhash(shingles: set[str], num_perm: int = 64) -> list[int]:
    """64-perm MinHash using hashlib."""
    sig = []
    for i in range(num_perm):
        min_h = None
        for sh in shingles:
            h = int(hashlib.md5(f"{i}:{sh}".encode()).hexdigest()[:8], 16)
            if min_h is None or h < min_h:
                min_h = h
        sig.append(min_h or 0)
    return sig


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def lsh_dedup(
    chunks: list[dict],
    threshold: float = 0.85,
    num_perm: int = 64,
    bands: int = 8,
) -> tuple[list[dict], list[tuple[int, int, float]]]:
    """MinHash LSH dedup. Returns (kept, [(idx_a, idx_b, jaccard), ...] duplicates)."""
    # band = rows
    rows = num_perm // bands
    sigs = [_minhash(_shingles(c.get("content", "")), num_perm) for c in chunks]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(bands):
            band_sig = tuple(sig[b * rows : (b + 1) * rows])
            buckets[(b, band_sig)].append(idx)

    # candidate pairs
    candidates: set[tuple[int, int]] = set()
    for lst in buckets.values():
        if len(lst) > 1:
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    a, b = lst[i], lst[j]
                    if a > b:
                        a, b = b, a
                    candidates.add((a, b))

    # verify with actual Jaccard
    shingle_sets = [_shingles(c.get("content", "")) for c in chunks]
    duplicates: list[tuple[int, int, float]] = []
    to_remove: set[int] = set()
    for a, b in sorted(candidates):
        if a in to_remove or b in to_remove:
            continue
        jac = _jaccard(shingle_sets[a], shingle_sets[b])
        if jac >= threshold:
            # keep shorter ordinal / earlier doc
            duplicates.append((a, b, jac))
            to_remove.add(b)  # keep first

    kept = [c for i, c in enumerate(chunks) if i not in to_remove]
    return kept, duplicates


def full_dedup_pipeline(chunks: list[dict]) -> dict:
    """Run question dedup + LSH, return stats."""
    q_kept, q_removed = dedup_by_question(chunks)
    l_kept, l_dups = lsh_dedup(q_kept, threshold=0.85)
    return {
        "original": len(chunks),
        "after_question_dedup": len(q_kept),
        "question_removed": q_removed,
        "after_lsh": len(l_kept),
        "lsh_removed": len(q_kept) - len(l_kept),
        "final": len(l_kept),
        "reduction_pct": round(100 * (1 - len(l_kept) / max(len(chunks), 1)), 1),
        "duplicate_pairs": l_dups[:10],  # sample
        "kept": l_kept,
    }
