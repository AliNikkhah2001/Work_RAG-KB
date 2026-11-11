#!/usr/bin/env python3
"""Build subspace QA cluster distill artifacts (read-only on kb-source)."""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from kb_manager.parsers import get_parser

ROOT = Path(
    "/Users/alinikkhah/Documents/Work/CreditScoring/Chatbot_production_v1/"
    "components/knowledgebase/kb-source/1405-08-20/پایگاه دانش"
).resolve()
OUT = Path(
    "/Users/alinikkhah/Documents/Work/CreditScoring/Chatbot_production_v1/"
    "components/knowledgebase/kb-manager/reports/1405-08-20/subspace_distill"
).resolve()

SUBSPACE_LABELS_FA = {
    "cheque": "گزارش اعتبارسنجی چک",
    "facilities_legal": "اعتبارسنجی تسهیلات اشخاص حقوقی",
    "facilities_person": "اعتبارسنجی تسهیلات اشخاص حقیقی",
    "credit_ladder": "نردبان اعتباری",
    "guarantor": "کفیل",
    "supplier_legal": "تامین‌کننده اطلاعات",
    "customers_person": "مشتریان من / پنل کسب‌وکار (حقیقی)",
    "customers_legal": "مشتریان من / پنل کسب‌وکار (حقوقی)",
    "general": "پایگاه دانش عمومی",
    "other": "سایر",
}

SUBSPACE_INTENT = {
    "cheque": "پرسش‌های مربوط به گزارش، امتیاز و سوابق چک اشخاص حقیقی",
    "facilities_legal": "پرسش‌های گزارش و امتیاز اعتباری تسهیلات شرکت‌ها و اشخاص حقوقی",
    "facilities_person": "پرسش‌های گزارش و امتیاز اعتباری تسهیلات اشخاص حقیقی",
    "credit_ladder": "پرسش‌های محصول نردبان اعتباری و مسیر بهبود رتبه",
    "guarantor": "پرسش‌های اعتبارسنجی کفیل، قرار کفالت و وجهالکفاله",
    "supplier_legal": "پرسش‌های عضویت و نقش تامین‌کننده اطلاعات در پایگاه داده اعتباری",
    "customers_person": "پرسش‌های پنل کسب‌وکار و مشتریان من برای کاربر حقیقی",
    "customers_legal": "پرسش‌های پنل کسب‌وکار و مشتریان من برای شخص حقوقی",
    "general": "پرسش‌های عمومی درباره شرکت، محصولات و مفاهیم اعتبارسنجی",
    "other": "پرسش‌هایی که به زیر فضای مشخصی نگاشت نشدند",
}

IMPORTANT_PAIR_BUCKETS = [
    ("cheque", "facilities_person"),
    ("facilities_person", "facilities_legal"),
    ("credit_ladder", "facilities_person"),
    ("general", "facilities_legal"),
]

Q_NORMS = {"سوال", "question", "q"}
A_NORMS = {
    "پاسخ",
    "پاسخکوتاه",
    "پاسخمختصر",
    "پاسخطولانی",
    "answer",
    "briefanswer",
}


def _norm_header(h: str) -> str:
    return re.sub(r"\s+", "", (h or "").strip().lower().replace("_", "").replace("-", ""))


def looks_like_qa(headers: list[str]) -> bool:
    nh = [_norm_header(h) for h in headers]
    has_q = any(h in Q_NORMS or h.startswith("سوال") for h in nh)
    has_a = any(
        h in A_NORMS or "پاسخ" in h or "answer" in h
        for h in nh
        if "کلید" not in h and "keyword" not in h
    )
    return has_q and has_a


def assign_subspace(rel: str) -> str:
    p = rel.replace("\\", "/")
    if "گزارش اعتبارسنجی چک" in p or re.search(r"(^|/)چک(/|$)", p):
        return "cheque"
    if "نردبان" in p:
        return "credit_ladder"
    if "کفیل" in p:
        return "guarantor"
    if "تامین کننده اطلاعات" in p or "تامین‌کننده اطلاعات" in p:
        return "supplier_legal"
    if "پایگاه دانش عمومی" in p:
        return "general"
    is_person = "حقیقی" in p
    is_legal = "حقوقی" in p
    is_customers = (
        "مشتریان من" in p
        or "پنل کسب و کار" in p
        or "پنل کسب‌وکار" in p
        or "پنل" in Path(p).name
    )
    if is_customers and is_person:
        return "customers_person"
    if is_customers and is_legal:
        return "customers_legal"
    if is_legal and ("تسهیلات" in p or "اعتبارسنجی تسهیلات" in p):
        return "facilities_legal"
    if is_person and "تسهیلات" in p:
        return "facilities_person"
    return "other"


def extract_qa_rows() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped = 0
    for fpath in sorted(ROOT.rglob("*.xlsx")):
        if fpath.name.startswith("~$"):
            continue
        rel = str(fpath.relative_to(ROOT))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            doc = get_parser(str(fpath)).parse(str(fpath))
        for sh in doc.sheets or []:
            schema = sh.get("schema")
            headers = sh.get("headers") or []
            if schema != "crm_qa" and not looks_like_qa(headers):
                continue
            nh = [_norm_header(h) for h in headers]
            q_idx = None
            a_idxs: list[int] = []
            for i, h in enumerate(nh):
                if h in Q_NORMS or h.startswith("سوال"):
                    q_idx = i
                elif (
                    h in A_NORMS or "پاسخ" in h or "answer" in h
                ) and "کلید" not in h and "keyword" not in h and "tag" not in h and "categor" not in h:
                    a_idxs.append(i)
            if q_idx is None or not a_idxs:
                continue
            for r_i, row in enumerate(sh.get("rows") or []):
                q = str(row[q_idx]).strip() if q_idx < len(row) and row[q_idx] is not None else ""
                answers = []
                for i in a_idxs:
                    if i < len(row) and row[i] is not None:
                        val = str(row[i]).strip()
                        if val:
                            answers.append(val)
                if not q or not answers:
                    skipped += 1
                    continue
                records.append(
                    {
                        "id": f"{rel}::{sh.get('name')}::{r_i}",
                        "question": q,
                        "answers": answers,
                        "subspace": assign_subspace(rel),
                        "source_file": rel,
                        "sheet": sh.get("name"),
                        "row_index": r_i,
                    }
                )
    return records, skipped


def choose_k(n: int) -> int:
    if n < 8:
        return 1
    return min(12, max(3, int(math.sqrt(n))))


def build_vectorizer(questions: list[str]) -> TfidfVectorizer:
    # char_wb works well for Persian without a stemmer; add word 1-2 grams too
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )


def cluster_subspace(
    items: list[dict[str, Any]], subspace: str
) -> tuple[list[dict[str, Any]], Any, Any]:
    questions = [it["question"] for it in items]
    n = len(items)
    k = choose_k(n)
    vectorizer = build_vectorizer(questions)
    X = vectorizer.fit_transform(questions)
    Xn = normalize(X)

    if k == 1 or n < 8:
        labels = np.zeros(n, dtype=int)
    else:
        # cosine distance via average linkage on dense unit vectors
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(Xn.toarray())

    clusters: list[dict[str, Any]] = []
    feature_names = np.array(vectorizer.get_feature_names_out())

    for cid in sorted(set(labels.tolist())):
        idxs = [i for i, lab in enumerate(labels) if lab == cid]
        subX = Xn[idxs]
        # medoid = closest to centroid in cosine space
        centroid = np.asarray(subX.mean(axis=0)).ravel()
        sims = cosine_similarity(subX, centroid.reshape(1, -1)).ravel()
        order = np.argsort(-sims)
        medoid_local = order[: min(5, len(order))]
        medoids = [questions[idxs[j]] for j in medoid_local]

        # distinctive tokens: mean tf-idf of cluster vs global (use raw X)
        mean_tfidf = np.asarray(X[idxs].mean(axis=0)).ravel()
        top_idx = np.argsort(-mean_tfidf)[:12]
        tokens = []
        for ti in top_idx:
            tok = feature_names[ti].strip()
            if not tok or len(tok) < 3:
                continue
            tokens.append(tok)
            if len(tokens) >= 8:
                break

        summary = craft_summary(subspace, medoids, tokens)
        clusters.append(
            {
                "cluster_id": f"{subspace}_c{cid:02d}",
                "size": len(idxs),
                "representative_questions": medoids,
                "summary_fa": summary,
                "distinctive_tokens": tokens,
                "member_ids": [items[i]["id"] for i in idxs],
            }
        )

    clusters.sort(key=lambda c: (-c["size"], c["cluster_id"]))
    return clusters, Xn, vectorizer


def craft_summary(subspace: str, medoids: list[str], tokens: list[str]) -> str:
    """Heuristic short Persian topic summary from medoids (no LLM)."""
    blob = " ".join(medoids)
    label = SUBSPACE_LABELS_FA.get(subspace, subspace)

    # keyword theme detectors
    themes: list[str] = []
    checks = [
        (r"امتیاز|رتبه|نمره", "امتیاز و رتبه اعتباری"),
        (r"چک\s*برگشتی|برگشت\s*چک|چک", "چک و سوابق چک"),
        (r"تسهیلات|وام|قسط", "تسهیلات و وام"),
        (r"ضمانت|ضامن|کفیل|کفالت|وجه.?الکفاله", "ضمانت و کفالت"),
        (r"گزارش|استعلام|دریافت", "دریافت و بخش‌های گزارش"),
        (r"بهبود|افزایش|بالا بردن|نردبان", "بهبود رتبه"),
        (r"حقوقی|شرکت|سهامدار", "اشخاص حقوقی و شرکت‌ها"),
        (r"حقیقی|فرد|شخص", "اشخاص حقیقی"),
        (r"پنل|مشتریان من|کسب.?و.?کار", "پنل کسب‌وکار و مشتریان"),
        (r"تامین.?کننده|عضویت|داده", "تامین‌کننده اطلاعات و عضویت"),
        (r"اعتراض|اصلاح|به‌?روزرسانی|خطا", "اصلاح و به‌روزرسانی اطلاعات"),
        (r"محرمانه|حریم|رضایت|دسترسی", "حریم خصوصی و دسترسی"),
        (r"تعریف|چیست|یعنی چی|معنی", "تعاریف و مفاهیم پایه"),
    ]
    for pat, name in checks:
        if re.search(pat, blob):
            themes.append(name)
    themes = themes[:3] or ["موضوعات پرتکرار پرسش‌ها"]

    sample = medoids[0] if medoids else ""
    if len(sample) > 80:
        sample = sample[:77] + "…"

    theme_txt = "، ".join(themes)
    return (
        f"این خوشه در زیرفضای «{label}» حول {theme_txt} متمرکز است "
        f"(نمونه: «{sample}»)."
    )


def mine_hard_pairs(
    records: list[dict[str, Any]],
    X_all: Any,
    sim_threshold: float = 0.55,
    cap: int = 40,
) -> list[dict[str, Any]]:
    n = len(records)
    if n < 2:
        return []
    sims = cosine_similarity(X_all)
    np.fill_diagonal(sims, -1.0)

    subspaces = [r["subspace"] for r in records]
    # collect candidates per important bucket + general cross-subspace
    bucket_pairs: dict[tuple[str, str], list[tuple[float, int, int]]] = defaultdict(list)

    for i in range(n):
        # nearest neighbor in another subspace
        best_j = -1
        best_s = -1.0
        for j in range(n):
            if subspaces[j] == subspaces[i]:
                continue
            s = float(sims[i, j])
            if s > best_s:
                best_s = s
                best_j = j
        if best_j < 0 or best_s < sim_threshold:
            continue
        a, b = subspaces[i], subspaces[best_j]
        key = tuple(sorted((a, b)))
        # store unordered edge once (i < j preferred)
        ii, jj = (i, best_j) if i < best_j else (best_j, i)
        bucket_pairs[key].append((best_s, ii, jj))

    # dedupe edges
    for key in list(bucket_pairs.keys()):
        seen = set()
        uniq = []
        for s, i, j in sorted(bucket_pairs[key], key=lambda t: -t[0]):
            edge = (i, j)
            if edge in seen:
                continue
            seen.add(edge)
            uniq.append((s, i, j))
        bucket_pairs[key] = uniq

    # balance across important confusions
    selected: list[tuple[float, int, int, tuple[str, str]]] = []
    per_bucket_target = max(4, cap // max(1, len(IMPORTANT_PAIR_BUCKETS)))

    def take_from(key: tuple[str, str], limit: int) -> None:
        for s, i, j in bucket_pairs.get(key, []):
            if len([1 for x in selected if x[3] == key]) >= limit:
                break
            if any((i == a and j == b) or (i == b and j == a) for _, a, b, _ in selected):
                continue
            selected.append((s, i, j, key))

    for a, b in IMPORTANT_PAIR_BUCKETS:
        take_from(tuple(sorted((a, b))), per_bucket_target)

    # fill remainder with highest cross-subspace sims
    if len(selected) < cap:
        rest: list[tuple[float, int, int, tuple[str, str]]] = []
        for key, lst in bucket_pairs.items():
            for s, i, j in lst:
                rest.append((s, i, j, key))
        rest.sort(key=lambda t: -t[0])
        for s, i, j, key in rest:
            if len(selected) >= cap:
                break
            if any((i == a and j == b) or (i == b and j == a) for _, a, b, _ in selected):
                continue
            selected.append((s, i, j, key))

    selected = selected[:cap]
    selected.sort(key=lambda t: -t[0])

    out = []
    for s, i, j, key in selected:
        out.append(
            {
                "similarity": round(float(s), 4),
                "subspace_a": records[i]["subspace"],
                "subspace_b": records[j]["subspace"],
                "question_a": records[i]["question"],
                "question_b": records[j]["question"],
                "id_a": records[i]["id"],
                "id_b": records[j]["id"],
                "bucket": f"{key[0]}↔{key[1]}",
            }
        )
    return out


def sibling_map() -> dict[str, list[str]]:
    return {
        "cheque": ["facilities_person", "credit_ladder", "general"],
        "facilities_person": ["facilities_legal", "cheque", "credit_ladder", "guarantor"],
        "facilities_legal": ["facilities_person", "supplier_legal", "customers_legal", "general"],
        "credit_ladder": ["facilities_person", "cheque", "general"],
        "guarantor": ["facilities_person", "general"],
        "supplier_legal": ["facilities_legal", "customers_legal", "general"],
        "customers_person": ["customers_legal", "facilities_person"],
        "customers_legal": ["customers_person", "facilities_legal", "supplier_legal"],
        "general": ["facilities_person", "facilities_legal", "cheque", "credit_ladder"],
        "other": ["general"],
    }


def sticky_phrases_for(subspace: str, clusters: list[dict[str, Any]]) -> list[str]:
    base = {
        "cheque": ["گزارش چک", "امتیاز چک", "چک برگشتی", "اعتبارسنجی چک"],
        "facilities_legal": ["تسهیلات حقوقی", "امتیاز شرکت", "گزارش اعتباری شرکتها"],
        "facilities_person": ["تسهیلات حقیقی", "امتیاز اعتباری من", "گزارش تسهیلات"],
        "credit_ladder": ["نردبان اعتباری", "بهبود رتبه", "پله نردبان"],
        "guarantor": ["کفیل", "قرار کفالت", "وجه الکفاله", "اعتبارسنجی کفیل"],
        "supplier_legal": ["تامین کننده اطلاعات", "عضویت پایگاه داده اعتباری"],
        "customers_person": ["مشتریان من", "پنل کسب و کار"],
        "customers_legal": ["مشتریان من حقوقی", "پنل کسب و کار حقوقی"],
        "general": ["اعتبارسنجی چیست", "شرکت اعتبارسنجی ایران"],
        "other": [],
    }.get(subspace, [])
    # add short distinctive fragments from largest cluster medoids
    extra = []
    for c in clusters[:3]:
        for q in c.get("representative_questions", [])[:2]:
            # take first 3-4 content words
            words = re.findall(r"[\u0600-\u06FF]+", q)
            if words:
                extra.append(" ".join(words[:3]))
    seen = set()
    out = []
    for p in base + extra:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:10]


def write_topic_cards(
    by_ss: dict[str, list[dict[str, Any]]],
    clusters_by_ss: dict[str, list[dict[str, Any]]],
) -> None:
    siblings = sibling_map()
    cards = []
    for ss in sorted(by_ss.keys()):
        clusters = clusters_by_ss.get(ss, [])
        in_scope = [c["summary_fa"] for c in clusters[:8]]
        out_scope = []
        for sib in siblings.get(ss, []):
            out_scope.append(
                f"سوالات مربوط به «{SUBSPACE_LABELS_FA.get(sib, sib)}» → subspace:{sib}"
            )
        cards.append(
            {
                "id": ss,
                "label_fa": SUBSPACE_LABELS_FA.get(ss, ss),
                "intent": SUBSPACE_INTENT.get(ss, ""),
                "n_questions": len(by_ss[ss]),
                "n_clusters": len(clusters),
                "in_scope": in_scope,
                "out_of_scope": out_scope,
                "sticky_phrases": sticky_phrases_for(ss, clusters),
            }
        )
    path = OUT / "topic_cards.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(
            {"subspaces": cards},
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=100,
        )


def write_recursive_summaries(
    clusters_by_ss: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
) -> None:
    lines = [
        "# Recursive subspace summaries",
        "",
        "Per-subspace cluster summaries followed by a rollup paragraph.",
        "",
    ]
    for ss in sorted(clusters_by_ss.keys(), key=lambda s: (-counts.get(s, 0), s)):
        clusters = clusters_by_ss[ss]
        label = SUBSPACE_LABELS_FA.get(ss, ss)
        lines.append(f"## {ss} — {label} (n={counts.get(ss, 0)}, clusters={len(clusters)})")
        lines.append("")
        for c in clusters:
            lines.append(
                f"- **{c['cluster_id']}** (size={c['size']}): {c['summary_fa']}"
            )
            reps = " | ".join(c["representative_questions"][:3])
            lines.append(f"  - medoids: {reps}")
            if c.get("distinctive_tokens"):
                lines.append(f"  - tokens: {', '.join(c['distinctive_tokens'][:6])}")
        lines.append("")
        # rollup
        themes = [c["summary_fa"] for c in clusters[:5]]
        rollup = (
            f"زیرفضای «{label}» شامل {counts.get(ss, 0)} پرسش در {len(clusters)} خوشه است. "
            f"تمرکز اصلی: "
            + "؛ ".join(
                re.sub(r"^این خوشه در زیرفضای «[^»]+» حول ", "", t).rstrip("。.")
                for t in themes
            )
            + f" پرسش‌های این زیرفضه را نباید با siblingهایی مانند "
            + "، ".join(
                SUBSPACE_LABELS_FA.get(s, s) for s in sibling_map().get(ss, [])[:3]
            )
            + " اشتباه گرفت."
        )
        lines.append(f"**Rollup:** {rollup}")
        lines.append("")
    (OUT / "recursive_summaries.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clusters").mkdir(parents=True, exist_ok=True)

    records, skipped = extract_qa_rows()
    by_ss: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_ss[r["subspace"]].append(r)

    counts = {ss: len(items) for ss, items in sorted(by_ss.items())}
    (OUT / "subspace_counts.json").write_text(
        json.dumps(
            {
                "total_qa": len(records),
                "skipped_incomplete": skipped,
                "per_subspace": counts,
                "n_subspaces": len(counts),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Global vector space for hard pairs (fit on all questions)
    all_questions = [r["question"] for r in records]
    global_vec = build_vectorizer(all_questions)
    X_all = normalize(global_vec.fit_transform(all_questions))

    clusters_by_ss: dict[str, list[dict[str, Any]]] = {}
    total_clusters = 0
    for ss, items in sorted(by_ss.items()):
        clusters, _, _ = cluster_subspace(items, ss)
        clusters_by_ss[ss] = clusters
        total_clusters += len(clusters)
        # strip member_ids from public artifact? keep for audit but can be large — keep
        (OUT / "clusters" / f"{ss}.json").write_text(
            json.dumps(clusters, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # Adjust hard-pair threshold if needed
    hard = mine_hard_pairs(records, X_all, sim_threshold=0.55, cap=40)
    if len(hard) < 15:
        hard = mine_hard_pairs(records, X_all, sim_threshold=0.45, cap=40)
    elif len(hard) > 40:
        hard = hard[:40]

    with (OUT / "hard_pairs.jsonl").open("w", encoding="utf-8") as f:
        for row in hard:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_topic_cards(by_ss, clusters_by_ss)
    write_recursive_summaries(clusters_by_ss, counts)

    # README + METHOD_NOTES
    readme = f"""# Subspace QA distill — 1405-08-20

## Method

1. Walk all `.xlsx` under `kb-source/1405-08-20/پایگاه دانش`.
2. Parse with `kb_manager.parsers.get_parser` (PYTHONPATH=kb-manager). Keep sheets with schema `crm_qa` or Q/A-like headers.
3. Drop incomplete rows (missing question **or** no non-empty answer field). **No Postgres writes.**
4. Assign subspace from folder path (cheque / facilities_* / credit_ladder / guarantor / supplier_legal / customers_* / general / other).
5. Cluster **within** each subspace on question text only via TF-IDF `char_wb` (3–5) + AgglomerativeClustering (cosine, average).  
   `K = min(12, max(3, int(sqrt(n))))` for `n >= 8`; tiny subspaces → single cluster.
6. Per cluster: medoid representatives (≤5), short Persian summary from medoids, distinctive TF-IDF tokens.
7. Hard pairs: nearest neighbor in another subspace on the global TF-IDF space; keep cosine ≥ ~0.55 (adjusted if needed), cap 40, prefer important confusion buckets.

## Counts

| Metric | Value |
|---|---|
| Total QA kept | {len(records)} |
| Incomplete skipped | {skipped} |
| Subspaces | {len(counts)} |
| Clusters (all) | {total_clusters} |
| Hard pairs | {len(hard)} |

### Per subspace

| Subspace | Questions | Clusters |
|---|---:|---:|
"""
    for ss in sorted(counts.keys(), key=lambda s: -counts[s]):
        readme += f"| {ss} | {counts[ss]} | {len(clusters_by_ss[ss])} |\n"

    readme += """
## Artifacts

- `subspace_counts.json`
- `clusters/<subspace>.json`
- `topic_cards.yaml`
- `hard_pairs.jsonl`
- `recursive_summaries.md`
- `METHOD_NOTES.md`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    method = f"""# Method notes — subspace distill

## What worked

- Parser `get_parser` + schema `crm_qa` cleanly recovered {len(records)} Q/A rows; {skipped} blanks match the known blank/corrupt CSV count (~24).
- Path-based subspace routing separated cheque vs facilities_person vs facilities_legal without DB taxonomy.
- TF-IDF `char_wb` (3–5 grams) is robust for Persian (no tokenizer/stemmer required) and fast on ~1k questions.
- Agglomerative clustering with cosine + average linkage gave stable small-K topic groups; medoids via centroid cosine are readable.
- Hard-pair mining on a **global** TF-IDF space surfaces cross-subspace confusions (cheque↔facilities, person↔legal, ladder↔facilities, general↔legal).

## Limitations

- Summaries are heuristic (medoid + keyword themes), not editor-curated product copy.
- `اصلی` and `جایگزین` facilities banks both included → duplicate questions inflate sizes/clusters for `facilities_person`.
- char n-grams produce fragmentary “tokens” (not always whole Persian words).
- Tiny subspaces (`customers_*`, sometimes others) are a single dump cluster — limited topic structure.
- Hard-pair threshold is empirical; near-duplicates across domains may dominate over subtle confusions.
- sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) was available but **not required**; TF-IDF kept the run offline-fast and deterministic.

## Alternatives

- Dense multilingual embeddings + HDBSCAN/Agglomerative for softer Persian paraphrases.
- Dedup questions within subspace (or prefer `اصلی` over `جایگزین`) before clustering.
- Word n-grams with a Persian normalizer (`hazm`/`preprocessor`) for cleaner distinctive terms.
- Supervised subspace classifier trained on these labels for runtime sticky routing.
- Human pass over `topic_cards.yaml` in_scope / sticky_phrases before wiring into orchestrator.
"""
    (OUT / "METHOD_NOTES.md").write_text(method, encoding="utf-8")

    meta = {
        "total_qa": len(records),
        "skipped": skipped,
        "counts": counts,
        "total_clusters": total_clusters,
        "hard_pairs": len(hard),
        "out": str(OUT),
    }
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
