"""FaMTEB dataset loader for Persian retrieval evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


# ---------------------------------------------------------------------------
# FaMTEB Dataset Configurations
# ---------------------------------------------------------------------------

FAMTEB_RETRIEVAL_DATASETS = {
    # Synthetic datasets (FaMTEB)
    "synper_qa": {
        "hf_path": "MCINext/synthetic-persian-qa-retrieval",
        "split": "test",
        "query_field": "query",
        "corpus_field": "answer",
        "id_field": "docid",
        "description": "Synthetic Persian QA Retrieval (GPT-4o-mini)",
    },
    "synper_chatbot_rag_topics": {
        "hf_path": "MCINext/synthetic-persian-chatbot-rag-topics-retrieval",
        "split": "test",
        "query_field": "query",
        "corpus_field": "topic",
        "id_field": "docid",
        "description": "Chatbot RAG Topics Retrieval",
    },
    "synper_chatbot_rag_summary": {
        "hf_path": "MCINext/synthetic-persian-chatbot-rag-summary-retrieval",
        "split": "test",
        "query_field": "conversation",
        "corpus_field": "summary",
        "id_field": "docid",
        "description": "Chatbot RAG Summary Retrieval",
    },
    "synper_chatbot_topics": {
        "hf_path": "MCINext/synthetic-persian-chatbot-topics-retrieval",
        "split": "test",
        "query_field": "query",
        "corpus_field": "topic",
        "id_field": "docid",
        "description": "Chatbot Topics Retrieval",
    },
    # BEIR-Fa (translated BEIR)
    "nq_fa": {
        "hf_path": "MCINext/nq-fa",
        "split": "test",
        "query_field": "query",
        "corpus_field": "text",
        "id_field": "docid",
        "description": "NQ-Fa (Translated Natural Questions)",
    },
    # MIRACL Persian
    "miracle_fa": {
        "hf_path": "miracl/miracl",
        "subset": "fa",
        "split": "dev",
        "query_field": "query",
        "corpus_field": "text",
        "id_field": "docid",
        "description": "MIRACL Persian (Wikipedia + human queries)",
    },
}

# Additional FaMTEB datasets for other tasks
FAMTEB_OTHER_DATASETS = {
    "synper_qa_pc": "MCINext/synthetic-persian-qa-pair-classification",
    "synper_chatbot_rag_faq_pc": "MCINext/synthetic-persian-chatbot-rag-faq-pair-classification",
    "synper_text_keywords_pc": "MCINext/synthetic-persian-text-keywords-pair-classification",
    "parsinlu_query_paraphrase": "MCINext/parsinlu-query-paraphrase",
    "farsi_paraphrase": "MCINext/farsi-paraphrase-detection",
    "exapcc": "MCINext/exapcc",
    "farstail": "MCINext/farstail",
}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievalSample:
    """A single retrieval sample with query, relevant docs, and metadata."""
    query: str
    relevant_doc_ids: List[str]
    query_id: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalDataset:
    """A retrieval dataset with queries and corpus."""
    name: str
    queries: Dict[str, str]  # query_id -> query text
    corpus: Dict[str, str]   # doc_id -> document text
    qrels: Dict[str, List[str]]  # query_id -> list of relevant doc_ids
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Dataset Loading Functions
# ---------------------------------------------------------------------------

def load_famteb_retrieval_dataset(
    dataset_key: str,
    max_samples: Optional[int] = None,
) -> RetrievalDataset:
    """Load a FaMTEB retrieval dataset from HuggingFace.

    Args:
        dataset_key: Key from FAMTEB_RETRIEVAL_DATASETS.
        max_samples: Optional limit on number of samples to load.

    Returns:
        RetrievalDataset with queries, corpus, and qrels.
    """
    if not _DATASETS_AVAILABLE:
        raise ImportError(
            "datasets library required for FaMTEB loading. "
            "Install with: pip install datasets"
        )

    if dataset_key not in FAMTEB_RETRIEVAL_DATASETS:
        raise ValueError(f"Unknown dataset key: {dataset_key}")

    config = FAMTEB_RETRIEVAL_DATASETS[dataset_key]
    hf_path = config["hf_path"]
    split = config["split"]
    query_field = config["query_field"]
    corpus_field = config["corpus_field"]
    id_field = config["id_field"]
    subset = config.get("subset")

    # Load dataset
    kwargs = {"split": split}
    if subset:
        kwargs["name"] = subset

    ds = load_dataset(hf_path, **kwargs)

    queries = {}
    corpus = {}
    qrels = {}

    for i, row in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        # F15 fix: validate schema — do not silently synthesize IDs that invalidate qrels
        if id_field not in row or not str(row.get(id_field, "")).strip():
            raise ValueError(f"Dataset {dataset_key} missing id_field '{id_field}' in row {i}: {list(row.keys())}")
        if query_field not in row:
            raise ValueError(f"Dataset {dataset_key} missing query_field '{query_field}' in row {i}")
        if corpus_field not in row:
            raise ValueError(f"Dataset {dataset_key} missing corpus_field '{corpus_field}' in row {i}")

        query_id = f"q_{i}"
        doc_id = str(row[id_field]).strip()
        
        query_text = str(row.get(query_field, "")).strip()
        corpus_text = str(row.get(corpus_field, "")).strip()

        if not query_text or not corpus_text:
            continue

        queries[query_id] = query_text
        corpus[doc_id] = corpus_text
        qrels[query_id] = [doc_id]

    return RetrievalDataset(
        name=dataset_key,
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        metadata={
            "description": config["description"],
            "hf_path": config["hf_path"],
            "split": split,
            "num_queries": len(queries),
            "num_docs": len(corpus),
        },
    )


def load_multiple_famteb_datasets(
    dataset_keys: List[str],
    max_samples_per_dataset: Optional[int] = None,
) -> Dict[str, RetrievalDataset]:
    """Load multiple FaMTEB datasets."""
    datasets = {}
    for key in dataset_keys:
        try:
            datasets[key] = load_famteb_retrieval_dataset(
                key, max_samples=max_samples_per_dataset
            )
            print(f"Loaded {key}: {len(datasets[key].queries)} queries, "
                  f"{len(datasets[key].corpus)} docs")
        except Exception as e:
            print(f"Failed to load {key}: {e}")
    return datasets


# ---------------------------------------------------------------------------
# Benchmark Integration
# ---------------------------------------------------------------------------

def convert_to_benchmark_format(dataset: RetrievalDataset) -> List[Dict[str, Any]]:
    """Convert RetrievalDataset to benchmark JSON format."""
    samples = []
    for query_id, query in dataset.queries.items():
        samples.append({
            "query": dataset.queries[query_id],
            "expected_chunk_ids": dataset.qrels.get(query_id, []),
            "expected_answer": "",
            "relevance_scores": {doc_id: 1.0 for doc_id in dataset.qrels.get(query_id, [])},
            "category": "factual",
            "difficulty": "medium",
            "format": "famteb",
            "gt": dataset.queries[query_id],
            "keywords": "",
            "gt_similarity": 1.0,
            "dataset_name": dataset.name,
        })
    return samples


def load_famteb_benchmark(
    dataset_keys: Optional[List[str]] = None,
    max_samples_per_dataset: int = 100,
) -> List[Dict[str, Any]]:
    """Load multiple FaMTEB datasets and convert to benchmark format.

    Args:
        dataset_keys: List of dataset keys to load. Defaults to all retrieval datasets.
        max_samples_per_dataset: Max samples per dataset.

    Returns:
        List of benchmark samples in standard format.
    """
    if dataset_keys is None:
        dataset_keys = list(FAMTEB_RETRIEVAL_DATASETS.keys())

    datasets = load_multiple_famteb_datasets(
        dataset_keys, max_samples_per_dataset
    )

    all_samples = []
    for key, dataset in datasets.items():
        samples = convert_to_benchmark_format(dataset)
        for s in samples:
            s["format"] = f"famteb_{dataset.name}"
        all_samples.extend(samples)

    return all_samples


# ---------------------------------------------------------------------------
# Save/Load Benchmark Datasets
# ---------------------------------------------------------------------------

def save_benchmark_dataset(
    samples: List[Dict[str, Any]],
    path: str,
) -> None:
    """Save benchmark samples to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def load_benchmark_dataset(path: str) -> List[Dict[str, Any]]:
    """Load benchmark samples from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Evaluation Metrics Extension
# ---------------------------------------------------------------------------

def compute_map_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int = 10,
) -> float:
    """Compute Average Precision at k."""
    if not relevant:
        return 0.0
    
    score = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def compute_recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int = 100,
) -> float:
    """Compute Recall at k."""
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(set(relevant) & retrieved_set) / len(relevant)


def compute_mrr(
    retrieved: List[str],
    relevant: List[str],
) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int = 10,
) -> float:
    """Compute NDCG@K with binary relevance."""
    import math
    
    if not relevant:
        return 0.0
    
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    
    # IDCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_all_metrics(
    retrieved: List[str],
    relevant: List[str],
    k_values: List[int] = [1, 3, 5, 10, 20, 100],
) -> Dict[str, float]:
    """Compute all retrieval metrics for a single query."""
    metrics = {}
    for k in k_values:
        metrics[f"hit@{k}"] = float(any(d in relevant for d in retrieved[:k]))
        metrics[f"recall@{k}"] = compute_recall_at_k(retrieved, relevant, k)
        metrics[f"map@{k}"] = compute_map_at_k(retrieved, relevant, k)
        metrics[f"ndcg@{k}"] = compute_ndcg_at_k(retrieved, relevant, k)
    metrics["mrr"] = compute_mrr(retrieved, relevant)
    return metrics