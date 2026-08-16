"""Synthetic test data generator for retrieval evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvalQuery:
    """A single evaluation query with expected results."""

    query: str
    expected_chunk_ids: list[str]
    expected_answer: str = ""
    relevance_scores: dict[str, float] = field(default_factory=dict)
    category: str = "factual"  # factual, procedural, comparative
    difficulty: str = "medium"  # easy, medium, hard


class SyntheticDataGenerator:
    """Generate synthetic evaluation datasets from ingested chunks.

    Methods:
    - generate_from_chunks: Create queries from existing QA chunks
    - generate_paraphrases: Create paraphrase variations
    - generate_negative_samples: Add irrelevant chunks as negatives
    - save_dataset: Store evaluation dataset as JSON
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def generate_from_chunks(
        self,
        output_path: str,
        max_queries: int = 100,
    ) -> list[EvalQuery]:
        """Generate evaluation queries from existing QA chunks.

        Each QA chunk becomes a query (using its question field) with
        the chunk itself as the expected result.
        """
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT id, content, metadata FROM chunks "
            "WHERE chunk_type = 'qa_pair' "
            "ORDER BY RANDOM() LIMIT ?",
            (max_queries,),
        )
        rows = cur.fetchall()
        conn.close()

        queries: list[EvalQuery] = []
        for chunk_id, content, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            fields = meta.get("fields", {})

            question = fields.get("question", "")
            answer = fields.get("answer", "")
            keywords = fields.get("keyword", "")

            if not question:
                continue

            query = EvalQuery(
                query=question,
                expected_chunk_ids=[chunk_id],
                expected_answer=answer,
                relevance_scores={chunk_id: 1.0},
                category="factual",
                difficulty="easy",
            )
            queries.append(query)

        self._save_dataset(queries, output_path)
        return queries

    def generate_paraphrases(
        self,
        queries: list[EvalQuery],
        output_path: str,
        paraphrases_per_query: int = 3,
    ) -> list[EvalQuery]:
        """Generate paraphrase variations of existing queries.

        Creates slight variations to test robustness to wording changes.
        """
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        # Persian paraphrase templates
        prefixes = [
            "\u0644\u0637\u0641\u0627",  # لطفا
            "\u0628\u0627\u06cc\u062f",  # باید
            "\u0622\u06cc\u0627 \u0627\u0633\u062a",  # آیا است
            "\u062a\u0635\u0645\u06cc\u0645 \u06a9\u0646\u06cc\u062f",  # توضیح دهید
            "\u0628\u06af\u0648\u06cc\u06cc\u062f",  # بگویید
        ]

        suffixes = [
            "\u0631\u0627 \u062a\u0635\u0645\u06cc\u0645 \u062f\u0647\u06cc\u062f",
            "\u0628\u0647 \u0635\u0648\u0631\u062a \u062a\u0648\u0636\u06cc\u062d \u062f\u0647\u06cc\u062f",
            "\u0628\u0627 \u062c\u0632\u0626\u06cc\u0627\u062a \u062a\u0648\u0636\u06cc\u062d \u062f\u0645\u06cc\u062f\u0647 \u0627\u0633\u062a",
        ]

        paraphrased: list[EvalQuery] = []
        for q in queries:
            words = q.query.split()
            for _ in range(paraphrases_per_query):
                # Simple paraphrase: add prefix or suffix, or shuffle middle words
                if len(words) > 3 and random.random() > 0.5:
                    # Shuffle middle words
                    mid = words[1:-1]
                    random.shuffle(mid)
                    new_query = " ".join([words[0]] + mid + [words[-1]])
                else:
                    # Add prefix/suffix
                    new_query = f"{random.choice(prefixes)} {q.query} {random.choice(suffixes)}"

                paraphrased.append(
                    EvalQuery(
                        query=new_query,
                        expected_chunk_ids=q.expected_chunk_ids,
                        expected_answer=q.expected_answer,
                        relevance_scores=q.relevance_scores,
                        category=q.category,
                        difficulty="medium",
                    )
                )

        all_queries = queries + paraphrased
        self._save_dataset(all_queries, output_path)
        return all_queries

    def generate_negative_samples(
        self,
        queries: list[EvalQuery],
        output_path: str,
        negatives_per_query: int = 2,
    ) -> list[EvalQuery]:
        """Add irrelevant chunks as negative samples for precision evaluation."""
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        cur = conn.cursor()

        # Get all chunk IDs
        cur.execute("SELECT id FROM chunks")
        all_ids = [row[0] for row in cur.fetchall()]
        conn.close()

        enriched: list[EvalQuery] = []
        for q in queries:
            # Sample negative chunk IDs (not in expected results)
            negatives = [
                cid for cid in all_ids if cid not in q.expected_chunk_ids
            ]
            sampled_neg = random.sample(
                negatives, min(negatives_per_query, len(negatives))
            )

            # Update relevance scores
            new_scores = dict(q.relevance_scores)
            for neg_id in sampled_neg:
                new_scores[neg_id] = 0.0

            enriched.append(
                EvalQuery(
                    query=q.query,
                    expected_chunk_ids=q.expected_chunk_ids + sampled_neg,
                    expected_answer=q.expected_answer,
                    relevance_scores=new_scores,
                    category=q.category,
                    difficulty=q.difficulty,
                )
            )

        self._save_dataset(enriched, output_path)
        return enriched

    def generate_full_dataset(
        self,
        output_path: str,
        max_queries: int = 100,
    ) -> list[EvalQuery]:
        """Generate a complete evaluation dataset with all methods."""
        # Step 1: Generate base queries from chunks
        queries = self.generate_from_chunks(output_path, max_queries)

        # Step 2: Add paraphrases
        queries = self.generate_paraphrases(queries, output_path)

        # Step 3: Add negative samples
        queries = self.generate_negative_samples(queries, output_path)

        return queries

    def _save_dataset(self, queries: list[EvalQuery], output_path: str) -> None:
        """Save evaluation dataset as JSON."""
        data = [
            {
                "query": q.query,
                "expected_chunk_ids": q.expected_chunk_ids,
                "expected_answer": q.expected_answer,
                "relevance_scores": q.relevance_scores,
                "category": q.category,
                "difficulty": q.difficulty,
            }
            for q in queries
        ]
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
