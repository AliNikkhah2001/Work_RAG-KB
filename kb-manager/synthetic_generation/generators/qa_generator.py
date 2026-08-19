"""QA pair generator for synthetic data generation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from synthetic_generation.generators.base import BaseGenerator, GenerationResult

logger = logging.getLogger(__name__)


class QAGenerator:
    """Generates QA pairs from chunk content."""

    def __init__(
        self,
        llm_client: Any,
        config: Dict[str, Any],
    ) -> None:
        self._llm = llm_client
        self._config = config
        
        # Load prompt template
        from pathlib import Path
        prompt_path = Path(__file__).parent.parent / "prompts" / "qa_generation.txt"
        self._prompt_template = Path(__file__).parent.parent.joinpath("prompts", "qa_generation.txt").read_text(encoding="utf-8")
        
        # Query type weights
        self._type_weights = config.get("generation", {}).get("query_types", {
            "verbatim": 0.15,
            "paraphrase": 0.20,
            "conversational": 0.20,
            "typo": 0.15,
            "keyword_only": 0.15,
            "reworded": 0.15,
        })
        
        self._num_queries = config.get("generation", {}).get("num_samples_per_chunk", 8)
        self._min_per_type = max(1, self._num_queries // len(self._type_weights))

    def generate(self, chunk_content: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Generate QA pairs for a chunk."""
        # Build prompt with chunk content
        prompt = self._build_prompt(chunk_content)
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse response
        try:
            queries = self._parse_response(response, chunk_id)
            return queries
        except Exception as e:
            logger.warning(f"Failed to parse QA generation for chunk {chunk_id}: {e}")
            return self._fallback_queries(chunk_id)

    def _build_prompt(self, chunk_content: str) -> str:
        """Build prompt for QA generation."""
        num_queries = self._config.get("generation", {}).get("num_samples_per_chunk", 8)
        min_per_type = max(1, self._num_queries // 6)
        
        # Truncate chunk if too long
        max_len = self._config.get("generation", {}).get("max_chunk_length", 2000)
        content = chunk_content[:max_len]
        
        # Load prompt template
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "prompts" / "qa_generation.txt"
        template = Path(template_path).read_text(encoding="utf-8")
        
        return template.format(
            chunk_content=content,
            num_queries=self._num_queries,
            min_per_type=max(1, self._num_queries // 6),
        )

    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return text response."""
        try:
            response = self._llm.generate(
                prompt,
                max_tokens=1024,
                temperature=0.4,
            )
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _parse_response(self, response: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Parse LLM response into list of queries."""
        import json
        import re
        
        text = str(response).strip()
        
        # Try to parse as JSON
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return self._normalize_queries(data, chunk_id)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array
        import re
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return self._normalize_queries(data, chunk_id)
            except json.JSONDecodeError:
                pass
        
        # Fallback
        logger.warning("Failed to parse LLM response as JSON")
        return self._fallback_queries()

    def _normalize_queries(self, queries: List[Dict], chunk_id: str) -> List[Dict]:
        """Normalize query format and add metadata."""
        normalized = []
        for q in queries:
            if not isinstance(q, dict):
                continue
            query_text = q.get("query", "").strip()
            qtype = q.get("type", "unknown")
            
            if not query_text:
                continue
            
            normalized.append({
                "query": query_text,
                "type": qtype,
                "chunk_id": chunk_id,
                "expected_chunk_id": chunk_id,
            })
        return normalized

    def _fallback_queries(self, chunk_id: str = "") -> List[Dict[str, Any]]:
        """Fallback queries when generation fails."""
        types = ["verbatim", "paraphrase", "conversational", "typo", "keyword_only", "reworded"]
        return [
            {"query": f"فallback query {i}", "type": t, "chunk_id": chunk_id, "expected_chunk_id": chunk_id}
            for i, t in enumerate(types)
        ]