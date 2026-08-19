"""Quality validator for synthetic data (LLM-as-judge)."""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation."""
    relevance_score: float      # 0-1: semantic relevance to target
    accuracy_score: float       # 0-1: factual accuracy
    naturalness_score: float    # 0-1: natural Persian
    diversity_score: float      # 0-1: diversity from other samples
    overall_score: float        # 0-1: weighted average
    passes_threshold: bool      # overall >= threshold
    feedback: str               # qualitative feedback


class SyntheticValidator:
    """Validates synthetic data quality using LLM-as-judge."""

    def __init__(
        self,
        llm_client: Any,
        config: Dict[str, Any],
    ) -> None:
        self._llm = llm_client
        self._threshold = config.get("generation", {}).get("validation", {}).get("threshold", 0.7)
        self._validator_model = config.get("generation", {}).get("validation", {}).get("validator_model", "gemma2:27b")
        
        self._prompt_template = Path(__file__).parent.parent.joinpath("prompts", "validation.txt").read_text(encoding="utf-8")
        
        self._cache: Dict[str, ValidationResult] = {}

    def validate_qa(self, query: str, target_chunk: str, chunk_id: str) -> ValidationResult:
        """Validate a single QA pair."""
        cache_key = f"qa:{chunk_id}:{hash(query)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._validate(query, target_chunk)
        self._cache[cache_key] = result
        return result

    def validate_conversation(self, conversation: List[Dict], target_chunk: str, chunk_id: str) -> ValidationResult:
        """Validate a conversation."""
        # Use last user turn for validation
        user_turns = [t["content"] for t in conversation if t["role"] == "user"]
        if not user_turns:
            return ValidationResult(
                relevance_score=0, accuracy_score=0, naturalness_score=0,
                diversity_score=0, overall_score=0, passes_threshold=False,
                feedback="No user turns found"
            )
        
        # Validate last user turn (most specific)
        return self.validate_qa(user_turns[-1], target_chunk, chunk_id)

    def _validate(self, generated_text: str, target_chunk: str) -> ValidationResult:
        """Run LLM-as-judge validation."""
        prompt = self._build_prompt(generated_text, target_chunk)
        
        try:
            response = self._call_llm(prompt)
            return self._parse_validation(response)
        except Exception as e:
            logger.warning(f"Validation LLM call failed: {e}")
            return ValidationResult(
                relevance_score=0, accuracy_score=0, naturalness_score=0,
                diversity_score=0, overall_score=0, passes_threshold=False,
                feedback=f"Validation failed: {e}"
            )

    def _build_prompt(self, generated_text: str, target_chunk: str) -> str:
        template_path = Path(__file__).parent.parent / "prompts" / "validation.txt"
        template = Path(template_path).read_text(encoding="utf-8")
        
        return template.format(
            target_chunk=target_chunk[:2000],
            generated_text=generated_text[:500],
        )

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self._llm.generate(
                prompt,
                max_tokens=512,
                temperature=0.2,  # Low temperature for consistent judging
            )
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"Validation LLM call failed: {e}")
            raise

    def _parse_validation(self, response: str) -> ValidationResult:
        """Parse validation response."""
        text = str(response).strip()
        
        try:
            # Try to extract JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse validation JSON: {text[:200]}")
            return ValidationResult(
                relevance_score=0, accuracy_score=0, naturalness_score=0,
                diversity_score=0, overall_score=0, passes_threshold=False,
                feedback="Failed to parse validation response"
            )
        
        scores = [
            data.get("relevance_score", 0),
            data.get("accuracy_score", 0),
            data.get("naturalness_score", 0),
            data.get("diversity_score", 0),
        ]
        overall = sum(scores) / len(scores) if scores else 0
        
        return ValidationResult(
            relevance_score=data.get("relevance_score", 0),
            accuracy_score=data.get("accuracy_score", 0),
            naturalness_score=data.get("naturalness_score", 0),
            diversity_score=data.get("diversity_score", 0),
            overall_score=overall,
            passes_threshold=overall >= self._threshold,
            feedback=data.get("feedback", ""),
        )

    def filter_batch(
        self,
        samples: List[Dict[str, Any]],
        chunk_texts: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Filter a batch of samples, keeping only those that pass validation."""
        passed = []
        for sample in samples:
            chunk_id = sample.get("chunk_id") or sample.get("target_chunk_id") or sample.get("expected_chunk_id")
            if not chunk_id or chunk_id not in chunk_texts:
                continue
            
            # Get text to validate
            if "conversation" in sample:
                result = self.validate_conversation(sample["conversation"], chunk_texts[chunk_id], chunk_id)
            else:
                query = sample.get("query", "")
                result = self.validate_qa(query, chunk_texts[chunk_id], chunk_id)
            
            if result.passes_threshold:
                sample["validation"] = {
                    "overall_score": result.overall_score,
                    "relevance": result.relevance_score,
                    "accuracy": result.accuracy_score,
                    "naturalness": result.naturalness_score,
                    "diversity": result.diversity_score,
                }
                passed.append(sample)
            else:
                logger.debug(f"Sample rejected: {result.feedback} (score: {result.overall_score:.2f})")
        
        logger.info(f"Validation: {len(passed)}/{len(samples)} samples passed (threshold={self._threshold})")
        return passed

    def get_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get validation statistics for a batch."""
        if not samples:
            return {}
        
        scores = [s.get("validation", {}).get("overall_score", 0) for s in samples if "validation" in s]
        if not scores:
            return {"count": 0}
        
        import statistics
        return {
            "count": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "min": min(scores),
            "max": max(scores),
            "passed": sum(1 for s in scores if s >= 0.7),
            "failed": sum(1 for s in scores if s < 0.7),
        }