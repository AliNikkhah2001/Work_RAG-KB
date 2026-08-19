"""Conversational generator for multi-turn synthetic conversations."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from synthetic_generation.generators.base import BaseGenerator, GenerationResult

logger = logging.getLogger(__name__)


class ConversationalGenerator:
    """Generates multi-turn conversational samples."""

    def __init__(
        self,
        llm_client: Any,
        config: Dict[str, Any],
    ) -> None:
        self._llm = llm_client
        self._config = config
        
        self._num_turns = config.get("generation", {}).get("conversation", {}).get("num_turns", 3)
        self._include_coref = config.get("generation", {}).get("conversation", {}).get("include_coreference", True)
        self._include_ellipsis = config.get("generation", {}).get("conversation", {}).get("include_ellipsis", True)

    def generate(self, question: str, answer: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Generate a multi-turn conversation from a QA pair."""
        prompt = self._build_prompt(question, answer)
        
        try:
            response = self._call_llm(prompt)
            return self._parse_response(response, chunk_id)
        except Exception as e:
            logger.warning(f"Conversational generation failed for chunk {chunk_id}: {e}")
            return self._fallback_conversation(question, answer, chunk_id)

    def _build_prompt(self, question: str, answer: str) -> str:
        """Build prompt for conversational generation."""
        template_path = Path(__file__).parent.parent / "prompts" / "conversational.txt"
        template = Path(template_path).read_text(encoding="utf-8")
        
        return template.format(
            question=question,
            answer=answer,
            num_turns=self._num_turns,
            chunk_id="",  # Will be filled after
        )

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self._llm.generate(
                prompt,
                max_tokens=1024,
                temperature=0.5,
            )
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _parse_response(self, response: str, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into conversation format."""
        text = str(response).strip()
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            else:
                return None
        
        # Validate required fields
        if not isinstance(data, dict):
            return None
        
        conversation = data.get("conversation", [])
        if not isinstance(conversation, list) or len(conversation) < 3:
            return None
        
        # Validate conversation format
        for turn in conversation:
            if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                return None
            if turn["role"] not in ("user", "assistant"):
                return None
        
        return {
            "conversation": conversation,
            "target_chunk_id": chunk_id,
            "topic": data.get("topic", ""),
            "coreference_used": data.get("coreference_used", True),
            "ellipsis_used": data.get("ellipsis_used", True),
            "num_turns": len(conversation) // 2,
        }

    def _fallback_conversation(self, question: str, answer: str, chunk_id: str) -> Dict[str, Any]:
        """Fallback conversation when generation fails."""
        return {
            "conversation": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "برای حقوقی هم همینه؟"},
                {"role": "assistant", "content": "بله، این правило برای هر دو مدل صدق می‌کند."},
                {"role": "user", "content": "جزئیات بیشتر بده"},
                {"role": "assistant", "content": answer[:200] + "..."},
            ],
            "target_chunk_id": chunk_id,
            "topic": "fallback",
            "coreference_used": True,
            "ellipsis_used": True,
            "num_turns": 3,
        }