"""Base classes for synthetic data generators."""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation task."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error: Optional[str] = None


class BaseGenerator(abc.ABC):
    """Abstract base class for all generators."""

    def __init__(
        self,
        llm_client: Any,
        prompt_template: str,
        config: Dict[str, Any],
    ) -> None:
        self._llm = llm_client
        self._prompt_template = prompt_template
        self._config = config

    @abc.abstractmethod
    def generate(self, input_data: Dict[str, Any]) -> GenerationResult:
        """Generate synthetic data from input."""
        pass

    def _load_prompt(self, prompt_name: str) -> str:
        """Load prompt template from file."""
        prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Prompt template not found: {prompt_name}")

    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response, handling common issues."""
        import re
        text = str(response).strip()
        
        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
        code_block = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array or object in text
        for pattern in [r'\[.*\]', r'\{.*\}']:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
        
        raise ValueError(f"Could not parse JSON from response: {response[:200]}...")

    def _build_prompt(self, template: str, **kwargs) -> str:
        """Format prompt template with variables."""
        return self._prompt_template.format(**kwargs)

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.4,
    ) -> str:
        """Call LLM and return text response."""
        from kb_manager.llm import LLMResponse
        
        response = self._llm.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.text if isinstance(response, str) else response.text