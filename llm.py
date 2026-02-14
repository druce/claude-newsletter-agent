# llm.py
"""Multi-vendor async LLM wrapper with structured output, batch processing, and retry logic."""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Vendor(str, Enum):
    """Supported LLM vendors."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass(frozen=True)
class LLMModel:
    """Model identity and capabilities."""
    model_id: str
    vendor: Vendor
    supports_logprobs: bool
    supports_reasoning: bool
    default_max_tokens: int
    display_name: str


# --- Pre-defined model constants ---

CLAUDE_SONNET_MODEL = LLMModel(
    model_id="claude-sonnet-4-5-20250929",
    vendor=Vendor.ANTHROPIC,
    supports_logprobs=False,
    supports_reasoning=True,
    default_max_tokens=8192,
    display_name="Claude Sonnet 4.5",
)

CLAUDE_HAIKU_MODEL = LLMModel(
    model_id="claude-haiku-4-5-20251001",
    vendor=Vendor.ANTHROPIC,
    supports_logprobs=False,
    supports_reasoning=True,
    default_max_tokens=8192,
    display_name="Claude Haiku 4.5",
)

GPT5_MINI_MODEL = LLMModel(
    model_id="gpt-5-mini",
    vendor=Vendor.OPENAI,
    supports_logprobs=True,
    supports_reasoning=True,
    default_max_tokens=4096,
    display_name="GPT-5 Mini",
)

GPT41_MINI_MODEL = LLMModel(
    model_id="gpt-4.1-mini",
    vendor=Vendor.OPENAI,
    supports_logprobs=True,
    supports_reasoning=False,
    default_max_tokens=4096,
    display_name="GPT-4.1 Mini",
)

GEMINI_FLASH_MODEL = LLMModel(
    model_id="gemini-2.0-flash",
    vendor=Vendor.GEMINI,
    supports_logprobs=False,
    supports_reasoning=True,
    default_max_tokens=8192,
    display_name="Gemini 2.0 Flash",
)

GEMINI_PRO_MODEL = LLMModel(
    model_id="gemini-2.5-pro",
    vendor=Vendor.GEMINI,
    supports_logprobs=False,
    supports_reasoning=True,
    default_max_tokens=8192,
    display_name="Gemini 2.5 Pro",
)


# --- Reasoning effort mapping ---

_ANTHROPIC_THINKING_MAP = {0: 0, 2: 1000, 4: 2000, 6: 4000, 8: 8000, 10: 16000}
_OPENAI_REASONING_MAP = {0: None, 2: "low", 4: "low", 6: "medium", 8: "high", 10: "high"}
_GEMINI_THINKING_MAP = {0: 0, 2: 1024, 4: 2048, 6: 4096, 8: 8192, 10: 16384}


def _validate_effort(effort: int) -> int:
    if effort < 0 or effort > 10:
        raise ValueError(f"reasoning_effort must be 0-10, got {effort}")
    # Round to nearest even number
    return round(effort / 2) * 2


def _anthropic_thinking_tokens(effort: int) -> int:
    return _ANTHROPIC_THINKING_MAP[_validate_effort(effort)]


def _openai_reasoning_effort(effort: int) -> Optional[str]:
    return _OPENAI_REASONING_MAP[_validate_effort(effort)]


def _gemini_thinking_tokens(effort: int) -> int:
    return _GEMINI_THINKING_MAP[_validate_effort(effort)]


# --- Abstract base class ---

class LLMAgent(ABC):
    """Vendor-agnostic async LLM agent with structured output and retry logic."""

    _fatal_exceptions: Tuple[Type[Exception], ...] = ()
    _temporary_exceptions: Tuple[Type[Exception], ...] = ()

    def __init__(
        self,
        model: LLMModel,
        system_prompt: str,
        user_prompt: str,
        output_type: Optional[Type[BaseModel]] = None,
        reasoning_effort: int = 0,
        max_concurrency: int = 12,
        temperature: float = 0.0,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.output_type = output_type
        self.reasoning_effort = reasoning_effort
        # Validate reasoning effort early
        if reasoning_effort < 0 or reasoning_effort > 10:
            raise ValueError(f"reasoning_effort must be 0-10, got {reasoning_effort}")
        self.max_concurrency = max_concurrency
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @abstractmethod
    async def _call_llm(
        self, system: str, user: str, output_schema: Optional[Dict[str, Any]]
    ) -> Any:
        """Vendor-specific API call. Returns raw response object."""
        ...

    @abstractmethod
    async def _parse_structured(self, raw: Any, output_type: Type[BaseModel]) -> BaseModel:
        """Vendor-specific structured output parsing."""
        ...

    @abstractmethod
    async def _parse_logprobs(self, raw: Any, target_tokens: List[str]) -> Dict[str, float]:
        """Vendor-specific logprob extraction."""
        ...
