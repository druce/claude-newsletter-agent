# llm.py
"""Multi-vendor async LLM wrapper with structured output, batch processing, and retry logic."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
