# llm.py
"""Multi-vendor async LLM wrapper with structured output, batch processing, and retry logic."""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import create_model

import math
import pandas as pd

import anthropic
import openai
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
# opus currently supports up to 128k output tokens, others 64k
# but errors above 16k with, requires streaming if it might take longer than 10 minutes
# mostly should return id and structured output, shouldn't need long context
# https://platform.claude.com/docs/en/about-claude/models/overview
CLAUDE_OPUS_MODEL = LLMModel(
    model_id="claude-opus-4-6",
    vendor=Vendor.ANTHROPIC,
    supports_logprobs=False,
    supports_reasoning=True,
    default_max_tokens=16384,
    display_name="Claude Sonnet 4.5",
)

CLAUDE_SONNET_MODEL = LLMModel(
    model_id="claude-sonnet-4-5-20250929",
    vendor=Vendor.ANTHROPIC,
    supports_logprobs=False,
    supports_reasoning=True,
    default_max_tokens=16384,
    display_name="Claude Sonnet 4.5",
)

CLAUDE_HAIKU_MODEL = LLMModel(
    model_id="claude-haiku-4-5-20251001",
    vendor=Vendor.ANTHROPIC,
    supports_logprobs=False,
    supports_reasoning=True,
    default_max_tokens=16384,
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

MODEL_DICT: Dict[str, LLMModel] = {
    CLAUDE_OPUS_MODEL.model_id: CLAUDE_OPUS_MODEL,
    CLAUDE_SONNET_MODEL.model_id: CLAUDE_SONNET_MODEL,
    CLAUDE_HAIKU_MODEL.model_id: CLAUDE_HAIKU_MODEL,
    GPT5_MINI_MODEL.model_id: GPT5_MINI_MODEL,
    GPT41_MINI_MODEL.model_id: GPT41_MINI_MODEL,
    GEMINI_FLASH_MODEL.model_id: GEMINI_FLASH_MODEL,
    GEMINI_PRO_MODEL.model_id: GEMINI_PRO_MODEL,
}

# --- Reasoning effort mapping ---

_ANTHROPIC_THINKING_MAP = {-1: 0, 0: 0, 2: 1024, 4: 2048, 6: 4096, 8: 8192, 10: 16384}
_OPENAI_REASONING_MAP = {
    -1: None,
    0: None,
    2: "low",
    4: "low",
    6: "medium",
    8: "high",
    10: "high",
}
_GEMINI_THINKING_MAP = {-1: 0, 0: 0, 2: 1024, 4: 2048, 6: 4096, 8: 8192, 10: 16384}


def _validate_effort(effort: int) -> int:
    if effort == -1:
        return -1
    if effort < 0 or effort > 10:
        raise ValueError(
            f"reasoning_effort must be 0-10 (or -1 to disable), got {effort}"
        )
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
        if reasoning_effort < -1 or reasoning_effort > 10:
            raise ValueError(
                f"reasoning_effort must be 0-10 (or -1 to disable), got {reasoning_effort}"
            )
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
    async def _parse_structured(
        self, raw: Any, output_type: Type[BaseModel]
    ) -> BaseModel:
        """Vendor-specific structured output parsing."""
        ...

    @abstractmethod
    async def _parse_logprobs(
        self, raw: Any, target_tokens: List[str]
    ) -> Dict[str, float]:
        """Vendor-specific logprob extraction."""
        ...

    async def prompt_dict(self, variables: Optional[Dict[str, Any]] = None) -> Any:
        """Single prompt with variable substitution. Returns structured output or raw string."""
        user_text = self.user_prompt.format(**(variables or {}))
        output_schema = None
        if self.output_type:
            output_schema = self.output_type.model_json_schema()

        raw = await self._call_with_retry(self.system_prompt, user_text, output_schema)

        if self.output_type:
            return await self._parse_structured(raw, self.output_type)
        # For text responses, subclass provides _extract_text; fallback to str
        if hasattr(self, "_extract_text"):
            return self._extract_text(raw)
        return str(raw)

    async def _call_with_retry(
        self, system: str, user: str, output_schema: Optional[Dict[str, Any]]
    ) -> Any:
        """Call LLM with tenacity retry on temporary exceptions."""

        @retry(
            retry=retry_if_exception_type(self._temporary_exceptions),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            reraise=True,
        )
        async def _do_call():
            return await self._call_llm(system, user, output_schema)

        return await _do_call()

    async def prompt_list(
        self,
        items: List[Dict[str, Any]],
        item_id_field: str = "id",
        item_list_field: str = "results_list",
        max_retries: int = 3,
    ) -> List[Dict[str, Any]]:
        """Process a list of items through the LLM. Returns list of result dicts.

        Retries on ID mismatch (LLM dropped/transposed items) with exponential backoff.
        """
        input_text = json.dumps(items)
        user_text = self.user_prompt.format(input_text=input_text)
        output_schema = None
        if self.output_type:
            output_schema = self.output_type.model_json_schema()

        sent_ids = [item[item_id_field] for item in items if item_id_field in item]

        for attempt in range(max_retries):
            raw = await self._call_with_retry(
                self.system_prompt, user_text, output_schema
            )

            if self.output_type:
                parsed = await self._parse_structured(raw, self.output_type)
                raw_results = getattr(parsed, item_list_field)
                # Convert Pydantic models to dicts for uniform downstream access
                results = [
                    r.model_dump() if hasattr(r, "model_dump") else r
                    for r in raw_results
                ]
            else:
                results = json.loads(
                    self._extract_text(raw)
                    if hasattr(self, "_extract_text")
                    else str(raw)
                )
                results = results[item_list_field]

            # Validate IDs if present
            if sent_ids:
                returned_ids = [r[item_id_field] for r in results if item_id_field in r]
                if sent_ids != returned_ids:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        logger.warning(
                            f"ID mismatch (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait}s: sent {sent_ids}, got {returned_ids}"
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise ValueError(
                        f"ID mismatch: sent {sent_ids}, got {returned_ids}"
                    )

            return results

        return results  # unreachable, but satisfies type checker

    async def filter_dataframe(
        self,
        df: "pd.DataFrame",
        chunk_size: int = 25,
        value_field: str = "output",
        item_list_field: str = "results_list",
        item_id_field: str = "id",
    ) -> "pd.Series":
        """Process DataFrame in chunks, return Series aligned to original index."""

        chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

        async def _process_chunk(chunk_df):
            async with self._semaphore:
                items = chunk_df.to_dict("records")
                results = await self.prompt_list(items, item_id_field, item_list_field)
                return results

        all_results = await asyncio.gather(*[_process_chunk(c) for c in chunks])

        flat_results = []
        for chunk_results in all_results:
            flat_results.extend(chunk_results)

        values = [r.get(value_field) for r in flat_results]
        return pd.Series(values, index=df.index)

    async def run_prompt_with_probs(
        self,
        variables: Optional[Dict[str, Any]] = None,
        target_tokens: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Get token probabilities. Native logprobs for OpenAI, simulated confidence otherwise."""
        target_tokens = target_tokens or []
        user_text = self.user_prompt.format(**(variables or {}))

        if self.model.supports_logprobs:
            raw = await self._call_with_retry(self.system_prompt, user_text, None)
            return await self._parse_logprobs(raw, target_tokens)
        else:
            ConfidenceModel = create_model("ConfidenceModel", confidence=(float, ...))
            schema = ConfidenceModel.model_json_schema()

            raw = await self._call_with_retry(self.system_prompt, user_text, schema)
            parsed = await self._parse_structured(raw, ConfidenceModel)
            confidence = parsed.confidence

            result = {}
            if target_tokens:
                result[target_tokens[0]] = confidence
            return result


class AnthropicAgent(LLMAgent):
    """Anthropic Claude agent using tool_use for structured output."""

    _fatal_exceptions = (
        anthropic.AuthenticationError,
        anthropic.BadRequestError,
    )
    _temporary_exceptions = (
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = anthropic.AsyncAnthropic()

    async def _call_llm(self, system, user, output_schema):
        kwargs = {
            "model": self.model.model_id,
            "max_tokens": self.model.default_max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        # Reasoning effort → thinking budget
        thinking_tokens = _anthropic_thinking_tokens(self.reasoning_effort)
        if thinking_tokens > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_tokens}
            # Anthropic requires temperature=1 when thinking is enabled
        else:
            kwargs["temperature"] = self.temperature

        # Structured output via tool_use
        # Note: Anthropic forbids tool_choice + thinking in the same request.
        # When thinking is enabled, we skip tool_choice and rely on the system
        # prompt instructing the model to return valid JSON.
        if output_schema and thinking_tokens <= 0:
            kwargs["tools"] = [
                {
                    "name": "structured_output",
                    "description": "Return structured data",
                    "input_schema": output_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        return await self._client.messages.create(**kwargs)

    async def _parse_structured(self, raw, output_type):
        for block in raw.content:
            if block.type == "tool_use":
                return output_type.model_validate(block.input)
        # Fallback: when thinking is enabled, tool_choice is not used,
        # so the response is plain text JSON instead of a tool_use block.
        text = self._extract_text(raw)
        return output_type.model_validate_json(text)

    async def _parse_logprobs(self, raw, target_tokens):
        # Anthropic doesn't support native logprobs; confidence is in structured output
        return {}

    def _extract_text(self, raw) -> str:
        """Extract text from Anthropic response, skipping thinking blocks."""
        for block in raw.content:
            if block.type == "text":
                return block.text
        return ""


class OpenAIAgent(LLMAgent):
    """OpenAI agent using json_schema for structured output and native logprobs."""

    _fatal_exceptions = (
        openai.AuthenticationError,
        openai.BadRequestError,
    )
    _temporary_exceptions = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = openai.AsyncOpenAI()

    async def _call_llm(self, system, user, output_schema):
        kwargs = {
            "model": self.model.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }

        # Reasoning effort for o-series models
        effort = _openai_reasoning_effort(self.reasoning_effort)
        if effort and self.model.supports_reasoning:
            kwargs["reasoning_effort"] = effort

        # Structured output via json_schema
        if output_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": output_schema,
                },
            }

        return await self._client.chat.completions.create(**kwargs)

    async def _parse_structured(self, raw, output_type):
        content = raw.choices[0].message.content
        return output_type.model_validate_json(content)

    async def _parse_logprobs(self, raw, target_tokens):
        probs = {}
        logprobs_data = raw.choices[0].logprobs
        if not logprobs_data or not logprobs_data.content:
            return probs
        for item in logprobs_data.content:
            for top in item.top_logprobs:
                if top.token in target_tokens:
                    probs[top.token] = math.exp(top.logprob)
        return probs

    def _extract_text(self, raw) -> str:
        return raw.choices[0].message.content


class GeminiAgent(LLMAgent):
    """Google Gemini agent using response_schema for structured output."""

    _fatal_exceptions = ()
    _temporary_exceptions = ()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = genai.Client()

    async def _call_llm(self, system, user, output_schema):
        config_kwargs = {
            "temperature": self.temperature,
            "system_instruction": system,
        }

        thinking_tokens = _gemini_thinking_tokens(self.reasoning_effort)
        if thinking_tokens > 0:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_tokens
            )

        if output_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = output_schema

        config = genai_types.GenerateContentConfig(**config_kwargs)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=self.model.model_id,
                contents=user,
                config=config,
            ),
        )

    async def _parse_structured(self, raw, output_type):
        return output_type.model_validate_json(raw.text)

    async def _parse_logprobs(self, raw, target_tokens):
        return {}

    def _extract_text(self, raw) -> str:
        return raw.text


def create_agent(
    model: LLMModel,
    system_prompt: str,
    user_prompt: str,
    output_type: Optional[Type[BaseModel]] = None,
    reasoning_effort: int = 0,
    max_concurrency: int = 12,
    temperature: float = 0.0,
) -> LLMAgent:
    """Factory: create the right agent subclass based on model vendor."""
    kwargs = dict(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_type=output_type,
        reasoning_effort=reasoning_effort,
        max_concurrency=max_concurrency,
        temperature=temperature,
    )
    if model.vendor == Vendor.ANTHROPIC:
        return AnthropicAgent(**kwargs)
    elif model.vendor == Vendor.OPENAI:
        return OpenAIAgent(**kwargs)
    elif model.vendor == Vendor.GEMINI:
        return GeminiAgent(**kwargs)
    else:
        raise ValueError(f"Unsupported vendor: {model.vendor}")
