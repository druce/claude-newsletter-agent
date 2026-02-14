# Phase 2: LLM Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `llm.py` — a multi-vendor async LLM wrapper supporting Anthropic, OpenAI, and Gemini with structured output, batch DataFrame processing, retry logic, and unified reasoning-effort control.

**Architecture:** Abstract base class `LLMAgent` with vendor-specific subclasses (`AnthropicAgent`, `OpenAIAgent`, `GeminiAgent`). All API calls are async. Structured output uses each vendor's native approach (tool_use, json_schema, response_schema). A `LLMModel` dataclass describes model capabilities. Factory function `create_agent()` routes model → subclass.

**Tech Stack:** anthropic, openai, google-genai, tenacity, pydantic, pandas, asyncio

**Design doc:** `docs/plans/2026-02-13-llm-layer-design.md`

---

### Task 1: Vendor Enum + LLMModel Dataclass + Model Constants

**Files:**
- Create: `llm.py`
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

```python
# tests/test_llm.py
"""Tests for llm.py LLM wrapper."""
import pytest


class TestVendor:
    def test_vendor_values(self):
        from llm import Vendor
        assert Vendor.ANTHROPIC.value == "anthropic"
        assert Vendor.OPENAI.value == "openai"
        assert Vendor.GEMINI.value == "gemini"


class TestLLMModel:
    def test_model_fields(self):
        from llm import LLMModel, Vendor
        m = LLMModel(
            model_id="test-model",
            vendor=Vendor.ANTHROPIC,
            supports_logprobs=False,
            supports_reasoning=True,
            default_max_tokens=4096,
            display_name="Test Model",
        )
        assert m.model_id == "test-model"
        assert m.vendor == Vendor.ANTHROPIC
        assert m.supports_logprobs is False
        assert m.supports_reasoning is True

    def test_model_is_frozen(self):
        from llm import LLMModel, Vendor
        m = LLMModel(
            model_id="test", vendor=Vendor.OPENAI,
            supports_logprobs=True, supports_reasoning=False,
            default_max_tokens=4096, display_name="Test",
        )
        with pytest.raises(AttributeError):
            m.model_id = "changed"


class TestModelConstants:
    def test_claude_sonnet_model(self):
        from llm import CLAUDE_SONNET_MODEL, Vendor
        assert CLAUDE_SONNET_MODEL.model_id == "claude-sonnet-4-5-20250929"
        assert CLAUDE_SONNET_MODEL.vendor == Vendor.ANTHROPIC
        assert CLAUDE_SONNET_MODEL.supports_logprobs is False
        assert CLAUDE_SONNET_MODEL.supports_reasoning is True

    def test_claude_haiku_model(self):
        from llm import CLAUDE_HAIKU_MODEL, Vendor
        assert CLAUDE_HAIKU_MODEL.model_id == "claude-haiku-4-5-20251001"
        assert CLAUDE_HAIKU_MODEL.vendor == Vendor.ANTHROPIC

    def test_gpt5_mini_model(self):
        from llm import GPT5_MINI_MODEL, Vendor
        assert GPT5_MINI_MODEL.vendor == Vendor.OPENAI
        assert GPT5_MINI_MODEL.supports_logprobs is True

    def test_gpt41_mini_model(self):
        from llm import GPT41_MINI_MODEL, Vendor
        assert GPT41_MINI_MODEL.vendor == Vendor.OPENAI
        assert GPT41_MINI_MODEL.supports_reasoning is False

    def test_gemini_flash_model(self):
        from llm import GEMINI_FLASH_MODEL, Vendor
        assert GEMINI_FLASH_MODEL.vendor == Vendor.GEMINI
        assert GEMINI_FLASH_MODEL.supports_logprobs is False

    def test_gemini_pro_model(self):
        from llm import GEMINI_PRO_MODEL, Vendor
        assert GEMINI_PRO_MODEL.vendor == Vendor.GEMINI
        assert GEMINI_PRO_MODEL.supports_reasoning is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py::TestVendor -v && pytest tests/test_llm.py::TestLLMModel -v && pytest tests/test_llm.py::TestModelConstants -v`
Expected: FAIL with ImportError (llm module doesn't exist yet)

**Step 3: Write minimal implementation**

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v -k "TestVendor or TestLLMModel or TestModelConstants"`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat(llm): add Vendor enum, LLMModel dataclass, and model constants"
```

---

### Task 2: LLMAgent Abstract Base Class + Reasoning Effort Mapping

**Files:**
- Modify: `llm.py`
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
class TestReasoningEffort:
    def test_anthropic_thinking_tokens(self):
        from llm import _anthropic_thinking_tokens
        assert _anthropic_thinking_tokens(0) == 0
        assert _anthropic_thinking_tokens(2) == 1000
        assert _anthropic_thinking_tokens(4) == 2000
        assert _anthropic_thinking_tokens(6) == 4000
        assert _anthropic_thinking_tokens(8) == 8000
        assert _anthropic_thinking_tokens(10) == 16000

    def test_openai_reasoning_effort(self):
        from llm import _openai_reasoning_effort
        assert _openai_reasoning_effort(0) is None
        assert _openai_reasoning_effort(2) == "low"
        assert _openai_reasoning_effort(4) == "low"
        assert _openai_reasoning_effort(6) == "medium"
        assert _openai_reasoning_effort(8) == "high"
        assert _openai_reasoning_effort(10) == "high"

    def test_gemini_thinking_tokens(self):
        from llm import _gemini_thinking_tokens
        assert _gemini_thinking_tokens(0) == 0
        assert _gemini_thinking_tokens(2) == 1024
        assert _gemini_thinking_tokens(6) == 4096
        assert _gemini_thinking_tokens(10) == 16384

    def test_invalid_reasoning_effort(self):
        from llm import _anthropic_thinking_tokens
        with pytest.raises(ValueError):
            _anthropic_thinking_tokens(-1)
        with pytest.raises(ValueError):
            _anthropic_thinking_tokens(11)


class TestLLMAgentInit:
    def test_cannot_instantiate_abc(self):
        from llm import LLMAgent, CLAUDE_SONNET_MODEL
        with pytest.raises(TypeError):
            LLMAgent(
                model=CLAUDE_SONNET_MODEL,
                system_prompt="You are helpful.",
                user_prompt="Hello {name}",
            )

    def test_prompt_template_stored(self):
        """Verify subclass can be instantiated and stores prompts."""
        from llm import LLMAgent, CLAUDE_SONNET_MODEL, LLMModel, Vendor
        # Create a minimal concrete subclass for testing
        class StubAgent(LLMAgent):
            _fatal_exceptions = ()
            _temporary_exceptions = ()
            async def _call_llm(self, system, user, output_schema):
                return ""
            async def _parse_structured(self, raw, output_type):
                return None
            async def _parse_logprobs(self, raw, target_tokens):
                return {}

        agent = StubAgent(
            model=CLAUDE_SONNET_MODEL,
            system_prompt="sys",
            user_prompt="Hello {name}",
        )
        assert agent.system_prompt == "sys"
        assert agent.user_prompt == "Hello {name}"
        assert agent.model == CLAUDE_SONNET_MODEL
        assert agent.reasoning_effort == 0
        assert agent.max_concurrency == 12
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v -k "TestReasoningEffort or TestLLMAgentInit"`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `llm.py` after model constants:

```python
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v -k "TestReasoningEffort or TestLLMAgentInit"`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat(llm): add LLMAgent ABC and reasoning effort mapping"
```

---

### Task 3: Retry Logic + prompt_dict()

**Files:**
- Modify: `llm.py`
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
import asyncio


def _make_stub_agent(responses=None, output_type=None, side_effect=None):
    """Helper: create a StubAgent that returns canned responses or raises."""
    from llm import LLMAgent, CLAUDE_SONNET_MODEL

    class StubAgent(LLMAgent):
        _fatal_exceptions = (ValueError,)
        _temporary_exceptions = (ConnectionError,)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.call_count = 0
            self._responses = responses or ["stub response"]
            self._side_effect = side_effect

        async def _call_llm(self, system, user, output_schema):
            self.call_count += 1
            if self._side_effect and self.call_count <= len(self._side_effect):
                exc = self._side_effect[self.call_count - 1]
                if exc is not None:
                    raise exc
            idx = min(self.call_count - 1, len(self._responses) - 1)
            return self._responses[idx]

        async def _parse_structured(self, raw, output_type):
            return output_type.model_validate_json(raw)

        async def _parse_logprobs(self, raw, target_tokens):
            return {}

    return StubAgent(
        model=CLAUDE_SONNET_MODEL,
        system_prompt="You are helpful.",
        user_prompt="Process: {text}",
        output_type=output_type,
        **kwargs if 'kwargs' in dir() else {},
    )


class TestPromptDict:
    def test_simple_string_response(self):
        agent = _make_stub_agent(responses=["hello world"])
        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert result == "hello world"

    def test_structured_output(self):
        class Output(BaseModel):
            answer: str

        agent = _make_stub_agent(
            responses=['{"answer": "42"}'],
            output_type=Output,
        )
        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert isinstance(result, Output)
        assert result.answer == "42"

    def test_variable_substitution(self):
        agent = _make_stub_agent(responses=["ok"])
        # Verify the user prompt gets variables substituted
        asyncio.run(agent.prompt_dict({"text": "hello"}))
        assert agent.call_count == 1


class TestRetryLogic:
    def test_retries_on_temporary_exception(self):
        agent = _make_stub_agent(
            responses=["ok"],
            side_effect=[ConnectionError("timeout"), None],
        )
        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert result == "ok"
        assert agent.call_count == 2

    def test_no_retry_on_fatal_exception(self):
        agent = _make_stub_agent(
            responses=["ok"],
            side_effect=[ValueError("bad request")],
        )
        with pytest.raises(ValueError, match="bad request"):
            asyncio.run(agent.prompt_dict({"text": "test"}))
        assert agent.call_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v -k "TestPromptDict or TestRetryLogic"`
Expected: FAIL (prompt_dict not defined)

**Step 3: Write minimal implementation**

Add to `LLMAgent` class in `llm.py`:

```python
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    async def prompt_dict(self, variables: Optional[Dict[str, Any]] = None) -> Any:
        """Single prompt with variable substitution. Returns structured output or raw string."""
        user_text = self.user_prompt.format(**(variables or {}))
        output_schema = None
        if self.output_type:
            output_schema = self.output_type.model_json_schema()

        raw = await self._call_with_retry(self.system_prompt, user_text, output_schema)

        if self.output_type:
            return await self._parse_structured(raw, self.output_type)
        return raw

    async def _call_with_retry(self, system: str, user: str, output_schema: Optional[Dict[str, Any]]) -> Any:
        """Call LLM with tenacity retry on temporary exceptions."""
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

        @retry(
            retry=retry_if_exception_type(self._temporary_exceptions),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=60),
            reraise=True,
        )
        async def _do_call():
            return await self._call_llm(system, user, output_schema)

        return await _do_call()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v -k "TestPromptDict or TestRetryLogic"`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat(llm): add prompt_dict with retry logic"
```

---

### Task 4: AnthropicAgent Implementation

**Files:**
- Modify: `llm.py`
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch


class TestAnthropicAgent:
    def _make_agent(self, output_type=None, reasoning_effort=0):
        from llm import AnthropicAgent, CLAUDE_SONNET_MODEL
        return AnthropicAgent(
            model=CLAUDE_SONNET_MODEL,
            system_prompt="You are helpful.",
            user_prompt="Process: {text}",
            output_type=output_type,
            reasoning_effort=reasoning_effort,
        )

    def test_instantiation(self):
        agent = self._make_agent()
        assert agent.model.vendor.value == "anthropic"

    @patch("llm.anthropic.AsyncAnthropic")
    def test_call_llm_text_response(self, mock_cls):
        from llm import AnthropicAgent
        agent = self._make_agent()

        # Mock the response
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "hello world"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert result == "hello world"
        mock_client.messages.create.assert_called_once()

    @patch("llm.anthropic.AsyncAnthropic")
    def test_call_llm_structured_output(self, mock_cls):
        class Output(BaseModel):
            answer: str

        agent = self._make_agent(output_type=Output)

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.input = {"answer": "42"}
        mock_response = MagicMock()
        mock_response.content = [mock_tool_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert isinstance(result, Output)
        assert result.answer == "42"

    @patch("llm.anthropic.AsyncAnthropic")
    def test_reasoning_effort_adds_thinking(self, mock_cls):
        agent = self._make_agent(reasoning_effort=6)

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "result"
        mock_response = MagicMock()
        mock_response.content = [mock_text_block]

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        asyncio.run(agent.prompt_dict({"text": "test"}))
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["budget_tokens"] == 4000
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py::TestAnthropicAgent -v`
Expected: FAIL with ImportError (AnthropicAgent not defined)

**Step 3: Write minimal implementation**

Add to `llm.py`:

```python
import anthropic


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
        if output_schema:
            kwargs["tools"] = [{
                "name": "structured_output",
                "description": "Return structured data",
                "input_schema": output_schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

        return await self._client.messages.create(**kwargs)

    async def _parse_structured(self, raw, output_type):
        for block in raw.content:
            if block.type == "tool_use":
                return output_type.model_validate(block.input)
        raise ValueError("No tool_use block in Anthropic response")

    async def _parse_logprobs(self, raw, target_tokens):
        # Anthropic doesn't support native logprobs; confidence is in structured output
        return {}

    def _extract_text(self, raw) -> str:
        """Extract text from Anthropic response, skipping thinking blocks."""
        for block in raw.content:
            if block.type == "text":
                return block.text
        return ""
```

Also update `prompt_dict` in base class to handle vendor-specific text extraction:

```python
    async def prompt_dict(self, variables=None):
        user_text = self.user_prompt.format(**(variables or {}))
        output_schema = None
        if self.output_type:
            output_schema = self.output_type.model_json_schema()

        raw = await self._call_with_retry(self.system_prompt, user_text, output_schema)

        if self.output_type:
            return await self._parse_structured(raw, self.output_type)
        # For text responses, subclass provides _extract_text; fallback to str
        if hasattr(self, '_extract_text'):
            return self._extract_text(raw)
        return str(raw)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py::TestAnthropicAgent -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat(llm): add AnthropicAgent with tool_use structured output"
```

---

### Task 5: OpenAIAgent Implementation

**Files:**
- Modify: `llm.py`
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
class TestOpenAIAgent:
    def _make_agent(self, output_type=None, reasoning_effort=0):
        from llm import OpenAIAgent, GPT41_MINI_MODEL
        return OpenAIAgent(
            model=GPT41_MINI_MODEL,
            system_prompt="You are helpful.",
            user_prompt="Process: {text}",
            output_type=output_type,
            reasoning_effort=reasoning_effort,
        )

    def test_instantiation(self):
        agent = self._make_agent()
        assert agent.model.vendor.value == "openai"

    @patch("llm.openai.AsyncOpenAI")
    def test_call_llm_text_response(self, mock_cls):
        agent = self._make_agent()

        mock_message = MagicMock()
        mock_message.content = "hello world"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.logprobs = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert result == "hello world"

    @patch("llm.openai.AsyncOpenAI")
    def test_call_llm_structured_output(self, mock_cls):
        class Output(BaseModel):
            answer: str

        agent = self._make_agent(output_type=Output)

        mock_message = MagicMock()
        mock_message.content = '{"answer": "42"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.logprobs = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert isinstance(result, Output)
        assert result.answer == "42"

    @patch("llm.openai.AsyncOpenAI")
    def test_logprobs_extraction(self, mock_cls):
        agent = self._make_agent()

        # Mock logprobs structure
        mock_token_logprob = MagicMock()
        mock_token_logprob.token = "1"
        mock_token_logprob.logprob = -0.1  # ~0.905

        mock_content_item = MagicMock()
        mock_content_item.token = "1"
        mock_content_item.logprob = -0.1
        mock_content_item.top_logprobs = [mock_token_logprob]

        mock_logprobs = MagicMock()
        mock_logprobs.content = [mock_content_item]

        mock_message = MagicMock()
        mock_message.content = "1"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.logprobs = mock_logprobs
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        agent._client = mock_client

        import math
        probs = asyncio.run(agent._parse_logprobs(mock_response, ["1"]))
        assert "1" in probs
        assert abs(probs["1"] - math.exp(-0.1)) < 0.01
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py::TestOpenAIAgent -v`
Expected: FAIL (OpenAIAgent not defined)

**Step 3: Write minimal implementation**

Add to `llm.py`:

```python
import math
import openai


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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py::TestOpenAIAgent -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat(llm): add OpenAIAgent with json_schema output and native logprobs"
```

---

### Task 6: GeminiAgent Implementation

**Files:**
- Modify: `llm.py`
- Modify: `requirements.txt` (add google-genai)
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
class TestGeminiAgent:
    def _make_agent(self, output_type=None, reasoning_effort=0):
        from llm import GeminiAgent, GEMINI_FLASH_MODEL
        return GeminiAgent(
            model=GEMINI_FLASH_MODEL,
            system_prompt="You are helpful.",
            user_prompt="Process: {text}",
            output_type=output_type,
            reasoning_effort=reasoning_effort,
        )

    def test_instantiation(self):
        agent = self._make_agent()
        assert agent.model.vendor.value == "gemini"

    @patch("llm.genai.Client")
    def test_call_llm_text_response(self, mock_cls):
        agent = self._make_agent()

        mock_response = MagicMock()
        mock_response.text = "hello world"

        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(return_value=mock_response)
        agent._client = mock_client

        # GeminiAgent uses sync client in async wrapper
        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert result == "hello world"

    @patch("llm.genai.Client")
    def test_call_llm_structured_output(self, mock_cls):
        class Output(BaseModel):
            answer: str

        agent = self._make_agent(output_type=Output)

        mock_response = MagicMock()
        mock_response.text = '{"answer": "42"}'

        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(return_value=mock_response)
        agent._client = mock_client

        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert isinstance(result, Output)
        assert result.answer == "42"

    @patch("llm.genai.Client")
    def test_reasoning_effort_adds_thinking(self, mock_cls):
        agent = self._make_agent(reasoning_effort=8)

        mock_response = MagicMock()
        mock_response.text = "result"

        mock_client = MagicMock()
        mock_client.models.generate_content = MagicMock(return_value=mock_response)
        agent._client = mock_client

        asyncio.run(agent.prompt_dict({"text": "test"}))
        call_kwargs = mock_client.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        # Verify thinking_config was set
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 8192
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py::TestGeminiAgent -v`
Expected: FAIL (GeminiAgent not defined)

**Step 3: Write minimal implementation**

Add to `llm.py`:

```python
from google import genai
from google.genai import types as genai_types


class GeminiAgent(LLMAgent):
    """Google Gemini agent using response_schema for structured output."""

    _fatal_exceptions = (
        # google.api_core.exceptions mapped by genai
    )
    _temporary_exceptions = (
        # google.api_core.exceptions mapped by genai
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = genai.Client()

    async def _call_llm(self, system, user, output_schema):
        config_kwargs = {
            "temperature": self.temperature,
            "system_instruction": system,
        }

        # Reasoning effort → thinking budget
        thinking_tokens = _gemini_thinking_tokens(self.reasoning_effort)
        if thinking_tokens > 0:
            config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_tokens
            )

        # Structured output via response_schema
        if output_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = output_schema

        config = genai_types.GenerateContentConfig(**config_kwargs)

        # genai Client is sync; run in executor for async compatibility
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
        # Gemini doesn't support native logprobs; confidence is in structured output
        return {}

    def _extract_text(self, raw) -> str:
        return raw.text
```

Also add `google-genai` to `requirements.txt`.

**Step 4: Run tests to verify they pass**

Run: `pip install google-genai && pytest tests/test_llm.py::TestGeminiAgent -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py requirements.txt
git commit -m "feat(llm): add GeminiAgent with response_schema structured output"
```

---

### Task 7: Batch Processing — prompt_list() and filter_dataframe()

**Files:**
- Modify: `llm.py`
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
import pandas as pd


def _make_batch_stub_agent(responses, output_type=None):
    """Create a StubAgent for batch tests with multiple canned responses."""
    from llm import LLMAgent, CLAUDE_SONNET_MODEL

    class BatchStubAgent(LLMAgent):
        _fatal_exceptions = ()
        _temporary_exceptions = ()

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.call_count = 0
            self._responses = responses
            self.captured_user_prompts = []

        async def _call_llm(self, system, user, output_schema):
            self.call_count += 1
            self.captured_user_prompts.append(user)
            idx = min(self.call_count - 1, len(self._responses) - 1)
            return self._responses[idx]

        async def _parse_structured(self, raw, output_type):
            return output_type.model_validate_json(raw)

        async def _parse_logprobs(self, raw, target_tokens):
            return {}

        def _extract_text(self, raw):
            return str(raw)

    return BatchStubAgent(
        model=CLAUDE_SONNET_MODEL,
        system_prompt="Classify items.",
        user_prompt="Items: {items_json}",
        output_type=output_type,
    )


class TestPromptList:
    def test_processes_list_of_items(self):
        class ItemResult(BaseModel):
            id: int
            output: str

        class ResultList(BaseModel):
            results_list: list

        items = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
        response_json = json.dumps({
            "results_list": [
                {"id": 1, "output": "yes"},
                {"id": 2, "output": "no"},
            ]
        })
        agent = _make_batch_stub_agent(
            responses=[response_json],
            output_type=ResultList,
        )
        results = asyncio.run(agent.prompt_list(items))
        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["output"] == "yes"

    def test_validates_returned_ids(self):
        class ResultList(BaseModel):
            results_list: list

        items = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
        # Response has wrong ID
        response_json = json.dumps({
            "results_list": [
                {"id": 1, "output": "yes"},
                {"id": 99, "output": "no"},
            ]
        })
        agent = _make_batch_stub_agent(
            responses=[response_json],
            output_type=ResultList,
        )
        with pytest.raises(ValueError, match="ID mismatch"):
            asyncio.run(agent.prompt_list(items, item_id_field="id"))


class TestFilterDataframe:
    def test_single_chunk(self):
        class ResultList(BaseModel):
            results_list: list

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "text": ["a", "b", "c"],
        })
        response_json = json.dumps({
            "results_list": [
                {"id": 1, "output": "x"},
                {"id": 2, "output": "y"},
                {"id": 3, "output": "z"},
            ]
        })
        agent = _make_batch_stub_agent(
            responses=[response_json],
            output_type=ResultList,
        )
        series = asyncio.run(agent.filter_dataframe(df, chunk_size=25))
        assert list(series) == ["x", "y", "z"]
        assert agent.call_count == 1

    def test_multiple_chunks(self):
        class ResultList(BaseModel):
            results_list: list

        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "text": ["a", "b", "c", "d"],
        })
        resp1 = json.dumps({"results_list": [{"id": 1, "output": "x"}, {"id": 2, "output": "y"}]})
        resp2 = json.dumps({"results_list": [{"id": 3, "output": "z"}, {"id": 4, "output": "w"}]})
        agent = _make_batch_stub_agent(
            responses=[resp1, resp2],
            output_type=ResultList,
        )
        series = asyncio.run(agent.filter_dataframe(df, chunk_size=2))
        assert list(series) == ["x", "y", "z", "w"]
        assert agent.call_count == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v -k "TestPromptList or TestFilterDataframe"`
Expected: FAIL (prompt_list and filter_dataframe not defined)

**Step 3: Write minimal implementation**

Add to `LLMAgent` class in `llm.py`:

```python
    async def prompt_list(
        self,
        items: List[Dict[str, Any]],
        item_id_field: str = "id",
        item_list_field: str = "results_list",
    ) -> List[Dict[str, Any]]:
        """Process a list of items through the LLM. Returns list of result dicts."""
        items_json = json.dumps(items)
        user_text = self.user_prompt.format(items_json=items_json)
        output_schema = None
        if self.output_type:
            output_schema = self.output_type.model_json_schema()

        raw = await self._call_with_retry(self.system_prompt, user_text, output_schema)

        if self.output_type:
            parsed = await self._parse_structured(raw, self.output_type)
            results = getattr(parsed, item_list_field)
        else:
            results = json.loads(self._extract_text(raw) if hasattr(self, '_extract_text') else str(raw))
            results = results[item_list_field]

        # Validate IDs if present
        sent_ids = [item[item_id_field] for item in items if item_id_field in item]
        if sent_ids:
            returned_ids = [r[item_id_field] for r in results if item_id_field in r]
            if sent_ids != returned_ids:
                raise ValueError(
                    f"ID mismatch: sent {sent_ids}, got {returned_ids}"
                )

        return results

    async def filter_dataframe(
        self,
        df: "pd.DataFrame",
        chunk_size: int = 25,
        value_field: str = "output",
        item_list_field: str = "results_list",
        item_id_field: str = "id",
    ) -> "pd.Series":
        """Process DataFrame in chunks, return Series aligned to original index."""
        import pandas as pd

        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v -k "TestPromptList or TestFilterDataframe"`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat(llm): add prompt_list and filter_dataframe batch processing"
```

---

### Task 8: Logprob / Confidence Score Support — run_prompt_with_probs()

**Files:**
- Modify: `llm.py`
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
class TestRunPromptWithProbs:
    def test_native_logprobs_openai(self):
        """OpenAI returns native logprobs via _parse_logprobs."""
        from llm import LLMAgent, GPT41_MINI_MODEL

        class StubOpenAI(LLMAgent):
            _fatal_exceptions = ()
            _temporary_exceptions = ()
            async def _call_llm(self, system, user, output_schema):
                return "mock_response"
            async def _parse_structured(self, raw, output_type):
                return None
            async def _parse_logprobs(self, raw, target_tokens):
                return {"1": 0.9, "0": 0.1}
            def _extract_text(self, raw):
                return raw

        agent = StubOpenAI(
            model=GPT41_MINI_MODEL,
            system_prompt="sys",
            user_prompt="Is this spam? {text}",
        )
        probs = asyncio.run(agent.run_prompt_with_probs({"text": "buy now"}, target_tokens=["1"]))
        assert "1" in probs
        assert abs(probs["1"] - 0.9) < 0.01

    def test_simulated_confidence_when_no_logprobs(self):
        """Non-logprob models return confidence via structured output."""
        from llm import LLMAgent, CLAUDE_SONNET_MODEL

        class StubClaude(LLMAgent):
            _fatal_exceptions = ()
            _temporary_exceptions = ()
            async def _call_llm(self, system, user, output_schema):
                # The schema should have had 'confidence' injected
                return '{"confidence": 0.85}'
            async def _parse_structured(self, raw, output_type):
                return output_type.model_validate_json(raw)
            async def _parse_logprobs(self, raw, target_tokens):
                return {}
            def _extract_text(self, raw):
                return raw

        agent = StubClaude(
            model=CLAUDE_SONNET_MODEL,
            system_prompt="sys",
            user_prompt="Is this spam? {text}",
        )
        probs = asyncio.run(agent.run_prompt_with_probs({"text": "buy now"}, target_tokens=["1"]))
        assert "1" in probs
        assert abs(probs["1"] - 0.85) < 0.01
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py::TestRunPromptWithProbs -v`
Expected: FAIL (run_prompt_with_probs not defined)

**Step 3: Write minimal implementation**

Add to `LLMAgent` class in `llm.py`:

```python
    async def run_prompt_with_probs(
        self,
        variables: Optional[Dict[str, Any]] = None,
        target_tokens: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Get token probabilities. Native logprobs for OpenAI, simulated confidence otherwise."""
        target_tokens = target_tokens or []
        user_text = self.user_prompt.format(**(variables or {}))

        if self.model.supports_logprobs:
            # Native logprobs path
            raw = await self._call_with_retry(self.system_prompt, user_text, None)
            return await self._parse_logprobs(raw, target_tokens)
        else:
            # Simulated confidence path: create a temporary schema with confidence field
            from pydantic import create_model
            ConfidenceModel = create_model("ConfidenceModel", confidence=(float, ...))
            schema = ConfidenceModel.model_json_schema()

            raw = await self._call_with_retry(self.system_prompt, user_text, schema)
            parsed = await self._parse_structured(raw, ConfidenceModel)
            confidence = parsed.confidence

            # Map confidence to the first target token
            result = {}
            if target_tokens:
                result[target_tokens[0]] = confidence
            return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py::TestRunPromptWithProbs -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py
git commit -m "feat(llm): add run_prompt_with_probs with native and simulated logprobs"
```

---

### Task 9: Factory Function + Update Dependencies + CLAUDE.md

**Files:**
- Modify: `llm.py`
- Modify: `requirements.txt`
- Modify: `CLAUDE.md`
- Test: `tests/test_llm.py`

**Step 1: Write the failing tests**

Append to `tests/test_llm.py`:

```python
class TestCreateAgent:
    def test_creates_anthropic_agent(self):
        from llm import create_agent, CLAUDE_SONNET_MODEL, AnthropicAgent
        agent = create_agent(
            model=CLAUDE_SONNET_MODEL,
            system_prompt="sys",
            user_prompt="hello {name}",
        )
        assert isinstance(agent, AnthropicAgent)

    def test_creates_openai_agent(self):
        from llm import create_agent, GPT41_MINI_MODEL, OpenAIAgent
        agent = create_agent(
            model=GPT41_MINI_MODEL,
            system_prompt="sys",
            user_prompt="hello {name}",
        )
        assert isinstance(agent, OpenAIAgent)

    def test_creates_gemini_agent(self):
        from llm import create_agent, GEMINI_FLASH_MODEL, GeminiAgent
        agent = create_agent(
            model=GEMINI_FLASH_MODEL,
            system_prompt="sys",
            user_prompt="hello {name}",
        )
        assert isinstance(agent, GeminiAgent)

    def test_unknown_vendor_raises(self):
        from llm import create_agent, LLMModel, Vendor
        fake_model = LLMModel(
            model_id="fake", vendor="unknown",
            supports_logprobs=False, supports_reasoning=False,
            default_max_tokens=100, display_name="Fake",
        )
        with pytest.raises(ValueError, match="Unsupported vendor"):
            create_agent(model=fake_model, system_prompt="s", user_prompt="u")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py::TestCreateAgent -v`
Expected: FAIL (create_agent not defined)

**Step 3: Write minimal implementation**

Add to `llm.py`:

```python
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
```

Update `requirements.txt`: add `google-genai` under LLM section.

Update `CLAUDE.md`:
- Change Phase 2 status from "not started" to "COMPLETE"
- Add `llm.py` to Key Files
- Add LLMAgent usage notes

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: ALL PASS (all test classes)

**Step 5: Commit**

```bash
git add llm.py tests/test_llm.py requirements.txt CLAUDE.md
git commit -m "feat(llm): add create_agent factory, update deps and docs"
```

---

### Task 10: Final Integration Verification

**Files:**
- No new files

**Step 1: Run all project tests**

Run: `pytest tests/ -v`
Expected: All tests pass (test_state.py, test_db.py, test_llm.py)

**Step 2: Verify imports work end-to-end**

```bash
python -c "
from llm import (
    Vendor, LLMModel, LLMAgent,
    AnthropicAgent, OpenAIAgent, GeminiAgent,
    create_agent,
    CLAUDE_SONNET_MODEL, CLAUDE_HAIKU_MODEL,
    GPT5_MINI_MODEL, GPT41_MINI_MODEL,
    GEMINI_FLASH_MODEL, GEMINI_PRO_MODEL,
)
print('All imports OK')
agent = create_agent(CLAUDE_SONNET_MODEL, 'sys', 'hello {name}')
print(f'Created: {type(agent).__name__} using {agent.model.display_name}')
"
```
Expected: "All imports OK" and "Created: AnthropicAgent using Claude Sonnet 4.5"

**Step 3: Commit final state if any cleanup needed**

```bash
git status  # Should be clean
```
