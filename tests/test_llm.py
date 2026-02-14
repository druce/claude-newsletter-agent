# tests/test_llm.py
"""Tests for llm.py LLM wrapper."""
import asyncio
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
    )


class TestPromptDict:
    def test_simple_string_response(self):
        agent = _make_stub_agent(responses=["hello world"])
        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert result == "hello world"

    def test_structured_output(self):
        from pydantic import BaseModel

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
        from pydantic import BaseModel

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

    @patch("llm.openai.AsyncOpenAI")
    def test_instantiation(self, mock_cls):
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
        from pydantic import BaseModel

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

    @patch("llm.genai.Client")
    def test_instantiation(self, mock_cls):
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

        result = asyncio.run(agent.prompt_dict({"text": "test"}))
        assert result == "hello world"

    @patch("llm.genai.Client")
    def test_call_llm_structured_output(self, mock_cls):
        from pydantic import BaseModel

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
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_budget == 8192


import json
import pandas as pd


def _make_batch_stub_agent(responses, output_type=None):
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
        from pydantic import BaseModel

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
        from pydantic import BaseModel

        class ResultList(BaseModel):
            results_list: list

        items = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
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
        from pydantic import BaseModel

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
        from pydantic import BaseModel

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


class TestRunPromptWithProbs:
    def test_native_logprobs_openai(self):
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
        from llm import LLMAgent, CLAUDE_SONNET_MODEL

        class StubClaude(LLMAgent):
            _fatal_exceptions = ()
            _temporary_exceptions = ()
            async def _call_llm(self, system, user, output_schema):
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
