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
