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
