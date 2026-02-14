"""Tests for tools/summarize.py — extract summaries tool (Step 4)."""
import asyncio
import os

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


TEST_DB = "test_summarize.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestSummarizeAction:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Article, AgentState
        AgentState.create_table(TEST_DB)
        Article.create_table(TEST_DB)
        yield

    @patch("tools.summarize.create_agent")
    def test_summarizes_articles_with_text(self, mock_create, tmp_path):
        from tools.summarize import summarize_action
        from state import NewsletterAgentState
        from db import Article

        # Create article text files
        text_file = tmp_path / "article1.txt"
        text_file.write_text("This is a long article about AI breakthroughs. " * 50)

        # Insert article with text_path
        Article(
            final_url="https://example.com/1",
            url="https://example.com/1",
            title="AI Breakthrough",
            source="TestSource",
            status="success",
            text_path=str(text_file),
            content_length=1000,
            domain="example.com",
        ).insert(TEST_DB)

        # Mock summary agent
        mock_summary_agent = AsyncMock()
        mock_summary_agent.prompt_list.return_value = [
            {"id": 1, "summary": "- Point one\n- Point two\n- Point three"}
        ]

        # Mock distiller agent
        mock_distill_agent = AsyncMock()
        mock_distill_agent.prompt_list.return_value = [
            {"id": 1, "short_summary": "AI breakthrough achieved in new study."}
        ]

        mock_create.side_effect = [mock_summary_agent, mock_distill_agent]

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(summarize_action(state))

        assert "1" in result  # summarized 1 article
        article = Article.get(TEST_DB, 1)
        assert article.summary == "- Point one\n- Point two\n- Point three"
        assert article.short_summary == "AI breakthrough achieved in new study."

    def test_handles_no_articles(self):
        from tools.summarize import summarize_action
        from state import NewsletterAgentState

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(summarize_action(state))
        assert "No" in result

    @patch("tools.summarize.create_agent")
    def test_skips_already_summarized(self, mock_create, tmp_path):
        from tools.summarize import summarize_action
        from state import NewsletterAgentState
        from db import Article

        text_file = tmp_path / "article.txt"
        text_file.write_text("Content here. " * 50)

        # Article already has a summary
        Article(
            final_url="https://example.com/1",
            url="https://example.com/1",
            title="Already Summarized",
            source="TestSource",
            status="success",
            text_path=str(text_file),
            content_length=1000,
            summary="Already has summary",
            short_summary="Already short",
            domain="example.com",
        ).insert(TEST_DB)

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(summarize_action(state))
        assert "No" in result
        mock_create.assert_not_called()
