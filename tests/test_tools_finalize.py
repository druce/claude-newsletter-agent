"""Tests for tools/finalize.py — finalize newsletter tool (Step 9)."""
import asyncio
import os

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


TEST_DB = "test_finalize.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestAssembleMarkdown:
    def test_assembles_sections(self):
        from tools.finalize import _assemble_markdown

        section_data = [
            {
                "section_title": "AI Takes Stage",
                "headline": "OpenAI launches GPT-5",
                "links": [{"site_name": "Reuters", "url": "https://reuters.com/1"}],
                "category": "AI Policy",
            },
            {
                "section_title": "AI Takes Stage",
                "headline": "Anthropic raises $5B",
                "links": [{"site_name": "FT", "url": "https://ft.com/2"}],
                "category": "AI Policy",
            },
            {
                "section_title": "Chip Wars",
                "headline": "Nvidia unveils H200",
                "links": [{"site_name": "Bloomberg", "url": "https://bloomberg.com/3"}],
                "category": "Hardware",
            },
        ]
        md = _assemble_markdown(section_data)
        assert "## AI Takes Stage" in md
        assert "## Chip Wars" in md
        assert "[Reuters]" in md
        assert "- OpenAI launches GPT-5" in md


class TestFinalizeAction:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Newsletter, AgentState
        AgentState.create_table(TEST_DB)
        Newsletter.create_table(TEST_DB)
        yield

    @patch("tools.finalize.send_gmail")
    @patch("tools.finalize.create_agent")
    def test_finalizes_newsletter(self, mock_create, mock_send):
        from tools.finalize import finalize_action
        from state import NewsletterAgentState
        from tools.models import StringResult, NewsletterCritique
        from db import Newsletter

        # Mock title agent
        mock_title = AsyncMock()
        mock_title.prompt_dict.return_value = StringResult(
            result="AI Reshapes Markets; Nvidia Surges"
        )

        # Mock draft agent
        mock_draft = AsyncMock()
        mock_draft.prompt_dict.return_value = StringResult(
            result="# AI Reshapes Markets\n\n## Section One\n- Story - [Source](url)"
        )

        # Mock critique agent (passes immediately)
        mock_critique = AsyncMock()
        mock_critique.prompt_dict.return_value = NewsletterCritique(
            overall_score=8.5,
            title_quality=8.0,
            structure_quality=8.5,
            section_quality=8.0,
            headline_quality=8.5,
            should_iterate=False,
            critique_text="Good newsletter.",
        )

        mock_create.side_effect = [mock_title, mock_draft, mock_critique]

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.newsletter_section_data = [
            {
                "section_title": "AI Policy",
                "headline": "OpenAI launches GPT-5",
                "links": [{"site_name": "Reuters", "url": "https://reuters.com/1"}],
                "category": "AI Policy",
            },
        ]

        result = asyncio.run(finalize_action(state))

        assert "8.5" in result  # score
        assert state.final_newsletter != ""
        assert state.newsletter_title != ""
        mock_send.assert_called_once()

    def test_handles_empty_sections(self):
        from tools.finalize import finalize_action
        from state import NewsletterAgentState
        from db import AgentState
        AgentState.create_table(TEST_DB)

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.newsletter_section_data = []
        result = asyncio.run(finalize_action(state))
        assert "No" in result
