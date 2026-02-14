"""Tests for tools/select_sections.py — select sections tool (Step 7)."""
import asyncio
import os

import pytest
from unittest.mock import AsyncMock, patch, call


TEST_DB = "test_select.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestSelectSectionsAction:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Article, AgentState
        AgentState.create_table(TEST_DB)
        Article.create_table(TEST_DB)
        yield

    def _insert_articles(self, count=5):
        from db import Article
        for i in range(count):
            Article(
                final_url=f"https://example.com/{i}",
                url=f"https://example.com/{i}",
                title=f"Article {i}",
                source="TestSource",
                status="success",
                content_length=5000,
                rating=float(8 - i),
                summary=f"Summary of article {i}",
                short_summary=f"Short summary {i}",
                cluster_label=str(i % 2),
                domain="example.com",
                site_name="Example",
            ).insert(TEST_DB)

    @patch("tools.select_sections.create_agent")
    def test_proposes_and_assigns_categories(self, mock_create):
        from tools.select_sections import select_sections_action
        from state import NewsletterAgentState

        self._insert_articles(4)

        # Mock agents in order: proposal, cleanup, assignment (x4), dedupe (x2)
        mock_proposal = AsyncMock()
        mock_proposal.prompt_dict.return_value = AsyncMock(
            categories=["AI Policy", "LLM Updates"]
        )

        mock_cleanup = AsyncMock()
        mock_cleanup.prompt_dict.return_value = AsyncMock(
            categories=["AI Policy", "LLM Updates"]
        )

        mock_assign = AsyncMock()
        mock_assign.prompt_dict.side_effect = [
            AsyncMock(topic_title="AI Policy"),
            AsyncMock(topic_title="LLM Updates"),
            AsyncMock(topic_title="AI Policy"),
            AsyncMock(topic_title="LLM Updates"),
        ]

        mock_dedupe1 = AsyncMock()
        mock_dedupe1.prompt_list.return_value = [
            {"id": 1, "dupe_id": -1},
            {"id": 3, "dupe_id": -1},
        ]

        mock_dedupe2 = AsyncMock()
        mock_dedupe2.prompt_list.return_value = [
            {"id": 2, "dupe_id": -1},
            {"id": 4, "dupe_id": -1},
        ]

        mock_create.side_effect = [
            mock_proposal, mock_cleanup, mock_assign, mock_dedupe1, mock_dedupe2,
        ]

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(select_sections_action(state))

        assert "4" in result  # organized 4 articles
        assert "2" in result  # into 2 sections
        assert len(state.newsletter_section_data) > 0

    def test_handles_no_articles(self):
        from tools.select_sections import select_sections_action
        from state import NewsletterAgentState

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(select_sections_action(state))
        assert "No" in result
