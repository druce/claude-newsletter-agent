"""Tests for tools/draft_sections.py — draft sections tool (Step 8)."""
import asyncio
import os

import pytest
from unittest.mock import AsyncMock, patch


TEST_DB = "test_draft.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestStorySelection:
    def test_tier1_must_include(self):
        from tools.draft_sections import _select_stories

        items = [
            {"id": 1, "rating": 9.5, "category": "AI Policy"},
            {"id": 2, "rating": 5.0, "category": "AI Policy"},
            {"id": 3, "rating": 8.5, "category": "LLM Updates"},
        ]
        selected = _select_stories(items, max_stories=100, tier1_threshold=8.0)
        # Tier 1 stories (rating > 8.0) are always included
        tier1_ids = {s["id"] for s in selected if s["rating"] > 8.0}
        assert 1 in tier1_ids
        assert 3 in tier1_ids

    def test_max_stories_cap(self):
        from tools.draft_sections import _select_stories

        items = [{"id": i, "rating": float(i), "category": "A"} for i in range(200)]
        selected = _select_stories(items, max_stories=50, tier1_threshold=8.0)
        assert len(selected) <= 50


class TestDraftSectionsAction:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Article, AgentState
        AgentState.create_table(TEST_DB)
        Article.create_table(TEST_DB)
        yield

    @patch("tools.draft_sections.create_agent")
    def test_drafts_sections(self, mock_create):
        from tools.draft_sections import draft_sections_action
        from state import NewsletterAgentState
        from tools.models import Section, Headline, HeadlineLink

        # Mock write_section agent
        mock_write = AsyncMock()
        mock_write.prompt_dict.return_value = Section(
            section_title="AI Takes Center Stage",
            headlines=[
                Headline(
                    headline="OpenAI launches GPT-5",
                    rating=9.0,
                    prune=False,
                    links=[HeadlineLink(site_name="Reuters", url="https://reuters.com/1")],
                ),
            ],
        )

        # Mock critique agent (returns passing scores)
        mock_critique = AsyncMock()
        from tools.models import SectionCritique, StoryAction
        mock_critique.prompt_dict.return_value = SectionCritique(
            coherence_score=8.5,
            quality_score=8.0,
            actions=[StoryAction(id=1, action="keep")],
        )

        mock_create.side_effect = [mock_write, mock_critique]

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.newsletter_section_data = [
            {
                "id": 1,
                "db_id": 1,
                "category": "AI Policy",
                "title": "OpenAI launches GPT-5",
                "summary": "Summary here",
                "short_summary": "Short summary",
                "rating": 9.0,
                "final_url": "https://reuters.com/1",
                "site_name": "Reuters",
                "source": "Reuters",
            },
        ]

        result = asyncio.run(draft_sections_action(state))

        assert "1" in result  # 1 section
        assert len(state.newsletter_section_data) > 0

    def test_handles_no_section_data(self):
        from tools.draft_sections import draft_sections_action
        from state import NewsletterAgentState
        from db import AgentState
        AgentState.create_table(TEST_DB)

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.newsletter_section_data = []
        result = asyncio.run(draft_sections_action(state))
        assert "No" in result
