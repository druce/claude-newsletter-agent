# tests/test_steps.py
"""Tests for steps/ — step runner utility and CLI scripts."""
import asyncio
import json
import os

import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch


TEST_DB = "test_steps.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestRunStep:
    def test_creates_fresh_state_and_checkpoints(self):
        from db import AgentState
        from state import NewsletterAgentState
        from steps import run_step

        AgentState.create_table(TEST_DB)

        def action(state: NewsletterAgentState) -> str:
            state.headline_data.append({"url": "https://test.com", "title": "Test"})
            return "Processed 1 item"

        result = run_step(
            step_name="gather_urls",
            session_id="test_session",
            db_path=TEST_DB,
            action=action,
        )
        assert result["status"] == "success"
        assert result["message"] == "Processed 1 item"

        # Verify checkpoint was saved
        loaded = NewsletterAgentState.load_latest_from_db("test_session", db_path=TEST_DB)
        assert loaded is not None
        assert len(loaded.headline_data) == 1
        assert loaded.is_step_complete("gather_urls")

    def test_loads_existing_state(self):
        from db import AgentState
        from state import NewsletterAgentState
        from steps import run_step

        AgentState.create_table(TEST_DB)

        # Create initial state with some data
        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data.append({"url": "https://existing.com", "title": "Existing"})
        state.complete_step("gather_urls", message="Initial gather")
        state.save_checkpoint("gather_urls")

        def action(state: NewsletterAgentState) -> str:
            # Should see the existing data
            assert len(state.headline_data) == 1
            return "Filtered"

        result = run_step(
            step_name="filter_urls",
            session_id="test_session",
            db_path=TEST_DB,
            action=action,
        )
        assert result["status"] == "success"

    def test_handles_action_error(self):
        from db import AgentState
        from state import NewsletterAgentState
        from steps import run_step

        AgentState.create_table(TEST_DB)

        def bad_action(state: NewsletterAgentState) -> str:
            raise ValueError("Something went wrong")

        result = run_step(
            step_name="gather_urls",
            session_id="test_session",
            db_path=TEST_DB,
            action=bad_action,
        )
        assert result["status"] == "error"
        assert "Something went wrong" in result["error"]

        # Step should be marked as error, not complete
        loaded = NewsletterAgentState.load_latest_from_db("test_session", db_path=TEST_DB)
        assert loaded is not None
        assert not loaded.is_step_complete("gather_urls")

    def test_async_action(self):
        from db import AgentState
        from state import NewsletterAgentState
        from steps import run_step

        AgentState.create_table(TEST_DB)

        async def async_action(state: NewsletterAgentState) -> str:
            return "Async done"

        result = run_step(
            step_name="gather_urls",
            session_id="test_session",
            db_path=TEST_DB,
            action=async_action,
        )
        assert result["status"] == "success"
        assert result["message"] == "Async done"


class TestGatherUrls:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Url, AgentState
        AgentState.create_table(TEST_DB)
        Url.create_table(TEST_DB)
        yield

    @patch("steps.gather_urls.Fetcher")
    def test_gathers_and_persists_urls(self, mock_fetcher_cls):
        from steps.gather_urls import gather_urls_action
        from state import NewsletterAgentState
        from db import Url

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "source": "TestSource",
                "results": [
                    {"title": "AI Article One", "url": "https://example.com/1", "source": "TestSource"},
                    {"title": "AI Article Two", "url": "https://example.com/2", "source": "TestSource"},
                ],
                "status": "success",
                "metadata": {"entry_count": 2},
            },
        ]
        mock_fetcher.__aenter__ = AsyncMock(return_value=mock_fetcher)
        mock_fetcher.__aexit__ = AsyncMock(return_value=False)
        mock_fetcher_cls.return_value = mock_fetcher

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(gather_urls_action(state))

        assert "2" in result  # message mentions count
        assert len(state.headline_data) == 2
        all_urls = Url.get_all(TEST_DB)
        assert len(all_urls) == 2

    @patch("steps.gather_urls.Fetcher")
    def test_deduplicates_across_sources(self, mock_fetcher_cls):
        from steps.gather_urls import gather_urls_action
        from state import NewsletterAgentState

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "source": "Source1",
                "results": [{"title": "Shared Article", "url": "https://shared.com/1", "source": "Source1"}],
                "status": "success",
                "metadata": {},
            },
            {
                "source": "Source2",
                "results": [{"title": "Shared Article", "url": "https://shared.com/1", "source": "Source2"}],
                "status": "success",
                "metadata": {},
            },
        ]
        mock_fetcher.__aenter__ = AsyncMock(return_value=mock_fetcher)
        mock_fetcher.__aexit__ = AsyncMock(return_value=False)
        mock_fetcher_cls.return_value = mock_fetcher

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(gather_urls_action(state))

        # state.add_headlines deduplicates by URL
        assert len(state.headline_data) == 1

    @patch("steps.gather_urls.Fetcher")
    def test_skips_failed_sources(self, mock_fetcher_cls):
        from steps.gather_urls import gather_urls_action
        from state import NewsletterAgentState

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "source": "GoodSource",
                "results": [{"title": "Good Article Here", "url": "https://good.com/1", "source": "GoodSource"}],
                "status": "success",
                "metadata": {},
            },
            {
                "source": "BadSource",
                "results": [],
                "status": "error",
                "metadata": {"error": "Connection failed"},
            },
        ]
        mock_fetcher.__aenter__ = AsyncMock(return_value=mock_fetcher)
        mock_fetcher.__aexit__ = AsyncMock(return_value=False)
        mock_fetcher_cls.return_value = mock_fetcher

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(gather_urls_action(state))

        assert len(state.headline_data) == 1


class TestFilterUrls:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Url, AgentState
        AgentState.create_table(TEST_DB)
        Url.create_table(TEST_DB)
        yield

    def test_filters_skiplist_domains(self):
        from steps.filter_urls import _filter_skiplist

        headlines = [
            {"url": "https://example.com/good", "title": "Good Article"},
            {"url": "https://finbold.com/bad", "title": "Bad Domain"},
            {"url": "https://www.cnn.com/news", "title": "Ignored Domain"},
        ]
        filtered = _filter_skiplist(headlines)
        assert len(filtered) == 1
        assert filtered[0]["url"] == "https://example.com/good"

    @patch("steps.filter_urls.create_agent")
    def test_classifies_ai_relevance(self, mock_create):
        from steps.filter_urls import filter_urls_action
        from state import NewsletterAgentState

        mock_agent = AsyncMock()
        mock_agent.filter_dataframe.return_value = pd.Series([True, False], index=[0, 1])
        mock_create.return_value = mock_agent

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data = [
            {"url": "https://example.com/ai", "title": "AI Research Breakthrough"},
            {"url": "https://example.com/sports", "title": "Sports Final Scores"},
        ]

        result = asyncio.run(filter_urls_action(state))
        # Only AI-relevant article should remain
        assert len(state.headline_data) == 1
        assert "ai" in state.headline_data[0]["url"]


class TestDownloadArticles:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Article, Site, AgentState
        AgentState.create_table(TEST_DB)
        Article.create_table(TEST_DB)
        Site.create_table(TEST_DB)
        yield

    @patch("steps.download_articles.get_browser", new_callable=AsyncMock)
    @patch("steps.download_articles.scrape_urls_concurrent", new_callable=AsyncMock)
    def test_downloads_and_creates_articles(self, mock_scrape, mock_browser, tmp_path):
        from steps.download_articles import download_articles_action
        from state import NewsletterAgentState
        from db import Article
        from lib.scrape import ScrapeResult

        mock_browser.return_value = AsyncMock()

        text_file = tmp_path / "test.txt"
        text_file.write_text("This is test content. " * 50)

        mock_scrape.return_value = [
            ScrapeResult(
                html_path=str(tmp_path / "test.html"),
                text_path=str(text_file),
                final_url="https://example.com/article-1",
                last_updated="2026-02-13",
                status="success",
            ),
        ]

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data = [
            {"url": "https://example.com/article-1", "title": "Test Article", "source": "TestSource"},
        ]

        result = asyncio.run(download_articles_action(state))

        assert "1" in result  # mentions count
        articles = Article.get_all(TEST_DB)
        assert len(articles) == 1
        assert articles[0].title == "Test Article"
        assert articles[0].status == "success"
        assert articles[0].content_length > 0

    @patch("steps.download_articles.get_browser", new_callable=AsyncMock)
    @patch("steps.download_articles.scrape_urls_concurrent", new_callable=AsyncMock)
    def test_skips_failed_scrapes(self, mock_scrape, mock_browser, tmp_path):
        from steps.download_articles import download_articles_action
        from state import NewsletterAgentState
        from db import Article
        from lib.scrape import ScrapeResult

        mock_browser.return_value = AsyncMock()

        text_file = tmp_path / "ok.txt"
        text_file.write_text("OK content here. " * 50)

        mock_scrape.return_value = [
            ScrapeResult(status="error"),
            ScrapeResult(
                html_path=str(tmp_path / "ok.html"),
                text_path=str(text_file),
                final_url="https://example.com/ok",
                status="success",
            ),
        ]

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data = [
            {"url": "https://bad.com/fail", "title": "Failed", "source": "S1"},
            {"url": "https://example.com/ok", "title": "OK Article", "source": "S2"},
        ]

        result = asyncio.run(download_articles_action(state))
        articles = Article.get_all(TEST_DB)
        # Only the successful scrape creates an article
        assert len(articles) == 1
        assert articles[0].status == "success"

    def test_handles_empty_headlines(self):
        from steps.download_articles import download_articles_action
        from state import NewsletterAgentState

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data = []

        result = asyncio.run(download_articles_action(state))
        assert "No" in result
