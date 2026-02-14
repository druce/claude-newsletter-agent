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
