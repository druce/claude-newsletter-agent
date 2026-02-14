"""Tests for tools/server.py — FastMCP tool registration."""
import asyncio
import os
import json

import pytest
from unittest.mock import patch, MagicMock


TEST_DB = "test_server.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestMCPServer:
    def test_server_has_five_tools(self):
        from tools.server import mcp

        tool_names = list(asyncio.run(mcp.get_tools()).keys())
        assert "check_workflow_status" in tool_names
        assert "extract_summaries" in tool_names
        assert "select_sections" in tool_names
        assert "draft_sections" in tool_names
        assert "finalize_newsletter" in tool_names
        assert len(tool_names) == 5

    @patch("tools.server.run_step")
    def test_extract_summaries_tool_calls_run_step(self, mock_run_step):
        from tools.server import extract_summaries

        mock_run_step.return_value = {"status": "success", "message": "Summarized 5 articles"}

        result = extract_summaries.fn(session_id="test_123", db_path=TEST_DB)
        parsed = json.loads(result)

        assert parsed["status"] == "success"
        mock_run_step.assert_called_once()
        call_args = mock_run_step.call_args
        assert call_args[1]["step_name"] == "extract_summaries"
        assert call_args[1]["session_id"] == "test_123"

    @patch("tools.server.run_step")
    def test_finalize_tool_calls_run_step(self, mock_run_step):
        from tools.server import finalize_newsletter

        mock_run_step.return_value = {"status": "success", "message": "Done"}

        result = finalize_newsletter.fn(session_id="test_123")
        parsed = json.loads(result)
        assert parsed["status"] == "success"


class TestCheckWorkflowStatus:
    def test_new_session_returns_no_state(self):
        from tools.server import check_workflow_status

        result = json.loads(check_workflow_status.fn(
            session_id="nonexistent_session", db_path=TEST_DB
        ))
        assert result["status"] == "new_session"
        assert result["steps"] == {}

    def test_partial_completion(self):
        from db import AgentState as AgentStateDB
        from state import NewsletterAgentState
        from tools.server import check_workflow_status

        AgentStateDB.create_table(TEST_DB)
        state = NewsletterAgentState(session_id="test_partial", db_path=TEST_DB)
        state.complete_step("gather_urls", message="Gathered 100 URLs")
        state.complete_step("filter_urls", message="Filtered to 50")
        state.save_checkpoint("filter_urls")

        result = json.loads(check_workflow_status.fn(
            session_id="test_partial", db_path=TEST_DB
        ))
        assert result["status"] == "ok"
        assert result["next_step"] == "download_articles"
        assert result["steps"]["gather_urls"]["status"] == "complete"
        assert result["steps"]["filter_urls"]["status"] == "complete"
        assert result["steps"]["download_articles"]["status"] == "not_started"

    def test_all_complete(self):
        from db import AgentState as AgentStateDB
        from state import NewsletterAgentState
        from tools.server import check_workflow_status

        AgentStateDB.create_table(TEST_DB)
        state = NewsletterAgentState(session_id="test_done", db_path=TEST_DB)
        for step in state.steps:
            state.complete_step(step.id, message="Done")
        state.save_checkpoint("finalize_newsletter")

        result = json.loads(check_workflow_status.fn(
            session_id="test_done", db_path=TEST_DB
        ))
        assert result["status"] == "ok"
        assert result["next_step"] == "all_complete"
        assert result["progress"] == "100%"

    def test_error_step_shown(self):
        from db import AgentState as AgentStateDB
        from state import NewsletterAgentState
        from tools.server import check_workflow_status

        AgentStateDB.create_table(TEST_DB)
        state = NewsletterAgentState(session_id="test_err", db_path=TEST_DB)
        state.complete_step("gather_urls", message="OK")
        state.error_step("filter_urls", "Connection timeout")
        state.save_checkpoint("filter_urls")

        result = json.loads(check_workflow_status.fn(
            session_id="test_err", db_path=TEST_DB
        ))
        assert result["steps"]["filter_urls"]["status"] == "error"
        assert "timeout" in result["steps"]["filter_urls"]["message"].lower()
