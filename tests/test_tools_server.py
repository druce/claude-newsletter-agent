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
    def test_server_has_four_tools(self):
        from tools.server import mcp

        tool_names = list(asyncio.run(mcp.get_tools()).keys())
        assert "extract_summaries" in tool_names
        assert "select_sections" in tool_names
        assert "draft_sections" in tool_names
        assert "finalize_newsletter" in tool_names

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
