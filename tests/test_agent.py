"""Tests for agent.py — newsletter agent launcher."""
import subprocess
from unittest.mock import patch, MagicMock
from datetime import datetime


class TestSessionId:
    def test_default_session_id_uses_today(self):
        from agent import build_session_id

        result = build_session_id(session=None)
        expected = f"newsletter_{datetime.now():%Y%m%d}"
        assert result == expected

    def test_explicit_session_id(self):
        from agent import build_session_id

        result = build_session_id(session="my_session")
        assert result == "my_session"


class TestSystemPrompt:
    def test_prompt_contains_session_id(self):
        from agent import build_system_prompt

        prompt = build_system_prompt("test_session_123")
        assert "test_session_123" in prompt

    def test_prompt_contains_all_steps(self):
        from agent import build_system_prompt

        prompt = build_system_prompt("s1")
        assert "gather_urls" in prompt
        assert "filter_urls" in prompt
        assert "download_articles" in prompt
        assert "extract_summaries" in prompt
        assert "rate_articles" in prompt
        assert "cluster_topics" in prompt
        assert "select_sections" in prompt
        assert "draft_sections" in prompt
        assert "finalize_newsletter" in prompt

    def test_prompt_contains_check_status_instruction(self):
        from agent import build_system_prompt

        prompt = build_system_prompt("s1")
        assert "check_workflow_status" in prompt


class TestBuildCommand:
    def test_basic_command(self):
        from agent import build_claude_command

        cmd = build_claude_command(
            user_msg="Run workflow",
            system_prompt="You are an agent",
            model="sonnet",
            max_turns=50,
        )
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "Run workflow" in cmd
        assert "--append-system-prompt" in cmd
        assert "You are an agent" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--model" in cmd
        assert "sonnet" in cmd
        assert "--max-turns" in cmd
        assert "50" in cmd

    def test_resume_user_message(self):
        from agent import build_user_message

        msg = build_user_message(session_id="s1", resume=True)
        assert "Resume" in msg or "resume" in msg
        assert "s1" in msg

    def test_fresh_user_message(self):
        from agent import build_user_message

        msg = build_user_message(session_id="s1", resume=False)
        assert "s1" in msg


class TestResumeValidation:
    def test_resume_without_session_raises(self):
        import pytest
        from agent import validate_args

        with pytest.raises(SystemExit):
            validate_args(resume=True, session=None)

    def test_resume_with_session_ok(self):
        from agent import validate_args

        validate_args(resume=True, session="my_session")  # no exception
