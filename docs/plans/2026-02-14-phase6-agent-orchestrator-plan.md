# Phase 6: Agent Orchestrator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Python launcher (`agent.py`) that invokes Claude Code as the agent runtime, plus a `check_workflow_status` MCP tool, to orchestrate the full 9-step newsletter pipeline autonomously.

**Architecture:** Claude Code is the agent runtime. A thin Python launcher (`agent.py`) sets up the session, registers the FastMCP MCP server, and invokes `claude -p` with a system prompt describing the 9-step workflow. Claude Code calls Bash for data-heavy steps and MCP tools for LLM-centric steps.

**Tech Stack:** Python 3.11, Claude Code CLI, FastMCP, SQLite, subprocess

---

### Task 1: Add `check_workflow_status` MCP tool

**Files:**
- Modify: `tools/server.py` (add new `@mcp.tool` function)
- Test: `tests/test_tools_server.py` (add tests)

**Step 1: Write failing tests**

Add to `tests/test_tools_server.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools_server.py::TestCheckWorkflowStatus -v`
Expected: FAIL — `check_workflow_status` doesn't exist yet

**Step 3: Implement `check_workflow_status`**

Add to `tools/server.py`, after the existing imports and before the existing `@mcp.tool` functions:

```python
@mcp.tool
def check_workflow_status(
    session_id: str, db_path: str = NEWSAGENTDB
) -> str:
    """Check which workflow steps are complete, in progress, or pending.

    Call this first to determine which step to execute next.
    Returns JSON with each step's status and overall progress.
    """
    from state import NewsletterAgentState

    state = NewsletterAgentState.load_latest_from_db(session_id, db_path=db_path)
    if state is None:
        return json.dumps({
            "status": "new_session",
            "message": "No existing state. Start from step 1.",
            "steps": {}
        })

    steps = {}
    for step in state.steps:
        steps[step.id] = {
            "status": step.status.value,
            "message": step.status_message or step.error_message,
        }

    current = state.get_current_step()
    return json.dumps({
        "status": "ok",
        "progress": f"{state.get_progress_percentage():.0f}%",
        "next_step": current.id if current else "all_complete",
        "steps": steps
    })
```

**Step 4: Update existing server test for tool count**

In `tests/test_tools_server.py`, update `test_server_has_four_tools`:

```python
def test_server_has_five_tools(self):
    from tools.server import mcp

    tool_names = list(asyncio.run(mcp.get_tools()).keys())
    assert "check_workflow_status" in tool_names
    assert "extract_summaries" in tool_names
    assert "select_sections" in tool_names
    assert "draft_sections" in tool_names
    assert "finalize_newsletter" in tool_names
    assert len(tool_names) == 5
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tools_server.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add tools/server.py tests/test_tools_server.py
git commit -m "feat(tools): add check_workflow_status MCP tool"
```

---

### Task 2: Create `agent.py` launcher

**Files:**
- Create: `agent.py`
- Test: `tests/test_agent.py`

**Step 1: Write failing tests**

Create `tests/test_agent.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL — `agent` module doesn't exist yet

**Step 3: Implement `agent.py`**

Create `agent.py`:

```python
#!/usr/bin/env python3
"""Newsletter agent launcher — invokes Claude Code as the orchestrator."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

SYSTEM_PROMPT_TEMPLATE = """\
You are a newsletter agent that generates a weekly AI news digest.
You execute a 9-step workflow, checking status between steps.

## Workflow Steps

1. gather_urls      — Bash: python steps/gather_urls.py --session {session_id}
2. filter_urls      — Bash: python steps/filter_urls.py --session {session_id}
3. download_articles — Bash: python steps/download_articles.py --session {session_id}
4. extract_summaries — MCP tool: extract_summaries(session_id="{session_id}")
5. rate_articles    — Bash: python steps/rate_articles.py --session {session_id}
6. cluster_topics   — Bash: python steps/cluster_topics.py --session {session_id}
7. select_sections  — MCP tool: select_sections(session_id="{session_id}")
8. draft_sections   — MCP tool: draft_sections(session_id="{session_id}")
9. finalize_newsletter — MCP tool: finalize_newsletter(session_id="{session_id}")

## Execution Rules

1. First, call the check_workflow_status MCP tool to see which steps are complete.
2. Execute steps in order. Skip any step already marked "complete".
3. After each Bash step, check the JSON output. If "status": "error", STOP and report the error. Do not retry.
4. After each MCP tool, check the returned JSON. If "status": "error", STOP and report the error.
5. After all 9 steps complete, report the final status.

## Important

- Always pass --session {session_id} to Bash steps.
- Always pass session_id="{session_id}" to MCP tools.
- Bash steps may take several minutes. Be patient.
- Do NOT read or modify article data directly. All data flows through the DB and tools.
- Do NOT attempt to fix errors yourself. Stop and report them.
"""


def build_session_id(session: str | None) -> str:
    if session:
        return session
    return f"newsletter_{datetime.now():%Y%m%d}"


def build_system_prompt(session_id: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(session_id=session_id)


def build_user_message(session_id: str, resume: bool) -> str:
    if resume:
        return (
            f"Resume the newsletter workflow for session {session_id}. "
            "Check status and continue from the next incomplete step."
        )
    return f"Run the full newsletter workflow for session {session_id}."


def build_claude_command(
    user_msg: str,
    system_prompt: str,
    model: str,
    max_turns: int,
) -> list[str]:
    return [
        "claude", "-p", user_msg,
        "--append-system-prompt", system_prompt,
        "--dangerously-skip-permissions",
        "--model", model,
        "--max-turns", str(max_turns),
    ]


def validate_args(resume: bool, session: str | None) -> None:
    if resume and not session:
        print("Error: --resume requires --session", file=sys.stderr)
        sys.exit(1)


def register_mcp_server() -> None:
    """Register the newsletter MCP server with Claude Code (idempotent)."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(project_dir, "tools", "server.py")
    subprocess.run(
        ["claude", "mcp", "add", "--transport", "stdio", "newsletter", "--",
         sys.executable, server_path],
        check=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the newsletter agent")
    parser.add_argument("--session", type=str, default=None,
                        help="Session ID (default: newsletter_YYYYMMDD)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing session (requires --session)")
    parser.add_argument("--model", default="sonnet",
                        help="Claude model to use (default: sonnet)")
    parser.add_argument("--max-turns", type=int, default=50,
                        help="Max agent turns (default: 50)")
    args = parser.parse_args()

    validate_args(args.resume, args.session)

    session_id = build_session_id(args.session)
    system_prompt = build_system_prompt(session_id)
    user_msg = build_user_message(session_id, args.resume)

    register_mcp_server()

    cmd = build_claude_command(user_msg, system_prompt, args.model, args.max_turns)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(cmd, cwd=project_dir)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add agent.py tests/test_agent.py
git commit -m "feat: add agent.py launcher for Claude Code orchestration"
```

---

### Task 3: Update CLAUDE.md project status

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Update the Project Status section and Key Files:

```
## Project Status

**Phase 1: Foundation — COMPLETE** (config, state, db)
**Phase 2: LLM Layer — COMPLETE** (llm.py with multi-vendor support)
**Phase 3: Library modules — COMPLETE** (scrape, fetch, dedupe, rating, cluster)
**Phase 4: Bash-based steps — COMPLETE** (gather_urls, filter_urls, download_articles, rate_articles, cluster_topics)
**Phase 5: MCP tools — COMPLETE** (summarize, select_sections, draft_sections, finalize, FastMCP server)
**Phase 6: Agent orchestrator — COMPLETE** (agent.py launcher, check_workflow_status tool)
```

Add `agent.py` to Key Files:

```
agent.py       — Newsletter agent launcher (invokes Claude Code as orchestrator)
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update project status for Phase 6 completion"
```

---

### Task 4: End-to-end smoke test

**Files:** None (manual verification)

**Step 1: Verify MCP server starts**

Run: `python tools/server.py` — should start without errors. Ctrl+C to stop.

**Step 2: Verify all tests pass**

Run: `pytest tests/test_tools_server.py tests/test_agent.py -v`
Expected: All pass

**Step 3: Verify agent.py CLI help**

Run: `python agent.py --help`
Expected: Shows usage with --session, --resume, --model, --max-turns flags
