#!/usr/bin/env python3
"""Newsletter agent launcher — invokes Claude Code as the orchestrator."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

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
    server_path = os.path.join(PROJECT_DIR, "tools", "server.py")
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
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
