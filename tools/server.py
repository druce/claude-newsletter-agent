"""FastMCP server registering all newsletter agent tools.

Run as MCP server: python tools/server.py
Or import `mcp` object for programmatic use.
"""
from __future__ import annotations

import json

from fastmcp import FastMCP

from config import NEWSAGENTDB
from steps import run_step

mcp = FastMCP("Newsletter Agent Tools")


@mcp.tool
def check_workflow_status(
    session_id: str, db_path: str = NEWSAGENTDB
) -> str:
    """Check which workflow steps are complete, in progress, or pending.

    Call this first to determine which step to execute next.
    Returns JSON with each step's status and overall progress.
    """
    import sqlite3

    from state import NewsletterAgentState

    try:
        state = NewsletterAgentState.load_latest_from_db(session_id, db_path=db_path)
    except sqlite3.OperationalError:
        state = None
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


@mcp.tool
def extract_summaries(
    session_id: str, db_path: str = NEWSAGENTDB
) -> str:
    """Step 4: Generate bullet-point summaries for downloaded articles.

    Reads articles from the DB that have text content but no summary.
    Uses Claude to generate 3-bullet summaries and distilled one-liners.
    Updates Article records in the database.
    """
    from tools.summarize import summarize_action

    result = run_step(
        step_name="extract_summaries",
        session_id=session_id,
        db_path=db_path,
        action=summarize_action,
    )
    return json.dumps(result)


@mcp.tool
def select_sections(
    session_id: str, db_path: str = NEWSAGENTDB
) -> str:
    """Step 7: Organize articles into newsletter sections/categories.

    Proposes categories from article content, assigns each article to a
    category, deduplicates within categories, and handles singletons.
    Updates newsletter_section_data in workflow state.
    """
    from tools.select_sections import select_sections_action

    result = run_step(
        step_name="select_sections",
        session_id=session_id,
        db_path=db_path,
        action=select_sections_action,
    )
    return json.dumps(result)


@mcp.tool
def draft_sections(
    session_id: str, db_path: str = NEWSAGENTDB
) -> str:
    """Step 8: Write newsletter section content with iterative quality refinement.

    Selects top stories, drafts each section with Claude, then runs a
    critique-optimize loop to improve headline quality and section coherence.
    Updates newsletter_section_data in workflow state.
    """
    from tools.draft_sections import draft_sections_action

    result = run_step(
        step_name="draft_sections",
        session_id=session_id,
        db_path=db_path,
        action=draft_sections_action,
    )
    return json.dumps(result)


@mcp.tool
def finalize_newsletter(
    session_id: str, db_path: str = NEWSAGENTDB
) -> str:
    """Step 9: Assemble final newsletter with title, critique-optimize, and email delivery.

    Assembles sections into markdown, generates a title, runs a full-newsletter
    critique-optimize loop, converts to HTML, sends via email, and saves to DB.
    """
    from tools.finalize import finalize_action

    result = run_step(
        step_name="finalize_newsletter",
        session_id=session_id,
        db_path=db_path,
        action=finalize_action,
    )
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run()
