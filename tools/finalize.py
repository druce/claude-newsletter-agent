#!/usr/bin/env python3
"""Step 9: Finalize newsletter — assemble, critique-optimize, HTML, email.

Run via: python tools/finalize.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from datetime import datetime
from typing import List

from config import MAX_CRITIQUE_ITERATIONS, NEWSAGENTDB
from db import Newsletter
from llm import create_agent
from prompts import (
    GENERATE_NEWSLETTER_TITLE,
    DRAFT_NEWSLETTER,
    CRITIQUE_NEWSLETTER,
    IMPROVE_NEWSLETTER,
)
from state import NewsletterAgentState
from steps import run_step
from tools.email_sender import markdown_to_html, send_gmail, wrap_newsletter_html
from tools.models import NewsletterCritique, StringResult

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 8.0


def _assemble_markdown(section_data: List[dict]) -> str:
    """Assemble section data into initial markdown draft."""
    # Group headlines by section_title preserving order
    sections: OrderedDict[str, list] = OrderedDict()
    for item in section_data:
        title = item.get("section_title", item.get("category", "Other"))
        if title not in sections:
            sections[title] = []
        links = item.get("links", [])
        link_strs = " ".join(
            f"[{l['site_name']}]({l['url']})" for l in links
        )
        headline = item["headline"]
        sections[title].append(f"- {headline} - {link_strs}")

    lines = []
    for section_title, headlines in sections.items():
        lines.append(f"## {section_title}")
        lines.extend(headlines)
        lines.append("")

    return "\n".join(lines)


async def finalize_action(state: NewsletterAgentState) -> str:
    """Assemble, polish, and deliver the final newsletter."""
    if not state.newsletter_section_data:
        return "No section data to finalize"

    Newsletter.create_table(state.db_path)

    # Stage 1: Assemble initial markdown
    initial_md = _assemble_markdown(state.newsletter_section_data)

    # Stage 2: Generate title
    title_agent = create_agent(
        model=GENERATE_NEWSLETTER_TITLE.model,
        system_prompt=GENERATE_NEWSLETTER_TITLE.system_prompt,
        user_prompt=GENERATE_NEWSLETTER_TITLE.user_prompt,
        output_type=StringResult,
        reasoning_effort=GENERATE_NEWSLETTER_TITLE.reasoning_effort,
    )
    title_result = await title_agent.prompt_dict({"input_text": initial_md})
    state.newsletter_title = title_result.result

    # Stage 3: Draft full newsletter
    draft_agent = create_agent(
        model=DRAFT_NEWSLETTER.model,
        system_prompt=DRAFT_NEWSLETTER.system_prompt,
        user_prompt=DRAFT_NEWSLETTER.user_prompt,
        output_type=StringResult,
        reasoning_effort=DRAFT_NEWSLETTER.reasoning_effort,
    )
    draft_result = await draft_agent.prompt_dict({"input_text": initial_md})
    current_newsletter = draft_result.result

    # Stage 4: Critique-optimize loop
    critique_agent = create_agent(
        model=CRITIQUE_NEWSLETTER.model,
        system_prompt=CRITIQUE_NEWSLETTER.system_prompt,
        user_prompt=CRITIQUE_NEWSLETTER.user_prompt,
        output_type=NewsletterCritique,
        reasoning_effort=CRITIQUE_NEWSLETTER.reasoning_effort,
    )

    final_score = 0.0
    iterations = 0

    for iteration in range(MAX_CRITIQUE_ITERATIONS):
        iterations = iteration + 1
        critique = await critique_agent.prompt_dict({"input_text": current_newsletter})
        final_score = critique.overall_score

        logger.info(
            "Critique iteration %d: score=%.1f, iterate=%s",
            iterations, final_score, critique.should_iterate,
        )

        if final_score >= QUALITY_THRESHOLD or not critique.should_iterate:
            break

        # Improve
        improve_agent = create_agent(
            model=IMPROVE_NEWSLETTER.model,
            system_prompt=IMPROVE_NEWSLETTER.system_prompt,
            user_prompt=IMPROVE_NEWSLETTER.user_prompt,
            output_type=StringResult,
            reasoning_effort=IMPROVE_NEWSLETTER.reasoning_effort,
        )
        improve_result = await improve_agent.prompt_dict({
            "newsletter": current_newsletter,
            "critique": critique.critique_text,
        })
        current_newsletter = improve_result.result

    state.final_newsletter = current_newsletter

    # Stage 5: HTML conversion + email
    today = datetime.now().strftime("%Y-%m-%d")
    newsletter_html = markdown_to_html(current_newsletter)
    styled_html = wrap_newsletter_html(newsletter_html, today)

    subject = f"AI News Digest — {state.newsletter_title}"
    try:
        send_gmail(subject, styled_html)
        email_status = "email sent"
    except Exception as e:
        logger.warning("Email delivery failed: %s", e)
        email_status = f"email failed: {e}"

    # Stage 6: Persist to DB
    Newsletter(
        session_id=state.session_id,
        date=datetime.now(),
        final_newsletter=current_newsletter,
    ).insert(state.db_path)

    return (
        f"Newsletter finalized (score: {final_score:.1f}, "
        f"{iterations} iterations), {email_status}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 9: Finalize newsletter")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="finalize_newsletter",
        session_id=args.session,
        db_path=args.db,
        action=finalize_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
