#!/usr/bin/env python3
"""Step 4: Extract summaries — generate bullet-point summaries + distilled one-liners.

Run via: python tools/summarize.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from config import NEWSAGENTDB
from db import Article
from llm import create_agent
from prompts import EXTRACT_SUMMARIES, ITEM_DISTILLER
from state import NewsletterAgentState
from steps import run_step
from tools.models import ArticleSummaryList, DistilledStoryList

logger = logging.getLogger(__name__)

SUMMARY_BATCH_SIZE = 10


def _load_unsummarized_articles(db_path: str) -> list[dict]:
    """Load articles that need summarization."""
    articles = Article.get_all(db_path)
    items = []
    for a in articles:
        if a.status != "success":
            continue
        if a.summary:  # already summarized
            continue
        if not a.text_path or not os.path.exists(a.text_path):
            continue
        try:
            with open(a.text_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            logger.warning("Cannot read text file for article %s", a.final_url)
            continue
        if not text.strip():
            continue
        items.append({
            "db_id": a.id,
            "id": a.id,
            "title": a.title,
            "text": text[:15000],  # truncate very long articles
            "rss_summary": a.rss_summary or "",
            "description": a.description or "",
        })
    return items


async def summarize_action(state: NewsletterAgentState) -> str:
    """Generate summaries for all unsummarized articles."""
    Article.create_table(state.db_path)

    items = _load_unsummarized_articles(state.db_path)
    if not items:
        return "No articles to summarize"

    # Stage 1: Generate 3-bullet summaries
    summary_agent = create_agent(
        model=EXTRACT_SUMMARIES.model,
        system_prompt=EXTRACT_SUMMARIES.system_prompt,
        user_prompt=EXTRACT_SUMMARIES.user_prompt,
        output_type=ArticleSummaryList,
        reasoning_effort=EXTRACT_SUMMARIES.reasoning_effort,
    )

    summaries = {}
    for i in range(0, len(items), SUMMARY_BATCH_SIZE):
        batch = items[i:i + SUMMARY_BATCH_SIZE]
        batch_input = [
            {"id": item["id"], "title": f"{item['title']}\n\n{item['text']}"}
            for item in batch
        ]
        results = await summary_agent.prompt_list(batch_input)
        for r in results:
            summaries[r["id"]] = r["summary"]

    # Stage 2: Generate short summaries (distilled one-liners)
    distill_items = []
    for item in items:
        summary = summaries.get(item["id"], "")
        if not summary:
            continue
        input_text = (
            f"{item['title']}\n"
            f"Description: {item['description']}\n"
            f"Summary:\n{summary}"
        )
        distill_items.append({"id": item["id"], "input_text": input_text})

    short_summaries = {}
    if distill_items:
        distill_agent = create_agent(
            model=ITEM_DISTILLER.model,
            system_prompt=ITEM_DISTILLER.system_prompt,
            user_prompt=ITEM_DISTILLER.user_prompt,
            output_type=DistilledStoryList,
            reasoning_effort=ITEM_DISTILLER.reasoning_effort,
        )
        for i in range(0, len(distill_items), SUMMARY_BATCH_SIZE):
            batch = distill_items[i:i + SUMMARY_BATCH_SIZE]
            results = await distill_agent.prompt_list(batch)
            for r in results:
                short_summaries[r["id"]] = r["short_summary"]

    # Update DB
    updated = 0
    for item in items:
        article = Article.get(state.db_path, item["db_id"])
        if article is None:
            continue
        summary = summaries.get(item["id"])
        if summary:
            article.summary = summary
            article.short_summary = short_summaries.get(item["id"], "")
            article.update(state.db_path)
            updated += 1

    return f"Summarized {updated} articles"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: Extract summaries")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="extract_summaries",
        session_id=args.session,
        db_path=args.db,
        action=summarize_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
