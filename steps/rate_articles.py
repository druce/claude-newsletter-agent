#!/usr/bin/env python3
"""Step 5: Rate articles — LLM assessments + Bradley-Terry battles.

Run via: python steps/rate_articles.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from config import NEWSAGENTDB
from db import Article
from lib.rating import rate_articles as rate_articles_lib
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


def _articles_to_dataframe(db_path: str) -> pd.DataFrame:
    """Load all successful articles from DB into a DataFrame."""
    articles = Article.get_all(db_path)
    if not articles:
        return pd.DataFrame()

    rows = []
    for a in articles:
        if a.status != "success":
            continue
        rows.append({
            "db_id": a.id,
            "id": a.id,
            "final_url": a.final_url,
            "title": a.title,
            "summary": a.summary or a.rss_summary or "",
            "content_length": a.content_length,
            "published": a.published,
            "reputation": a.reputation or 0.0,
            "source": a.source,
            "domain": a.domain,
        })
    return pd.DataFrame(rows)


async def rate_articles_action(state: NewsletterAgentState) -> str:
    """Load articles, run rating pipeline, update DB."""
    Article.create_table(state.db_path)

    df = _articles_to_dataframe(state.db_path)
    if df.empty:
        return "No articles to rate"

    rated_df = await rate_articles_lib(df)

    # Update Article records with rating data
    for _, row in rated_df.iterrows():
        db_id = int(row["db_id"])
        article = Article.get(state.db_path, db_id)
        if article is None:
            continue
        article.rating = float(row["rating"])
        article.update(state.db_path)

    return f"Rated {len(rated_df)} articles"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: Rate articles")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="rate_articles",
        session_id=args.session,
        db_path=args.db,
        action=rate_articles_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
