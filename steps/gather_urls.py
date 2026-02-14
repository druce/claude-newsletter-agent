#!/usr/bin/env python3
"""Step 1: Gather URLs from all sources.

Run via: python steps/gather_urls.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime

from config import NEWSAGENTDB
from db import Url
from lib.fetch import Fetcher
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


async def gather_urls_action(state: NewsletterAgentState) -> str:
    """Fetch all sources and persist URLs to DB and state."""
    Url.create_table(state.db_path)

    async with Fetcher(sources_file=state.sources_file) as fetcher:
        results = await fetcher.fetch_all()

    total_urls = 0
    total_sources = 0
    failed_sources = 0

    for source_result in results:
        source_name = source_result["source"]
        if source_result["status"] != "success":
            failed_sources += 1
            logger.warning("Source %s failed: %s", source_name, source_result.get("metadata", {}))
            continue

        total_sources += 1
        for item in source_result["results"]:
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title:
                continue

            # Persist to Url table (skip duplicates)
            try:
                record = Url(
                    initial_url=url,
                    final_url=url,
                    title=title,
                    source=item.get("source", source_name),
                    created_at=datetime.now(),
                )
                record.insert(state.db_path)
            except sqlite3.IntegrityError:
                pass  # URL already exists

            total_urls += 1

    # Add to state (handles dedup internally)
    all_headlines = []
    for source_result in results:
        if source_result["status"] == "success":
            all_headlines.extend(source_result["results"])
    state.add_headlines(all_headlines)

    return f"Gathered {total_urls} URLs from {total_sources} sources ({failed_sources} failed)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Gather URLs from sources")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="gather_urls",
        session_id=args.session,
        db_path=args.db,
        action=gather_urls_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
