#!/usr/bin/env python3
"""Step 3: Download article content via browser scraping.

Run via: python steps/download_articles.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from typing import Dict, List

import tldextract

from config import FIREFOX_PROFILE_PATH, NEWSAGENTDB, PAGES_DIR, TEXT_DIR
from db import Article, Site
from lib.scrape import get_browser, scrape_urls_concurrent
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


def _get_content_length(text_path: str) -> int:
    """Read text file and return character count."""
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            return len(f.read())
    except (FileNotFoundError, OSError):
        return 0


def _extract_domain(url: str) -> str:
    """Extract registerable domain from URL."""
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


async def download_articles_action(state: NewsletterAgentState) -> str:
    """Scrape article content for all headlines and create Article records."""
    Article.create_table(state.db_path)
    Site.create_table(state.db_path)

    if not state.headline_data:
        return "No headlines to download"

    # Prepare URL list for scraper
    urls_to_scrape: List[Dict[str, str]] = [
        {"url": h["url"], "title": h.get("title", "")}
        for h in state.headline_data
        if h.get("url")
    ]

    browser = await get_browser(FIREFOX_PROFILE_PATH)
    results = await scrape_urls_concurrent(
        urls=urls_to_scrape,
        browser_context=browser,
        html_dir=PAGES_DIR,
        text_dir=TEXT_DIR,
    )

    success_count = 0
    error_count = 0

    for headline, scrape_result in zip(state.headline_data, results):
        url = headline.get("url", "")
        title = headline.get("title", "")
        source = headline.get("source", "")
        domain = _extract_domain(url)

        if scrape_result.status not in ("success", "no_content"):
            error_count += 1
            continue

        content_length = _get_content_length(scrape_result.text_path)

        # Upsert Site record
        try:
            site = Site(domain_name=domain, site_name=source)
            site.insert(state.db_path)
        except sqlite3.IntegrityError:
            pass

        # Create Article record
        try:
            article = Article(
                final_url=scrape_result.final_url or url,
                url=url,
                source=source,
                title=title,
                html_path=scrape_result.html_path,
                text_path=scrape_result.text_path,
                content_length=content_length,
                domain=domain,
                site_name=source,
                status=scrape_result.status,
                last_updated=scrape_result.last_updated or None,
                rss_summary=headline.get("summary"),
                published=headline.get("published"),
            )
            article.insert(state.db_path)
            success_count += 1
        except sqlite3.IntegrityError:
            # Article already exists (same final_url) — skip
            success_count += 1

    return f"Downloaded {success_count} articles ({error_count} failed)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Download articles")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="download_articles",
        session_id=args.session,
        db_path=args.db,
        action=download_articles_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
