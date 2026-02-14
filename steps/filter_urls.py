#!/usr/bin/env python3
"""Step 2: Filter URLs — domain skiplist + LLM AI-relevance classification.

Run via: python steps/filter_urls.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd
from pydantic import BaseModel

from config import DOMAIN_SKIPLIST, IGNORE_LIST, NEWSAGENTDB
from db import Url
from llm import create_agent
from prompts import FILTER_URLS
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Schema for FILTER_URLS structured output."""
    id: int
    value: bool


class FilterResultList(BaseModel):
    results_list: List[FilterResult]


def _filter_skiplist(headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove headlines from domains in DOMAIN_SKIPLIST and IGNORE_LIST."""
    blocked = set(DOMAIN_SKIPLIST) | set(IGNORE_LIST)
    filtered = []
    for h in headlines:
        url = h.get("url", "")
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Check both with and without www prefix
        bare = hostname.removeprefix("www.")
        if hostname in blocked or bare in blocked:
            continue
        filtered.append(h)
    return filtered


async def filter_urls_action(state: NewsletterAgentState) -> str:
    """Filter headlines by domain skiplist, then classify AI-relevance via LLM."""
    initial_count = len(state.headline_data)

    # Step 1: Domain skiplist
    state.headline_data = _filter_skiplist(state.headline_data)
    after_skiplist = len(state.headline_data)

    # Step 2: LLM classification
    if not state.headline_data:
        return f"No headlines to filter (started with {initial_count})"

    df = pd.DataFrame(state.headline_data).reset_index(drop=True)
    df["id"] = df.index

    agent = create_agent(
        model=FILTER_URLS.model,
        system_prompt=FILTER_URLS.system_prompt,
        user_prompt=FILTER_URLS.user_prompt,
        output_type=FilterResultList,
        reasoning_effort=FILTER_URLS.reasoning_effort,
    )

    is_ai = await agent.filter_dataframe(
        df=df[["id", "title"]],
        chunk_size=25,
        value_field="value",
        item_list_field="results_list",
        item_id_field="id",
    )

    # Keep only AI-relevant headlines
    ai_mask = is_ai.astype(bool)
    ai_urls = set(df.loc[ai_mask, "url"].tolist()) if "url" in df.columns else set()
    state.headline_data = [h for h in state.headline_data if h.get("url") in ai_urls]

    # Update Url records
    Url.create_table(state.db_path)
    for _, row in df.iterrows():
        url_str = row.get("url", "")
        is_ai_val = bool(ai_mask.get(row.name, False))
        try:
            from db import _connect
            with _connect(state.db_path) as conn:
                conn.execute(
                    "UPDATE urls SET isAI = ? WHERE initial_url = ?",
                    (int(is_ai_val), url_str),
                )
        except Exception:
            pass

    after_filter = len(state.headline_data)
    return (
        f"Filtered {initial_count} -> {after_skiplist} (skiplist) -> "
        f"{after_filter} AI-relevant headlines"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Filter URLs")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="filter_urls",
        session_id=args.session,
        db_path=args.db,
        action=filter_urls_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
