#!/usr/bin/env python3
"""Step 7: Select sections — organize articles into newsletter categories.

Run via: python tools/select_sections.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Dict, List

from config import NEWSAGENTDB
from db import Article
from llm import create_agent
from prompts import CAT_PROPOSAL, CAT_CLEANUP, CAT_ASSIGNMENT, DEDUPE_ARTICLES
from state import NewsletterAgentState
from steps import run_step
from tools.models import (
    TopicCategoryList,
    TopicAssignment,
    DedupeResultList,
)

logger = logging.getLogger(__name__)


def _load_articles_for_selection(db_path: str) -> List[dict]:
    """Load rated + clustered articles from DB."""
    articles = Article.get_all(db_path)
    items = []
    for a in articles:
        if a.status != "success":
            continue
        if not a.summary:
            continue
        items.append({
            "db_id": a.id,
            "id": a.id,
            "final_url": a.final_url,
            "title": a.title,
            "summary": a.summary,
            "short_summary": a.short_summary or "",
            "rating": a.rating,
            "cluster_label": a.cluster_label or "",
            "topics": a.topics or "",
            "source": a.source,
            "site_name": a.site_name,
            "domain": a.domain,
        })
    return items


def _format_articles_for_proposal(items: List[dict]) -> str:
    """Format articles as markdown for the category proposal prompt."""
    lines = []
    for item in sorted(items, key=lambda x: x["rating"], reverse=True):
        lines.append(f"{item['title']} - {item['site_name']}")
        lines.append(f"Rating: {item['rating']:.1f}")
        if item["topics"]:
            lines.append(f"Topics: {item['topics']}")
        lines.append(f"{item['summary']}")
        lines.append("---")
    return "\n".join(lines)


async def _propose_categories(items: List[dict]) -> List[str]:
    """Use LLM to propose 10-30 topic categories."""
    agent = create_agent(
        model=CAT_PROPOSAL.model,
        system_prompt=CAT_PROPOSAL.system_prompt,
        user_prompt=CAT_PROPOSAL.user_prompt,
        output_type=TopicCategoryList,
        reasoning_effort=CAT_PROPOSAL.reasoning_effort,
    )
    formatted = _format_articles_for_proposal(items)
    result = await agent.prompt_dict({"input_text": formatted})
    return result.categories


async def _cleanup_categories(categories: List[str]) -> List[str]:
    """Use LLM to deduplicate and polish category names."""
    agent = create_agent(
        model=CAT_CLEANUP.model,
        system_prompt=CAT_CLEANUP.system_prompt,
        user_prompt=CAT_CLEANUP.user_prompt,
        output_type=TopicCategoryList,
        reasoning_effort=CAT_CLEANUP.reasoning_effort,
    )
    cats_text = "\n".join(f"- {c}" for c in categories)
    result = await agent.prompt_dict({"input_text": cats_text})
    return result.categories


async def _assign_articles(
    items: List[dict], categories: List[str]
) -> Dict[int, str]:
    """Assign each article to a category via parallel LLM calls."""
    agent = create_agent(
        model=CAT_ASSIGNMENT.model,
        system_prompt=CAT_ASSIGNMENT.system_prompt,
        user_prompt=CAT_ASSIGNMENT.user_prompt,
        output_type=TopicAssignment,
        reasoning_effort=CAT_ASSIGNMENT.reasoning_effort,
    )
    topics_str = "\n".join(categories)

    async def _assign_one(item: dict) -> tuple[int, str]:
        input_text = f"{item['title']}\n{item['short_summary'] or item['summary']}"
        result = await agent.prompt_dict({
            "topics": topics_str,
            "input_text": input_text,
        })
        return item["id"], result.topic_title

    tasks = [_assign_one(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assignments = {}
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Assignment failed: %s", r)
            continue
        article_id, category = r
        assignments[article_id] = category

    return assignments


async def _dedupe_category(
    items: List[dict],
) -> List[dict]:
    """Deduplicate articles within a category. Returns items with dupe_id field."""
    if len(items) <= 1:
        for item in items:
            item["dupe_id"] = -1
        return items

    agent = create_agent(
        model=DEDUPE_ARTICLES.model,
        system_prompt=DEDUPE_ARTICLES.system_prompt,
        user_prompt=DEDUPE_ARTICLES.user_prompt,
        output_type=DedupeResultList,
        reasoning_effort=DEDUPE_ARTICLES.reasoning_effort,
    )
    input_items = [{"id": item["id"], "summary": item["summary"]} for item in items]
    results = await agent.prompt_list(
        input_items,
        item_id_field="id",
        item_list_field="results_list",
    )

    dupe_map = {r["id"]: r["dupe_id"] for r in results}
    for item in items:
        item["dupe_id"] = dupe_map.get(item["id"], -1)
    return items


def _handle_singletons(
    assignments: Dict[int, str], items_by_id: Dict[int, dict]
) -> Dict[int, str]:
    """Move articles from singleton categories (1 article) to 'Other'."""
    category_counts: Dict[str, int] = {}
    for cat in assignments.values():
        category_counts[cat] = category_counts.get(cat, 0) + 1

    result = {}
    for article_id, cat in assignments.items():
        if category_counts[cat] == 1 and cat != "Other":
            result[article_id] = "Other"
        else:
            result[article_id] = cat
    return result


async def select_sections_action(state: NewsletterAgentState) -> str:
    """Organize articles into newsletter sections."""
    Article.create_table(state.db_path)

    items = _load_articles_for_selection(state.db_path)
    if not items:
        return "No articles to organize"

    items_by_id = {item["id"]: item for item in items}

    # Stage 1: Propose categories
    categories = await _propose_categories(items)
    logger.info("Proposed %d categories", len(categories))

    # Stage 2: Clean up categories
    categories = await _cleanup_categories(categories)
    logger.info("Cleaned to %d categories", len(categories))

    # Stage 3: Assign articles to categories
    assignments = await _assign_articles(items, categories)

    # Stage 4: Handle singletons
    assignments = _handle_singletons(assignments, items_by_id)

    # Stage 5: Deduplicate within categories
    categories_with_items: Dict[str, List[dict]] = {}
    for article_id, cat in assignments.items():
        if cat not in categories_with_items:
            categories_with_items[cat] = []
        item = items_by_id.get(article_id)
        if item:
            categories_with_items[cat].append(item)

    deduped_items = []
    for cat, cat_items in categories_with_items.items():
        result = await _dedupe_category(cat_items)
        for item in result:
            if item["dupe_id"] == -1:
                item["category"] = cat
                deduped_items.append(item)

    # Build newsletter_section_data
    section_data = []
    for item in deduped_items:
        section_data.append({
            "id": item["id"],
            "db_id": item["db_id"],
            "category": item["category"],
            "title": item["title"],
            "summary": item["summary"],
            "short_summary": item["short_summary"],
            "rating": item["rating"],
            "final_url": item["final_url"],
            "site_name": item["site_name"],
            "source": item["source"],
        })

    state.newsletter_section_data = section_data

    n_sections = len(set(item["category"] for item in section_data))
    return f"Organized {len(section_data)} articles into {n_sections} sections"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: Select sections")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="select_sections",
        session_id=args.session,
        db_path=args.db,
        action=select_sections_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
