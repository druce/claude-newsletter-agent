#!/usr/bin/env python3
"""Step 8: Draft sections — write newsletter content with iterative critique.

Run via: python tools/draft_sections.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Dict, List

from config import MAX_CRITIQUE_ITERATIONS, NEWSAGENTDB
from llm import create_agent
from prompts import WRITE_SECTION, CRITIQUE_SECTION
from state import NewsletterAgentState
from steps import run_step
from tools.models import Section, SectionCritique

logger = logging.getLogger(__name__)

TIER1_THRESHOLD = 8.0
MAX_STORIES = 100
QUALITY_THRESHOLD = 7.0


def _select_stories(
    items: List[dict],
    max_stories: int = MAX_STORIES,
    tier1_threshold: float = TIER1_THRESHOLD,
) -> List[dict]:
    """Select stories for the newsletter: Tier 1 (must-include) + Tier 2 (by rating)."""
    tier1 = [item for item in items if item["rating"] > tier1_threshold]
    tier2_candidates = [item for item in items if item["rating"] <= tier1_threshold]
    tier2_candidates.sort(key=lambda x: x["rating"], reverse=True)

    remaining_slots = max(0, max_stories - len(tier1))
    tier2 = tier2_candidates[:remaining_slots]

    selected = tier1 + tier2
    return selected[:max_stories]


def _group_by_category(items: List[dict]) -> Dict[str, List[dict]]:
    """Group items by category."""
    groups: Dict[str, List[dict]] = {}
    for item in items:
        cat = item.get("category", "Other")
        if cat not in groups:
            groups[cat] = []
        groups[cat].append(item)
    return groups


async def _draft_one_section(
    category: str, items: List[dict], write_agent
) -> Section:
    """Draft a single newsletter section."""
    # Format items for the prompt
    section_items = []
    for item in sorted(items, key=lambda x: x["rating"], reverse=True):
        section_items.append({
            "rating": item["rating"],
            "summary": item.get("short_summary") or item["summary"],
            "site_name": item.get("site_name", ""),
            "url": item["final_url"],
        })

    input_text = json.dumps(section_items)
    result = await write_agent.prompt_dict({"input_text": input_text})
    return result


async def _critique_one_section(
    section_title: str,
    headlines_text: str,
    target_categories: List[str],
    critique_agent,
) -> SectionCritique:
    """Critique a single section."""
    cats_str = "\n".join(f"- {c}" for c in target_categories)
    result = await critique_agent.prompt_dict({
        "section_title": section_title,
        "target_categories": cats_str,
        "input_text": headlines_text,
    })
    return result


def _apply_critique(
    section: Section, critique: SectionCritique
) -> Section:
    """Apply critique actions to a section."""
    action_map = {a.id: a for a in critique.actions}

    new_headlines = []
    for i, headline in enumerate(section.headlines):
        action = action_map.get(i)
        if action is None:
            new_headlines.append(headline)
            continue
        if action.action == "drop":
            continue
        if action.action == "rewrite" and action.rewritten_headline:
            headline.headline = action.rewritten_headline
        new_headlines.append(headline)

    return Section(section_title=section.section_title, headlines=new_headlines)


def _sections_to_state_data(
    sections: Dict[str, Section], category_items: Dict[str, List[dict]]
) -> List[dict]:
    """Convert Section objects to newsletter_section_data format."""
    result = []
    for cat, section in sections.items():
        for headline in section.headlines:
            if headline.prune:
                continue
            links = [{"site_name": l.site_name, "url": l.url} for l in headline.links]
            result.append({
                "category": cat,
                "section_title": section.section_title,
                "headline": headline.headline,
                "rating": headline.rating,
                "links": links,
            })
    return result


async def draft_sections_action(state: NewsletterAgentState) -> str:
    """Draft newsletter sections with iterative critique."""
    if not state.newsletter_section_data:
        return "No section data to draft"

    items = state.newsletter_section_data
    selected = _select_stories(items)
    grouped = _group_by_category(selected)

    if not grouped:
        return "No categories to draft"

    # Create agents
    write_agent = create_agent(
        model=WRITE_SECTION.model,
        system_prompt=WRITE_SECTION.system_prompt,
        user_prompt=WRITE_SECTION.user_prompt,
        output_type=Section,
        reasoning_effort=WRITE_SECTION.reasoning_effort,
    )

    critique_agent = create_agent(
        model=CRITIQUE_SECTION.model,
        system_prompt=CRITIQUE_SECTION.system_prompt,
        user_prompt=CRITIQUE_SECTION.user_prompt,
        output_type=SectionCritique,
        reasoning_effort=CRITIQUE_SECTION.reasoning_effort,
    )

    # Draft all sections in parallel
    all_categories = list(grouped.keys())
    draft_tasks = [
        _draft_one_section(cat, grouped[cat], write_agent)
        for cat in all_categories
    ]
    draft_results = await asyncio.gather(*draft_tasks, return_exceptions=True)

    sections: Dict[str, Section] = {}
    for cat, result in zip(all_categories, draft_results):
        if isinstance(result, Exception):
            logger.warning("Draft failed for %s: %s", cat, result)
            continue
        sections[cat] = result

    # Critique-optimize loop
    for iteration in range(MAX_CRITIQUE_ITERATIONS):
        low_quality = []

        for cat, section in sections.items():
            headlines_text = "\n".join(
                f"- [{i}] {h.headline} (rating: {h.rating})"
                for i, h in enumerate(section.headlines)
            )
            critique = await _critique_one_section(
                section.section_title, headlines_text, all_categories, critique_agent
            )

            if critique.quality_score < QUALITY_THRESHOLD:
                low_quality.append(cat)

            sections[cat] = _apply_critique(section, critique)

        if not low_quality:
            logger.info("All sections pass quality threshold after %d iterations", iteration + 1)
            break

        # Re-draft low-quality sections
        for cat in low_quality:
            if cat in grouped:
                try:
                    sections[cat] = await _draft_one_section(
                        cat, grouped[cat], write_agent
                    )
                except Exception as e:
                    logger.warning("Re-draft failed for %s: %s", cat, e)

    # Convert to state data
    state.newsletter_section_data = _sections_to_state_data(sections, grouped)

    total_headlines = sum(
        len([h for h in s.headlines if not h.prune]) for s in sections.values()
    )
    return f"Drafted {len(sections)} sections with {total_headlines} headlines"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: Draft sections")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="draft_sections",
        session_id=args.session,
        db_path=args.db,
        action=draft_sections_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
