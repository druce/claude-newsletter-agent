"""Pydantic output models for structured LLM responses in MCP tools.

These models define the JSON schemas used by create_agent(output_type=...)
for all tool LLM calls. Each model corresponds to a specific prompt's
expected output format.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


# --- Extract Summaries (Step 4) ---

class ArticleSummary(BaseModel):
    id: int
    summary: str


class ArticleSummaryList(BaseModel):
    results_list: List[ArticleSummary]


class DistilledStory(BaseModel):
    id: int
    short_summary: str


class DistilledStoryList(BaseModel):
    results_list: List[DistilledStory]


# --- Select Sections (Step 7) ---

class TopicCategoryList(BaseModel):
    categories: List[str]


class TopicAssignment(BaseModel):
    topic_title: str


class DedupeResult(BaseModel):
    id: int
    dupe_id: int


class DedupeResultList(BaseModel):
    results_list: List[DedupeResult]


class FilterResult(BaseModel):
    id: int
    output: bool


class FilterResultList(BaseModel):
    results_list: List[FilterResult]


# --- Draft Sections (Step 8) ---

class HeadlineLink(BaseModel):
    site_name: str
    url: str


class Headline(BaseModel):
    headline: str
    rating: float
    prune: bool
    links: List[HeadlineLink]


class Section(BaseModel):
    section_title: str
    headlines: List[Headline]


class StoryAction(BaseModel):
    id: int
    action: str  # "keep", "drop", "rewrite", "move"
    rewritten_headline: Optional[str] = None
    target_category: Optional[str] = None


class SectionCritique(BaseModel):
    coherence_score: float
    quality_score: float
    actions: List[StoryAction]


# --- Finalize Newsletter (Step 9) ---

class NewsletterCritique(BaseModel):
    overall_score: float
    title_quality: float
    structure_quality: float
    section_quality: float
    headline_quality: float
    should_iterate: bool
    critique_text: str


class StringResult(BaseModel):
    result: str
