"""Article scoring — recency, length, composite rating, LLM assessments, Bradley-Terry battles."""
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime
from typing import Optional

import pandas as pd

from llm import create_agent
from prompts import RATE_QUALITY, RATE_ON_TOPIC, RATE_IMPORTANCE

logger = logging.getLogger(__name__)

LN2 = math.log(2)
MAX_ARTICLE_AGE_DAYS = 7


def compute_recency_score(published_date: Optional[datetime]) -> float:
    """Half-life of 1 day: 2 * exp(-ln2 * age_days) - 1.

    Returns 0.0 for articles older than MAX_ARTICLE_AGE_DAYS or None dates.
    """
    if published_date is None:
        return 0.0
    age = (datetime.now() - published_date).total_seconds() / 86400
    if age > MAX_ARTICLE_AGE_DAYS:
        return 0.0
    return 2 * math.exp(-LN2 * age) - 1


def compute_length_score(content_length: int) -> float:
    """log10(content_length) - 3, clipped to [0, 2]."""
    if content_length <= 0:
        return 0.0
    raw = math.log10(content_length) - 3
    return max(0.0, min(2.0, raw))


def compute_composite_rating(
    reputation: float,
    length_score: float,
    on_topic: float,
    importance: float,
    low_quality: float,
    recency: float,
    bt_zscore: float,
) -> float:
    """Composite article rating formula."""
    return reputation + length_score + on_topic + importance - low_quality + recency + bt_zscore


async def _assess_with_probs(df: pd.DataFrame, prompt_config) -> pd.Series:
    """Generic LLM probability assessment. Iterates rows, gets P(token='1')."""
    agent = create_agent(
        model=prompt_config.model,
        system_prompt=prompt_config.system_prompt,
        user_prompt=prompt_config.user_prompt,
        reasoning_effort=prompt_config.reasoning_effort,
    )

    async def _assess_one(row):
        input_text = f"Title: {row.get('title', '')}\nSummary: {row.get('summary', '')}"
        probs = await agent.run_prompt_with_probs(
            variables={"input_text": input_text},
            target_tokens=["1", "0"],
        )
        return probs.get("1", 0.0)

    # Run concurrently with semaphore (agent already has one)
    tasks = [_assess_one(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    return pd.Series(list(results), index=df.index)


async def assess_quality(df: pd.DataFrame) -> pd.Series:
    """LLM probability of low quality."""
    return await _assess_with_probs(df, RATE_QUALITY)


async def assess_on_topic(df: pd.DataFrame) -> pd.Series:
    """LLM probability of AI-relevance."""
    return await _assess_with_probs(df, RATE_ON_TOPIC)


async def assess_importance(df: pd.DataFrame) -> pd.Series:
    """LLM probability of importance."""
    return await _assess_with_probs(df, RATE_IMPORTANCE)
