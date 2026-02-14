"""Article scoring — recency, length, composite rating, LLM assessments, Bradley-Terry battles."""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

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
