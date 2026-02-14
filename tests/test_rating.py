# tests/test_rating.py
"""Tests for lib/rating.py — article scoring and Bradley-Terry battles."""
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestComputeRecencyScore:
    def test_today_scores_one(self):
        from lib.rating import compute_recency_score
        score = compute_recency_score(datetime.now())
        assert abs(score - 1.0) < 0.05

    def test_one_day_ago_scores_zero(self):
        from lib.rating import compute_recency_score
        score = compute_recency_score(datetime.now() - timedelta(days=1))
        assert abs(score) < 0.05

    def test_two_days_ago_negative(self):
        from lib.rating import compute_recency_score
        score = compute_recency_score(datetime.now() - timedelta(days=2))
        assert score < 0

    def test_old_article_clamped(self):
        from lib.rating import compute_recency_score
        score = compute_recency_score(datetime.now() - timedelta(days=30))
        assert score == 0.0

    def test_none_returns_zero(self):
        from lib.rating import compute_recency_score
        score = compute_recency_score(None)
        assert score == 0.0


class TestComputeLengthScore:
    def test_thousand_chars(self):
        from lib.rating import compute_length_score
        # log10(1000) - 3 = 0
        assert compute_length_score(1000) == 0.0

    def test_ten_thousand_chars(self):
        from lib.rating import compute_length_score
        # log10(10000) - 3 = 1
        assert abs(compute_length_score(10000) - 1.0) < 0.01

    def test_zero_length(self):
        from lib.rating import compute_length_score
        assert compute_length_score(0) == 0.0

    def test_clipped_at_two(self):
        from lib.rating import compute_length_score
        # log10(1_000_000) - 3 = 3 → clipped to 2
        assert compute_length_score(1_000_000) == 2.0

    def test_negative_clipped_at_zero(self):
        from lib.rating import compute_length_score
        # log10(100) - 3 = -1 → clipped to 0
        assert compute_length_score(100) == 0.0


class TestCompositeRating:
    def test_formula(self):
        from lib.rating import compute_composite_rating
        score = compute_composite_rating(
            reputation=1.5,
            length_score=1.0,
            on_topic=0.8,
            importance=0.7,
            low_quality=0.1,
            recency=0.5,
            bt_zscore=0.0,
        )
        expected = 1.5 + 1.0 + 0.8 + 0.7 - 0.1 + 0.5 + 0.0
        assert abs(score - expected) < 0.001
