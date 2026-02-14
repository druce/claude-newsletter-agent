# tests/test_rating.py
"""Tests for lib/rating.py — article scoring and Bradley-Terry battles."""
from datetime import datetime, timedelta
import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch


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


class TestAssessQuality:
    @pytest.mark.asyncio
    @patch("lib.rating.create_agent")
    async def test_returns_series(self, mock_create):
        from lib.rating import assess_quality
        mock_agent = AsyncMock()
        mock_agent.run_prompt_with_probs.return_value = {"1": 0.2}
        mock_create.return_value = mock_agent

        df = pd.DataFrame({
            "id": [1, 2],
            "title": ["Good article", "Bad clickbait"],
            "summary": ["Solid reporting", "You won't believe"],
        }, index=[0, 1])
        result = await assess_quality(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


class TestAssessOnTopic:
    @pytest.mark.asyncio
    @patch("lib.rating.create_agent")
    async def test_returns_series(self, mock_create):
        from lib.rating import assess_on_topic
        mock_agent = AsyncMock()
        mock_agent.run_prompt_with_probs.return_value = {"1": 0.9}
        mock_create.return_value = mock_agent

        df = pd.DataFrame({
            "id": [1],
            "title": ["AI Model Release"],
            "summary": ["OpenAI released new model"],
        })
        result = await assess_on_topic(df)
        assert len(result) == 1


class TestAssessImportance:
    @pytest.mark.asyncio
    @patch("lib.rating.create_agent")
    async def test_returns_series(self, mock_create):
        from lib.rating import assess_importance
        mock_agent = AsyncMock()
        mock_agent.run_prompt_with_probs.return_value = {"1": 0.8}
        mock_create.return_value = mock_agent

        df = pd.DataFrame({
            "id": [1],
            "title": ["Major AI Policy"],
            "summary": ["Government announces AI regulation"],
        })
        result = await assess_importance(df)
        assert len(result) == 1
