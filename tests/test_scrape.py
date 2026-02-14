# tests/test_scrape.py
"""Tests for lib/scrape.py — utility functions and RateLimiter."""
import asyncio
import os
import pytest


class TestSanitizeFilename:
    def test_basic_sanitize(self):
        from lib.scrape import sanitize_filename
        assert sanitize_filename("Hello World!") == "Hello_World"

    def test_url_characters(self):
        from lib.scrape import sanitize_filename
        result = sanitize_filename("https://example.com/path?q=1")
        assert "/" not in result
        assert "?" not in result
        assert ":" not in result

    def test_long_filename_truncated(self):
        from lib.scrape import sanitize_filename
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 200

    def test_empty_string(self):
        from lib.scrape import sanitize_filename
        result = sanitize_filename("")
        assert result == "unnamed"


class TestCleanUrl:
    def test_strips_utm_params(self):
        from lib.scrape import clean_url
        url = "https://example.com/article?utm_source=twitter&utm_medium=social"
        result = clean_url(url)
        assert "utm_" not in result
        assert result == "https://example.com/article"

    def test_preserves_meaningful_params(self):
        from lib.scrape import clean_url
        url = "https://example.com/search?q=ai+news"
        result = clean_url(url)
        assert "q=ai" in result

    def test_strips_fragment(self):
        from lib.scrape import clean_url
        url = "https://example.com/article#comments"
        result = clean_url(url)
        assert "#" not in result


class TestNormalizeHtml:
    def test_extracts_text_from_html(self, tmp_path):
        from lib.scrape import normalize_html
        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><body><article><p>This is the main article text about AI.</p></article></body></html>"
        )
        text = normalize_html(str(html_file))
        assert "main article text" in text

    def test_returns_empty_for_missing_file(self):
        from lib.scrape import normalize_html
        result = normalize_html("/nonexistent/path.html")
        assert result == ""


class TestScrapeResult:
    def test_dataclass_fields(self):
        from lib.scrape import ScrapeResult
        r = ScrapeResult(
            html_path="download/html/test.html",
            text_path="download/text/test.txt",
            final_url="https://example.com/article",
            last_updated="2026-02-13",
            status="success",
        )
        assert r.html_path == "download/html/test.html"
        assert r.status == "success"

    def test_default_status(self):
        from lib.scrape import ScrapeResult
        r = ScrapeResult()
        assert r.status == "pending"


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_first_request_allowed(self):
        from lib.scrape import RateLimiter
        limiter = RateLimiter(rate_limit_seconds=5.0)
        can_proceed, wait_time = await limiter.try_acquire("example.com")
        assert can_proceed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_second_request_blocked(self):
        from lib.scrape import RateLimiter
        limiter = RateLimiter(rate_limit_seconds=5.0)
        await limiter.try_acquire("example.com")
        can_proceed, wait_time = await limiter.try_acquire("example.com")
        assert can_proceed is False
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_different_domains_independent(self):
        from lib.scrape import RateLimiter
        limiter = RateLimiter(rate_limit_seconds=5.0)
        await limiter.try_acquire("example.com")
        can_proceed, _ = await limiter.try_acquire("other.com")
        assert can_proceed is True

    @pytest.mark.asyncio
    async def test_daily_cap(self):
        from lib.scrape import RateLimiter
        limiter = RateLimiter(rate_limit_seconds=0.0, daily_cap=2)
        await limiter.try_acquire("example.com")
        await limiter.try_acquire("example.com")
        can_proceed, _ = await limiter.try_acquire("example.com")
        assert can_proceed is False
