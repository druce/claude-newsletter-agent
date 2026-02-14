# tests/test_scrape.py
"""Tests for lib/scrape.py — utility functions and RateLimiter."""
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


class TestIsValidHtml:
    def test_valid_html_with_doctype(self):
        from lib.scrape import is_valid_html
        html = "<!DOCTYPE html><html><head></head><body>" + "<p>Content</p>" * 100 + "</body></html>"
        assert is_valid_html(html) is True

    def test_valid_html_without_doctype(self):
        from lib.scrape import is_valid_html
        html = "<html><head></head><body>" + "<p>Content</p>" * 100 + "</body></html>"
        assert is_valid_html(html) is True

    def test_valid_html_tag_only(self):
        from lib.scrape import is_valid_html
        html = "<HTML lang='en'><head></head><body>" + "<p>Content</p>" * 100 + "</body></html>"
        assert is_valid_html(html) is True

    def test_binary_content_rejected(self):
        from lib.scrape import is_valid_html
        # Simulate binary garbage wrapped in HTML tags (>30% non-ASCII)
        binary_chars = "".join(chr(i) for i in range(128, 256)) * 50
        content = "<html><head></head><body>" + binary_chars
        assert is_valid_html(content) is False

    def test_too_short_rejected(self):
        from lib.scrape import is_valid_html
        assert is_valid_html("<html><body>short</body></html>") is False

    def test_custom_min_length(self):
        from lib.scrape import is_valid_html
        html = "<html><body>short</body></html>"
        assert is_valid_html(html, min_length=10) is True

    def test_no_html_markers_rejected(self):
        from lib.scrape import is_valid_html
        content = "Just some plain text without any HTML markers. " * 50
        assert is_valid_html(content) is False

    def test_empty_string_rejected(self):
        from lib.scrape import is_valid_html
        assert is_valid_html("") is False

    def test_legitimate_utf8_passes(self):
        from lib.scrape import is_valid_html
        # Accented characters stay well under the 30% threshold
        body = "<p>Les données résumées pour l'année en français.</p>" * 20
        html = "<!DOCTYPE html><html><body>" + body + "</body></html>"
        assert is_valid_html(html) is True


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


from unittest.mock import AsyncMock, MagicMock, patch


class TestGetBrowser:
    @pytest.mark.asyncio
    @patch("lib.scrape._create_browser_context", new_callable=AsyncMock)
    @patch("lib.scrape.async_playwright")
    async def test_returns_browser_context(self, mock_pw, mock_create):
        from lib.scrape import get_browser, _reset_browser_cache
        import lib.scrape as scrape_mod
        _reset_browser_cache()
        scrape_mod._playwright_instance = None
        mock_pw_instance = AsyncMock()
        mock_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)
        mock_create.return_value = AsyncMock()
        ctx = await get_browser("/tmp/test_profile")
        assert ctx is not None

    @pytest.mark.asyncio
    @patch("lib.scrape._create_browser_context", new_callable=AsyncMock)
    @patch("lib.scrape.async_playwright")
    async def test_caches_browser_context(self, mock_pw, mock_create):
        from lib.scrape import get_browser, _reset_browser_cache
        import lib.scrape as scrape_mod
        _reset_browser_cache()
        scrape_mod._playwright_instance = None
        mock_pw_instance = AsyncMock()
        mock_pw.return_value.start = AsyncMock(return_value=mock_pw_instance)
        mock_create.return_value = AsyncMock()
        await get_browser("/tmp/test_profile")
        await get_browser("/tmp/test_profile")
        # Should only create one browser context
        assert mock_create.call_count == 1


class TestScrapeUrl:
    @pytest.mark.asyncio
    @patch("lib.scrape.asyncio.sleep", new_callable=AsyncMock)
    @patch("lib.scrape.enable_fast_mode", new_callable=AsyncMock)
    async def test_saves_html_and_text(self, mock_fast, mock_sleep, tmp_path):
        from lib.scrape import scrape_url, ScrapeResult
        mock_page = AsyncMock()
        mock_page.content.return_value = (
            "<!DOCTYPE html><html><body>"
            + "<p>AI article content here.</p>" * 50
            + "</body></html>"
        )
        mock_page.url = "https://example.com/final-article"
        mock_page.evaluate = AsyncMock(return_value=None)

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        result = await scrape_url(
            url="https://example.com/article",
            title="Test Article",
            browser_context=mock_context,
            html_dir=str(tmp_path / "html"),
            text_dir=str(tmp_path / "text"),
        )
        assert isinstance(result, ScrapeResult)
        assert result.status in ("success", "no_content")
        assert result.final_url == "https://example.com/final-article"
        mock_fast.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_navigation_error(self):
        from lib.scrape import scrape_url, ScrapeResult
        mock_page = AsyncMock()
        mock_page.goto.side_effect = Exception("Navigation failed")

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        result = await scrape_url(
            url="https://example.com/broken",
            title="Broken",
            browser_context=mock_context,
        )
        assert result.status == "error"

    @pytest.mark.asyncio
    @patch("lib.scrape.asyncio.sleep", new_callable=AsyncMock)
    @patch("lib.scrape.enable_fast_mode", new_callable=AsyncMock)
    async def test_returns_corrupt_for_binary_content(self, mock_fast, mock_sleep, tmp_path):
        from lib.scrape import scrape_url
        binary_chars = "".join(chr(i) for i in range(128, 256)) * 50
        corrupt_html = "<html><head></head><body>" + binary_chars

        mock_page = AsyncMock()
        mock_page.content.return_value = corrupt_html
        mock_page.url = "https://example.com/corrupt"
        mock_page.evaluate = AsyncMock(return_value="")

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        result = await scrape_url(
            url="https://example.com/corrupt",
            title="Corrupt Page",
            browser_context=mock_context,
            html_dir=str(tmp_path / "html"),
            text_dir=str(tmp_path / "text"),
        )
        assert result.status == "corrupt"
        assert result.html_path == ""
        assert result.text_path == ""

    @pytest.mark.asyncio
    @patch("lib.scrape.asyncio.sleep", new_callable=AsyncMock)
    @patch("lib.scrape.enable_fast_mode", new_callable=AsyncMock)
    async def test_does_not_write_files_for_corrupt_content(self, mock_fast, mock_sleep, tmp_path):
        from lib.scrape import scrape_url
        binary_chars = "".join(chr(i) for i in range(128, 256)) * 50
        corrupt_html = "<html><head></head><body>" + binary_chars

        mock_page = AsyncMock()
        mock_page.content.return_value = corrupt_html
        mock_page.url = "https://example.com/corrupt"
        mock_page.evaluate = AsyncMock(return_value="")

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        html_dir = tmp_path / "html"
        text_dir = tmp_path / "text"

        await scrape_url(
            url="https://example.com/corrupt",
            title="Corrupt Page",
            browser_context=mock_context,
            html_dir=str(html_dir),
            text_dir=str(text_dir),
        )
        # Directories should not be created for corrupt content
        assert not html_dir.exists()
        assert not text_dir.exists()

    @pytest.mark.asyncio
    @patch("lib.scrape.asyncio.sleep", new_callable=AsyncMock)
    @patch("lib.scrape.enable_fast_mode", new_callable=AsyncMock)
    async def test_valid_content_still_succeeds(self, mock_fast, mock_sleep, tmp_path):
        """Regression: good HTML still works after adding is_valid_html check."""
        from lib.scrape import scrape_url
        good_html = "<!DOCTYPE html><html><body>" + "<p>Real article content.</p>" * 100 + "</body></html>"

        mock_page = AsyncMock()
        mock_page.content.return_value = good_html
        mock_page.url = "https://example.com/good"
        mock_page.evaluate = AsyncMock(return_value="Good Article")

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        result = await scrape_url(
            url="https://example.com/good",
            title="Good Article",
            browser_context=mock_context,
            html_dir=str(tmp_path / "html"),
            text_dir=str(tmp_path / "text"),
        )
        assert result.status in ("success", "no_content")
        assert result.html_path != ""


class TestScrapeUrlsConcurrent:
    @pytest.mark.asyncio
    @patch("lib.scrape.scrape_url")
    async def test_processes_multiple_urls(self, mock_scrape):
        from lib.scrape import scrape_urls_concurrent, ScrapeResult
        mock_scrape.return_value = ScrapeResult(
            html_path="test.html", text_path="test.txt",
            final_url="https://example.com", status="success"
        )
        urls = [
            {"url": "https://a.com/1", "title": "A"},
            {"url": "https://b.com/2", "title": "B"},
        ]
        mock_context = AsyncMock()
        results = await scrape_urls_concurrent(urls, mock_context, concurrency=2)
        assert len(results) == 2
        assert all(r.status == "success" for r in results)
