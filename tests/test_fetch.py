# tests/test_fetch.py
"""Tests for lib/fetch.py — source fetching and link extraction."""
import pytest
from unittest.mock import AsyncMock, patch


class TestParseSourceLinks:
    def test_extracts_matching_links(self, tmp_path):
        from lib.fetch import parse_source_links
        html_file = tmp_path / "test.html"
        html_file.write_text("""
        <html><body>
            <a href="https://example.com/ai-article-1">AI Article 1 Long Enough Title</a>
            <a href="https://example.com/sports-news">Sports News With A Long Title</a>
            <a href="https://example.com/ai-article-2">AI Article Two With Long Title</a>
        </body></html>
        """)
        config = {
            "include": [r"^https://example\.com/ai-"],
            "minlength": 5,
        }
        results = parse_source_links(str(html_file), config)
        assert len(results) == 2
        assert all("ai-article" in r["url"] for r in results)

    def test_excludes_matching_links(self, tmp_path):
        from lib.fetch import parse_source_links
        html_file = tmp_path / "test.html"
        html_file.write_text("""
        <html><body>
            <a href="https://example.com/good-article">Good Article Title Here Long Enough</a>
            <a href="https://example.com/video/bad">Bad Video Link Here Long Enough Title</a>
        </body></html>
        """)
        config = {
            "exclude": [r"/video/"],
            "minlength": 5,
        }
        results = parse_source_links(str(html_file), config)
        assert len(results) == 1
        assert "good-article" in results[0]["url"]

    def test_filters_short_titles(self, tmp_path):
        from lib.fetch import parse_source_links
        html_file = tmp_path / "test.html"
        html_file.write_text("""
        <html><body>
            <a href="https://example.com/a">Short</a>
            <a href="https://example.com/b">This Is A Sufficiently Long Title For The Article</a>
        </body></html>
        """)
        config = {"minlength": 28}
        results = parse_source_links(str(html_file), config)
        assert len(results) == 1

    def test_returns_standard_format(self, tmp_path):
        from lib.fetch import parse_source_links
        html_file = tmp_path / "test.html"
        html_file.write_text("""
        <html><body>
            <a href="https://example.com/article">This Is A Sufficiently Long Title For The Article</a>
        </body></html>
        """)
        config = {}
        results = parse_source_links(str(html_file), config)
        assert "url" in results[0]
        assert "title" in results[0]


class TestGetOgTags:
    def test_extracts_og_tags(self, tmp_path):
        from lib.fetch import get_og_tags
        html_file = tmp_path / "og.html"
        html_file.write_text("""
        <html><head>
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG Description">
            <meta property="og:url" content="https://example.com/canonical">
        </head><body></body></html>
        """)
        tags = get_og_tags(str(html_file))
        assert tags["og:title"] == "OG Title"
        assert tags["og:url"] == "https://example.com/canonical"


class TestFetcherRss:
    @pytest.mark.asyncio
    @patch("lib.fetch.aiohttp.ClientSession")
    async def test_parses_rss_entries(self, mock_session_cls):
        from lib.fetch import Fetcher
        # Create mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="""<?xml version="1.0"?>
        <rss version="2.0">
        <channel>
            <title>Test Feed</title>
            <item>
                <title>AI Breakthrough in Healthcare</title>
                <link>https://example.com/ai-health</link>
                <pubDate>Thu, 13 Feb 2026 12:00:00 GMT</pubDate>
                <description>Summary text</description>
            </item>
        </channel>
        </rss>""")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response
        mock_session.close = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        fetcher = Fetcher(
            sources={"TestSource": {"type": "rss", "rss": "https://example.com/feed.xml"}},
        )
        fetcher._session = mock_session
        result = await fetcher.fetch_rss("TestSource")
        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "AI Breakthrough in Healthcare"
        assert result["results"][0]["source"] == "TestSource"


class TestFetcherFetchAll:
    @pytest.mark.asyncio
    @patch("lib.fetch.Fetcher.fetch_rss")
    @patch("lib.fetch.Fetcher.fetch_html")
    async def test_dispatches_by_type(self, mock_html, mock_rss):
        from lib.fetch import Fetcher
        mock_rss.return_value = {"source": "RSS", "results": [{"title": "T", "url": "u"}], "status": "success", "metadata": {}}
        mock_html.return_value = {"source": "HTML", "results": [{"title": "T", "url": "u"}], "status": "success", "metadata": {}}

        fetcher = Fetcher(
            sources={
                "RSSSource": {"type": "rss", "rss": "https://feed.xml"},
                "HTMLSource": {"type": "html", "url": "https://page.com", "filename": "test"},
            },
        )
        fetcher._session = AsyncMock()
        results = await fetcher.fetch_all()
        assert len(results) == 2
        source_names = [r["source"] for r in results]
        assert "RSS" in source_names or "RSSSource" in source_names
