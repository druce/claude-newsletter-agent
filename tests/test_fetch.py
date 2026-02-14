# tests/test_fetch.py
"""Tests for lib/fetch.py — source fetching and link extraction."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCleanSummary:
    def test_strips_html_tags(self):
        from lib.fetch import _clean_summary
        assert _clean_summary("<p>Hello <b>world</b></p>") == "Hello world"

    def test_empty_input(self):
        from lib.fetch import _clean_summary
        assert _clean_summary("") == ""

    def test_plain_text_passthrough(self):
        from lib.fetch import _clean_summary
        assert _clean_summary("No HTML here") == "No HTML here"

    def test_html_entities(self):
        from lib.fetch import _clean_summary
        result = _clean_summary("AT&amp;T &lt;rocks&gt;")
        assert result == "AT&T <rocks>"

    def test_nested_tags(self):
        from lib.fetch import _clean_summary
        raw = "<div><p>First <a href='#'>link</a></p><p>Second</p></div>"
        assert _clean_summary(raw) == "First link Second"


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


class TestFetchRssMetadata:
    @pytest.mark.asyncio
    async def test_returns_url_keyed_metadata(self):
        from lib.fetch import Fetcher

        rss_xml = """<?xml version="1.0"?>
        <rss version="2.0"><channel><title>Test</title>
            <item>
                <title>Article One</title>
                <link>https://example.com/one</link>
                <pubDate>Thu, 13 Feb 2026 12:00:00 GMT</pubDate>
                <description><![CDATA[<p>Summary <b>one</b></p>]]></description>
            </item>
            <item>
                <title>Article Two</title>
                <link>https://example.com/two?utm_source=rss</link>
                <pubDate>Fri, 14 Feb 2026 08:00:00 GMT</pubDate>
            </item>
        </channel></rss>"""

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value=rss_xml)

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_resp

        fetcher = Fetcher(sources={"TestSrc": {"type": "html", "url": "https://example.com", "rss": "https://example.com/feed"}})
        fetcher._session = mock_session

        meta = await fetcher._fetch_rss_metadata("TestSrc")
        assert "https://example.com/one" in meta
        assert meta["https://example.com/one"]["published"] == "Thu, 13 Feb 2026 12:00:00 GMT"
        # Summary should be cleaned of HTML
        assert meta["https://example.com/one"]["summary"] == "Summary one"
        # utm params should be stripped from URLs
        assert "https://example.com/two" in meta

    @pytest.mark.asyncio
    async def test_returns_empty_on_http_error(self):
        from lib.fetch import Fetcher

        mock_resp = AsyncMock()
        mock_resp.status = 400

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_resp

        fetcher = Fetcher(sources={"Src": {"type": "html", "url": "https://x.com", "rss": "https://x.com/feed"}})
        fetcher._session = mock_session
        assert await fetcher._fetch_rss_metadata("Src") == {}

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        from lib.fetch import Fetcher

        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Network error")

        fetcher = Fetcher(sources={"Src": {"type": "html", "url": "https://x.com", "rss": "https://x.com/feed"}})
        fetcher._session = mock_session
        assert await fetcher._fetch_rss_metadata("Src") == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_rss_field(self):
        from lib.fetch import Fetcher

        fetcher = Fetcher(sources={"Src": {"type": "html", "url": "https://x.com"}})
        fetcher._session = AsyncMock()
        assert await fetcher._fetch_rss_metadata("Src") == {}


class TestFetchHtmlRssEnrichment:
    @pytest.mark.asyncio
    @patch("lib.fetch.Fetcher._fetch_rss_metadata")
    @patch("lib.fetch.parse_source_links")
    @patch("lib.fetch.get_browser", new_callable=AsyncMock)
    @patch("lib.fetch.scrape_url", new_callable=AsyncMock)
    async def test_enriches_links_with_rss_metadata(self, mock_scrape, mock_browser, mock_parse, mock_rss_meta):
        from lib.fetch import Fetcher
        from lib.scrape import ScrapeResult

        mock_scrape.return_value = ScrapeResult(html_path="/tmp/test.html", status="success")
        mock_parse.return_value = [
            {"title": "Article One", "url": "https://example.com/one"},
            {"title": "Article Two", "url": "https://example.com/two"},
        ]
        mock_rss_meta.return_value = {
            "https://example.com/one": {"published": "2026-02-13", "summary": "Summary one"},
        }

        fetcher = Fetcher(sources={"Src": {"type": "html", "url": "https://example.com", "rss": "https://example.com/feed", "filename": "src"}})
        fetcher._session = AsyncMock()
        result = await fetcher.fetch_html("Src")

        assert result["status"] == "success"
        results = result["results"]
        # First link enriched
        assert results[0]["published"] == "2026-02-13"
        assert results[0]["summary"] == "Summary one"
        # Second link not enriched — no published/summary keys
        assert "published" not in results[1]

    @pytest.mark.asyncio
    @patch("lib.fetch.parse_source_links")
    @patch("lib.fetch.get_browser", new_callable=AsyncMock)
    @patch("lib.fetch.scrape_url", new_callable=AsyncMock)
    async def test_no_rss_field_skips_enrichment(self, mock_scrape, mock_browser, mock_parse):
        from lib.fetch import Fetcher
        from lib.scrape import ScrapeResult

        mock_scrape.return_value = ScrapeResult(html_path="/tmp/test.html", status="success")
        mock_parse.return_value = [
            {"title": "Article", "url": "https://example.com/art"},
        ]

        fetcher = Fetcher(sources={"Src": {"type": "html", "url": "https://example.com", "filename": "src"}})
        fetcher._session = AsyncMock()
        result = await fetcher.fetch_html("Src")

        assert result["status"] == "success"
        assert "published" not in result["results"][0]


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
