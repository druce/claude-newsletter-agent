"""Source fetching -- RSS, HTML landing pages, REST API."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
import requests
import yaml
from bs4 import BeautifulSoup

from config import DOWNLOAD_DIR, MIN_TITLE_LEN, DEFAULT_CONCURRENCY, SHORT_REQUEST_TIMEOUT
from lib.scrape import scrape_url, get_browser, clean_url

logger = logging.getLogger(__name__)

# Pattern to detect number-only link text (e.g. page navigation "1", "2", "3")
_NUMBER_ONLY_RE = re.compile(r"^\d+$")


def _clean_summary(raw: str) -> str:
    """Strip HTML tags from an RSS/API summary, returning plain text."""
    if not raw:
        return ""
    return BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)


# ---------------------------------------------------------------------------
# HTML link extraction
# ---------------------------------------------------------------------------

def parse_source_links(html_path: str, source_config: dict) -> List[dict]:
    """Extract article links from a saved HTML file.

    Reads the HTML file at *html_path*, parses all ``<a>`` tags, and applies
    the filtering rules from *source_config*:

    - ``include`` (list[str]): regex patterns -- keep only URLs matching at
      least one pattern.  If omitted, all URLs pass.
    - ``exclude`` (list[str]): regex patterns -- drop URLs matching any
      pattern.
    - ``minlength`` (int): minimum title character length (default
      ``MIN_TITLE_LEN``).

    Returns a list of ``{"title": str, "url": str}`` dicts.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as exc:
        logger.error("Could not read %s: %s", html_path, exc)
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Determine base URL for resolving relative links
    base_tag = soup.find("base")
    base_url = str(base_tag["href"]) if base_tag and base_tag.get("href") else source_config.get("url", "")

    include_patterns = source_config.get("include", [])
    exclude_patterns = source_config.get("exclude", [])
    minlength = source_config.get("minlength", MIN_TITLE_LEN)

    results: List[dict] = []
    seen_urls: set = set()

    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"]).strip()
        text = a_tag.get_text(strip=True)

        # Skip empty text or number-only text (pagination links)
        if not text or _NUMBER_ONLY_RE.match(text):
            continue

        # Resolve relative URLs
        if base_url and not href.startswith(("http://", "https://")):
            href = urljoin(base_url, href)

        # Skip non-http links (mailto:, javascript:, etc.)
        if not href.startswith(("http://", "https://")):
            continue

        # Skip URLs with empty path (top-level domain only)
        parsed = urlparse(href)
        if not parsed.path or parsed.path == "/":
            continue

        # Clean tracking params
        href = clean_url(href)

        # Deduplicate
        if href in seen_urls:
            continue
        seen_urls.add(href)

        # Apply exclude patterns -- drop if any match
        if exclude_patterns and any(re.search(pat, href) for pat in exclude_patterns):
            continue

        # Apply include patterns -- keep only if at least one matches
        if include_patterns and not any(re.search(pat, href) for pat in include_patterns):
            continue

        # Filter by title length
        if len(text) < minlength:
            continue

        results.append({"title": text, "url": href})

    return results


# ---------------------------------------------------------------------------
# OpenGraph tag extraction
# ---------------------------------------------------------------------------

def get_og_tags(html_path: str) -> dict:
    """Extract OpenGraph meta tags from a saved HTML file.

    Returns a dict like ``{"og:title": "...", "og:description": "..."}``.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as exc:
        logger.error("Could not read %s: %s", html_path, exc)
        return {}

    soup = BeautifulSoup(html, "html.parser")
    tags: dict = {}

    for meta in soup.find_all("meta"):
        prop = str(meta.get("property", ""))
        if prop.startswith("og:") and meta.get("content"):
            tags[prop] = str(meta["content"])

    return tags


# ---------------------------------------------------------------------------
# Fetcher class
# ---------------------------------------------------------------------------

class Fetcher:
    """Async source fetcher that dispatches to RSS, HTML, or API handlers.

    Parameters
    ----------
    sources:
        Dict of source configurations.  If ``None``, loads from
        *sources_file*.
    sources_file:
        Path to a YAML file with source definitions (default
        ``sources.yaml``).
    max_concurrent:
        Maximum number of concurrent fetch operations.
    """

    def __init__(
        self,
        sources: Optional[Dict[str, dict]] = None,
        sources_file: str = "sources.yaml",
        max_concurrent: int = DEFAULT_CONCURRENCY,
    ) -> None:
        if sources is not None:
            self.sources = sources
            logger.info("[fetcher_init] Initialized with %d provided sources", len(self.sources))
        else:
            logger.info("[fetcher_init] Loading sources from %s", sources_file)
            with open(sources_file, "r", encoding="utf-8") as f:
                self.sources = yaml.safe_load(f)

            # Log source breakdown
            rss_sources = [k for k, v in self.sources.items() if v.get("type") == "rss"]
            html_sources = [k for k, v in self.sources.items() if v.get("type") == "html"]
            api_sources = [k for k, v in self.sources.items() if v.get("type") == "rest"]
            logger.info(
                "[fetcher_init] Loaded %d sources: %d RSS, %d HTML, %d API",
                len(self.sources), len(rss_sources), len(html_sources), len(api_sources),
            )

        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("[fetcher_init] Fetcher initialized with max_concurrent=%d", max_concurrent)

    async def __aenter__(self) -> Fetcher:
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ---- RSS ---------------------------------------------------------------

    async def fetch_rss(self, source_key: str) -> dict:
        """Fetch and parse an RSS feed for *source_key*.

        Returns
        -------
        dict
            ``{"source": str, "results": list, "status": str, "metadata": dict}``
        """
        source = self.sources[source_key]
        rss_url = source.get("rss", source.get("url", ""))

        if self._session is None:
            return {
                "source": source_key,
                "results": [],
                "status": "error",
                "metadata": {"error": "No active session"},
            }

        try:
            logger.info("[fetch_rss] Fetching RSS from %s: %s", source_key, rss_url)
            resp = await self._session.get(
                rss_url,
                timeout=aiohttp.ClientTimeout(total=SHORT_REQUEST_TIMEOUT),
            )
            if resp.status != 200:
                return {
                    "source": source_key,
                    "results": [],
                    "status": "error",
                    "metadata": {"http_status": resp.status},
                }
            content = await resp.text()

            feed = feedparser.parse(content)

            entries: List[dict] = []
            for entry in feed.entries:
                title = str(getattr(entry, "title", "") or "").strip()
                link = str(getattr(entry, "link", "") or "").strip()
                if not title or not link:
                    continue

                link = clean_url(link)

                result_entry: dict = {
                    "title": title,
                    "url": link,
                    "source": source_key,
                }

                # Add optional fields
                pub = getattr(entry, "published", None)
                if pub:
                    result_entry["published"] = str(pub)
                summ = getattr(entry, "summary", None)
                if summ:
                    result_entry["summary"] = _clean_summary(str(summ))

                entries.append(result_entry)

            feed_title = str(getattr(feed.feed, "title", "") or "")
            logger.info("[fetch_rss] RSS fetch successful for %s: %d articles", source_key, len(entries))
            return {
                "source": source_key,
                "results": entries,
                "status": "success",
                "metadata": {"feed_title": feed_title, "entry_count": len(entries)},
            }

        except Exception as exc:
            logger.error("RSS fetch failed for %s: %s", source_key, exc)
            return {
                "source": source_key,
                "results": [],
                "status": "error",
                "metadata": {"error": str(exc)},
            }

    # ---- RSS metadata enrichment -------------------------------------------

    async def _fetch_rss_metadata(self, source_key: str) -> Dict[str, dict]:
        """Fetch RSS feed for a source and return metadata keyed by cleaned URL.

        Returns ``{url: {"published": ..., "summary": ...}}``.
        Returns empty dict on any error (best-effort enrichment).
        """
        source = self.sources[source_key]
        rss_url = source.get("rss", "")
        if not rss_url or self._session is None:
            return {}

        try:
            logger.info("[rss_metadata] Fetching RSS metadata for %s: %s", source_key, rss_url)
            resp = await self._session.get(
                rss_url,
                timeout=aiohttp.ClientTimeout(total=SHORT_REQUEST_TIMEOUT),
            )
            if resp.status != 200:
                logger.warning("[rss_metadata] HTTP %d for %s", resp.status, source_key)
                return {}
            content = await resp.text()
            feed = feedparser.parse(content)

            metadata: Dict[str, dict] = {}
            for entry in feed.entries:
                link = str(getattr(entry, "link", "") or "").strip()
                if not link:
                    continue
                link = clean_url(link)
                entry_meta: dict = {}
                pub = getattr(entry, "published", None)
                if pub:
                    entry_meta["published"] = str(pub)
                summ = getattr(entry, "summary", None)
                if summ:
                    entry_meta["summary"] = _clean_summary(str(summ))
                if entry_meta:
                    metadata[link] = entry_meta
            logger.info("[rss_metadata] Got metadata for %d URLs from %s", len(metadata), source_key)
            return metadata
        except Exception as exc:
            logger.warning("[rss_metadata] Failed for %s: %s", source_key, exc)
            return {}

    # ---- HTML landing pages ------------------------------------------------

    async def fetch_html(self, source_key: str, do_download: bool = True) -> dict:
        """Scrape an HTML landing page and extract article links.

        Uses a headless browser (via ``get_browser``) to render JS-heavy
        pages, then ``parse_source_links`` to pull out article URLs.

        When *do_download* is ``False``, reads an existing HTML file from
        disk instead of re-downloading.

        Returns the same dict shape as ``fetch_rss``.
        """
        source = self.sources[source_key]
        url = source.get("url", "")
        filename = source.get("filename", source_key)

        try:
            if do_download:
                logger.info("[fetch_html] Fetching HTML from %s: %s", source_key, url)
                from config import FIREFOX_PROFILE_PATH, SLEEP_TIME
                browser = await get_browser(FIREFOX_PROFILE_PATH)

                # Scrape the landing page with source-specific parameters
                scrape_result = await scrape_url(
                    url=url,
                    title=filename,
                    browser_context=browser,
                    html_dir=DOWNLOAD_DIR,
                    click_xpath=source.get("click", ""),
                    scrolls=source.get("scroll", 0),
                    scroll_div=source.get("scroll_div", ""),
                    initial_sleep=source.get("initial_sleep", SLEEP_TIME),
                )

                if scrape_result.status not in ("success", "no_content"):
                    return {
                        "source": source_key,
                        "results": [],
                        "status": scrape_result.status,
                        "metadata": {"html_path": scrape_result.html_path},
                    }

                html_path = scrape_result.html_path
            else:
                logger.info("[fetch_html] Using existing HTML for %s", source_key)
                html_path = os.path.join(DOWNLOAD_DIR, f"{filename}.html")

            # Extract links from the saved HTML
            links = parse_source_links(html_path, source)

            # Tag each link with source
            for link in links:
                link["source"] = source_key

            # Enrich with RSS metadata if source has an rss field
            if source.get("rss"):
                rss_meta = await self._fetch_rss_metadata(source_key)
                enriched = 0
                for link in links:
                    meta = rss_meta.get(link["url"])
                    if meta:
                        link.update(meta)
                        enriched += 1
                if enriched:
                    logger.info("[fetch_html] Enriched %d/%d links with RSS metadata for %s",
                                enriched, len(links), source_key)

            logger.info("[fetch_html] HTML fetch successful for %s: %d articles", source_key, len(links))
            return {
                "source": source_key,
                "results": links,
                "status": "success",
                "metadata": {
                    "html_path": html_path,
                    "link_count": len(links),
                },
            }

        except Exception as exc:
            logger.error("HTML fetch failed for %s: %s", source_key, exc)
            return {
                "source": source_key,
                "results": [],
                "status": "error",
                "metadata": {"error": str(exc)},
            }

    # ---- REST API ----------------------------------------------------------

    async def fetch_api(self, source_key: str) -> dict:
        """Call a REST API source (dispatches by function_name).

        Currently supports ``fn_extract_newsapi``.

        Returns the same dict shape as ``fetch_rss``.
        """
        source = self.sources[source_key]
        fn_name = source.get("function_name", "")

        fn_map = {
            "fn_extract_newsapi": self.extract_newsapi,
        }

        handler = fn_map.get(fn_name)
        if handler is None:
            return {
                "source": source_key,
                "results": [],
                "status": "error",
                "metadata": {"error": f"Unknown function: {fn_name}"},
            }

        return handler()

    # ---- fetch_all ---------------------------------------------------------

    async def fetch_all(self, do_download: bool = True) -> list:
        """Fetch all sources concurrently, dispatching by type.

        Parameters
        ----------
        do_download:
            When ``False``, HTML sources use existing files on disk
            instead of re-downloading.

        Returns a list of result dicts (one per source).
        """
        logger.info("[fetch_all] Starting fetch_all for %d sources", len(self.sources))

        async def _fetch_one(key: str) -> dict:
            async with self._semaphore:
                source_type = self.sources[key].get("type", "html")
                if source_type == "rss":
                    return await self.fetch_rss(key)
                elif source_type == "rest":
                    return await self.fetch_api(key)
                else:
                    return await self.fetch_html(key, do_download=do_download)

        tasks = [asyncio.create_task(_fetch_one(k)) for k in self.sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final: list = []
        for key, result in zip(self.sources, results):
            if isinstance(result, BaseException):
                logger.error("Fetch failed for %s: %s", key, result)
                final.append({
                    "source": key,
                    "results": [],
                    "status": "error",
                    "metadata": {"error": str(result)},
                })
            else:
                final.append(result)

        success_count = sum(1 for r in final if r.get("status") == "success")
        error_count = sum(1 for r in final if r.get("status") == "error")
        total_articles = sum(len(r.get("results", [])) for r in final if r.get("status") == "success")
        logger.info(
            "[fetch_all] fetch_all completed: %d successful, %d failed, %d total articles",
            success_count, error_count, total_articles,
        )

        return final

    # ---- NewsAPI -----------------------------------------------------------

    def extract_newsapi(self) -> dict:
        """Query the NewsAPI for recent AI articles.

        Requires ``NEWSAPI_API_KEY`` in the environment.

        Returns the same dict shape as ``fetch_rss``.
        """
        api_key = os.environ.get("NEWSAPI_API_KEY", "")
        if not api_key:
            logger.warning("NEWSAPI_API_KEY not set; skipping NewsAPI")
            return {
                "source": "NewsAPI",
                "results": [],
                "status": "error",
                "metadata": {"error": "NEWSAPI_API_KEY not set"},
            }

        yesterday = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        params = {
            "q": "artificial intelligence",
            "from": yesterday,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": api_key,
        }

        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params=params,
                timeout=SHORT_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

            entries: List[dict] = []
            for article in data.get("articles", []):
                title = (article.get("title") or "").strip()
                url = (article.get("url") or "").strip()
                if not title or not url:
                    continue
                entries.append({
                    "title": title,
                    "url": clean_url(url),
                    "source": "NewsAPI",
                    "published": article.get("publishedAt", ""),
                    "summary": _clean_summary(article.get("description", "")),
                })

            return {
                "source": "NewsAPI",
                "results": entries,
                "status": "success",
                "metadata": {"total_results": data.get("totalResults", 0)},
            }

        except Exception as exc:
            logger.error("NewsAPI request failed: %s", exc)
            return {
                "source": "NewsAPI",
                "results": [],
                "status": "error",
                "metadata": {"error": str(exc)},
            }
