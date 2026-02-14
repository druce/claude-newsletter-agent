"""Web scraping utilities — sanitization, URL cleaning, HTML extraction, rate limiting."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import tldextract
import trafilatura
from bs4 import BeautifulSoup
from camoufox.async_api import AsyncCamoufox

from config import (
    DEFAULT_CONCURRENCY,
    DOMAIN_DAILY_CAP,
    DOMAIN_RATE_LIMIT,
    PAGES_DIR,
    SHORT_REQUEST_TIMEOUT,
    TEXT_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces non-alphanumeric characters (except hyphens, underscores, dots)
    with underscores, strips leading/trailing underscores, and truncates to
    200 characters.  Returns ``"unnamed"`` for empty input.
    """
    if not name or not name.strip():
        return "unnamed"

    # Replace unsafe characters with underscores
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace any remaining non-word characters (except hyphens and dots)
    name = re.sub(r'[^\w\-_.]', '_', name)
    # Strip leading/trailing underscores
    name = name.strip('_')
    # Truncate to 200 characters
    name = name[:200]
    # If after stripping we ended up empty, return a default
    return name or "unnamed"


def clean_url(url: str) -> str:
    """Clean a URL by removing tracking parameters (utm_*) and fragments.

    Preserves meaningful query parameters (e.g. ``?q=...``).
    """
    parsed = urlparse(url)

    # Filter out utm_* tracking params
    params = parse_qs(parsed.query, keep_blank_values=True)
    filtered = {k: v for k, v in params.items() if not k.startswith("utm_")}

    # Rebuild query string (doseq=True to handle multi-value params)
    clean_query = urlencode(filtered, doseq=True)

    # Reconstruct without fragment
    cleaned = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        clean_query,
        "",  # no fragment
    ))
    return cleaned


def normalize_html(html_path: str) -> str:
    """Extract text content from an HTML file using trafilatura.

    Returns the extracted text, or an empty string if the file is missing
    or extraction fails.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as exc:
        logger.error("Could not read %s: %s", html_path, exc)
        return ""

    try:
        text = trafilatura.extract(html_content)
        return text.strip() if text else ""
    except Exception as exc:
        logger.warning("trafilatura extraction failed for %s: %s", html_path, exc)
        return ""


# ---------------------------------------------------------------------------
# ScrapeResult data class
# ---------------------------------------------------------------------------

@dataclass
class ScrapeResult:
    """Result of a single URL scrape operation."""
    html_path: str = ""
    text_path: str = ""
    final_url: str = ""
    last_updated: str = ""
    status: str = "pending"


# ---------------------------------------------------------------------------
# Per-domain state for rate limiting
# ---------------------------------------------------------------------------

@dataclass
class _DomainState:
    """Internal per-domain tracking for RateLimiter."""
    last_request: float = 0.0
    request_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Async per-domain rate limiter with optional daily cap.

    Parameters
    ----------
    rate_limit_seconds:
        Minimum interval between requests to the same domain.
    daily_cap:
        Maximum number of requests per domain per session.
        ``0`` means unlimited.
    """

    def __init__(
        self,
        rate_limit_seconds: float = 5.0,
        daily_cap: int = DOMAIN_DAILY_CAP,
    ) -> None:
        self._delay = rate_limit_seconds
        self._daily_cap = daily_cap
        self._domains: Dict[str, _DomainState] = defaultdict(_DomainState)

    async def try_acquire(self, domain: str) -> Tuple[bool, float]:
        """Atomically check rate limit and daily cap for *domain*.

        Returns
        -------
        (can_proceed, wait_time_seconds)
            ``can_proceed`` is ``True`` when the request may go ahead
            immediately.  ``wait_time_seconds`` is the remaining cooldown
            (``0.0`` when allowed).
        """
        state = self._domains[domain]
        async with state.lock:
            # Check daily cap first
            if self._daily_cap > 0 and state.request_count >= self._daily_cap:
                return False, 0.0

            # Check timing
            now = monotonic()
            elapsed = now - state.last_request

            if state.last_request == 0.0 or elapsed >= self._delay:
                # Allowed — mark this request
                state.last_request = now
                state.request_count += 1
                return True, 0.0

            # Too soon
            return False, self._delay - elapsed


# ---------------------------------------------------------------------------
# Browser cache (module-level singleton)
# ---------------------------------------------------------------------------

_browser_cache: Dict[str, Any] = {}
_browser_lock = asyncio.Lock()


def _reset_browser_cache() -> None:
    """Reset browser cache (for testing)."""
    _browser_cache.clear()


# ---------------------------------------------------------------------------
# Browser management
# ---------------------------------------------------------------------------

# Patterns that indicate the page was blocked by bot detection
_BOT_BLOCK_PATTERNS = re.compile(
    r"Access Denied|Bot Detection|Captcha|Just a moment",
    re.IGNORECASE,
)


async def get_browser(user_data_dir: str) -> Any:
    """Get or create a cached Camoufox browser context.

    Uses ``AsyncCamoufox`` with ``persistent_context=True`` so cookies and
    local-storage survive across runs.  The context is cached at module level
    so only one browser is launched per ``user_data_dir``.

    Parameters
    ----------
    user_data_dir:
        Path to the Firefox profile directory for persistent context.

    Returns
    -------
    BrowserContext
        A Playwright-compatible browser context backed by Camoufox.
    """
    global _browser_lock

    # Fast path: already cached
    if user_data_dir in _browser_cache:
        return _browser_cache[user_data_dir]

    async with _browser_lock:
        # Double-check after acquiring lock
        if user_data_dir in _browser_cache:
            return _browser_cache[user_data_dir]

        cm = AsyncCamoufox(
            persistent_context=True,
            user_data_dir=user_data_dir,
            headless=True,
        )
        context = await cm.__aenter__()
        _browser_cache[user_data_dir] = context
        return context


# ---------------------------------------------------------------------------
# Single-URL scraping
# ---------------------------------------------------------------------------

async def scrape_url(
    url: str,
    title: str,
    browser_context: Any,
    html_dir: str = PAGES_DIR,
    text_dir: str = TEXT_DIR,
) -> ScrapeResult:
    """Scrape a single URL and save HTML + extracted text to disk.

    Parameters
    ----------
    url:
        The URL to navigate to.
    title:
        Human-readable title; used to derive the on-disk filename.
    browser_context:
        A Playwright ``BrowserContext`` (from ``get_browser``).
    html_dir:
        Directory for raw HTML files.
    text_dir:
        Directory for extracted plain-text files.

    Returns
    -------
    ScrapeResult
        Contains paths, final URL, and a status string
        (``"success"``, ``"no_content"``, ``"blocked"``, or ``"error"``).
    """
    page = None
    try:
        page = await browser_context.new_page()

        # Navigate
        await page.goto(url, timeout=SHORT_REQUEST_TIMEOUT * 1000, wait_until="domcontentloaded")

        # Get page content
        html_source = await page.content()
        final_url = page.url

        # Detect bot-blocking pages via document title
        try:
            doc_title = await page.evaluate("document.title")
        except Exception:
            doc_title = ""

        if doc_title and _BOT_BLOCK_PATTERNS.search(doc_title):
            return ScrapeResult(
                final_url=final_url,
                status="blocked",
            )

        # Ensure output directories exist
        os.makedirs(html_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)

        safe_name = sanitize_filename(title)
        html_path = os.path.join(html_dir, f"{safe_name}.html")
        text_path = os.path.join(text_dir, f"{safe_name}.txt")

        # Save raw HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_source)

        # Extract text via normalize_html (reads the file we just wrote)
        text = normalize_html(html_path)

        # Save extracted text
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Try to extract last_updated from meta tags or JSON-LD
        last_updated = _extract_last_updated(html_source)

        if not text.strip():
            status = "no_content"
        else:
            status = "success"

        return ScrapeResult(
            html_path=html_path,
            text_path=text_path,
            final_url=final_url,
            last_updated=last_updated,
            status=status,
        )

    except Exception as exc:
        logger.error("Error scraping %s: %s", url, exc)
        return ScrapeResult(final_url=url, status="error")
    finally:
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass


def _extract_last_updated(html_source: str) -> str:
    """Best-effort extraction of publish/update date from HTML.

    Checks meta tags and JSON-LD ``datePublished`` fields.
    Returns ISO date string or empty string.
    """
    try:
        soup = BeautifulSoup(html_source, "html.parser")

        # Meta tag selectors (in priority order)
        meta_selectors = [
            ("property", "article:published_time"),
            ("property", "og:published_time"),
            ("property", "article:modified_time"),
            ("property", "og:updated_time"),
            ("name", "pubdate"),
            ("name", "publish_date"),
            ("name", "Last-Modified"),
            ("name", "lastmod"),
        ]
        for attr, val in meta_selectors:
            tag = soup.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                return tag["content"]

        # JSON-LD datePublished
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get("datePublished"):
                    return data["datePublished"]
            except (json.JSONDecodeError, TypeError):
                continue

    except Exception:
        pass

    return ""


# ---------------------------------------------------------------------------
# Concurrent multi-URL scraping
# ---------------------------------------------------------------------------

async def scrape_urls_concurrent(
    urls: List[Dict[str, str]],
    browser_context: Any,
    concurrency: int = DEFAULT_CONCURRENCY,
    rate_limit_seconds: float = DOMAIN_RATE_LIMIT,
    html_dir: str = PAGES_DIR,
    text_dir: str = TEXT_DIR,
) -> List[ScrapeResult]:
    """Scrape multiple URLs concurrently with per-domain rate limiting.

    Parameters
    ----------
    urls:
        List of dicts, each with ``"url"`` and ``"title"`` keys.
    browser_context:
        A Playwright ``BrowserContext`` (from ``get_browser``).
    concurrency:
        Maximum number of pages open simultaneously.
    rate_limit_seconds:
        Minimum interval between requests to the same domain.
    html_dir:
        Directory for raw HTML files.
    text_dir:
        Directory for extracted plain-text files.

    Returns
    -------
    list[ScrapeResult]
        One result per input URL, in the same order.
    """
    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = RateLimiter(rate_limit_seconds=rate_limit_seconds)

    async def _scrape_one(item: Dict[str, str]) -> ScrapeResult:
        async with semaphore:
            # Extract domain for rate limiting
            extracted = tldextract.extract(item["url"])
            domain = f"{extracted.domain}.{extracted.suffix}"

            # Rate-limit: wait if needed
            can_proceed, wait_time = await rate_limiter.try_acquire(domain)
            if not can_proceed and wait_time > 0:
                await asyncio.sleep(wait_time)
                # Try again after waiting
                await rate_limiter.try_acquire(domain)

            return await scrape_url(
                url=item["url"],
                title=item["title"],
                browser_context=browser_context,
                html_dir=html_dir,
                text_dir=text_dir,
            )

    tasks = [asyncio.create_task(_scrape_one(item)) for item in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert any exceptions to error ScrapeResults
    final: List[ScrapeResult] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Task %d raised %s: %s", i, type(result).__name__, result)
            final.append(ScrapeResult(status="error"))
        else:
            final.append(result)

    return final
