"""Web scraping utilities — sanitization, URL cleaning, HTML extraction, rate limiting."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import tldextract
import trafilatura
from bs4 import BeautifulSoup
from playwright.async_api import BrowserContext, async_playwright
from playwright_stealth import Stealth

from config import (
    DEFAULT_CONCURRENCY,
    DOMAIN_DAILY_CAP,
    DOMAIN_RATE_LIMIT,
    PAGES_DIR,
    SHORT_REQUEST_TIMEOUT,
    SLEEP_TIME,
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


def is_valid_html(content: str, min_length: int = 1000) -> bool:
    """Check whether *content* looks like legitimate HTML rather than binary/compressed data.

    Returns ``False`` when any of these conditions hold:

    * Content shorter than *min_length* bytes.
    * No ``<!doctype`` or ``<html`` marker in the first 500 characters.
    * More than 30 % non-ASCII characters in the first 10 000 characters
      (corrupt/binary payloads typically exceed 70 %).
    """
    if len(content) < min_length:
        return False

    head = content[:500].lower()
    if "<!doctype" not in head and "<html" not in head:
        return False

    sample = content[:10_000]
    non_ascii = sum(1 for ch in sample if ord(ch) > 127)
    if non_ascii / len(sample) > 0.30:
        return False

    return True


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
_playwright_instance: Any = None
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


async def _create_browser_context(p: Any, user_data_dir: str) -> BrowserContext:
    """Create a new Playwright Firefox persistent context with stealth settings."""
    viewport = random.choice([
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1280, "height": 720},
    ])
    device_scale_factor = random.choice([1, 1.25, 1.5, 1.75, 2])
    color_scheme = random.choice(["light", "dark", "no-preference"])
    timezone_id = random.choice([
        "America/New_York", "Europe/London", "Europe/Paris",
        "Asia/Tokyo", "Australia/Sydney", "America/Los_Angeles",
    ])
    locale = random.choice(["en-US", "en-GB"])
    extra_http_headers = {
        "Accept-Language": f"{locale.split('-')[0]},{locale};q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "DNT": "1" if random.choice([True, False]) else "0",
    }

    context = await p.firefox.launch_persistent_context(
        user_data_dir=user_data_dir,
        headless=True,
        viewport=viewport,
        device_scale_factor=device_scale_factor,
        timezone_id=timezone_id,
        color_scheme=color_scheme,
        extra_http_headers=extra_http_headers,
        ignore_default_args=["--enable-automation"],
        args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
        firefox_user_prefs={
            "browser.link.open_newwindow": 3,
            "browser.link.open_newwindow.restriction": 0,
            "browser.tabs.loadInBackground": False,
        },
        user_agent=(
            "Mozilla/5.0 (Macintosh; ARM Mac OS X 14.4; rv:125.0) "
            "Gecko/20100101 Firefox/125.0"
        ),
        accept_downloads=True,
    )

    # Apply playwright-stealth to the context
    stealth = Stealth()
    page = await context.new_page()
    await stealth.apply_stealth_async(page)
    await page.close()

    return context


async def get_browser(user_data_dir: str) -> BrowserContext:
    """Get or create a cached Playwright Firefox browser context.

    Uses ``firefox.launch_persistent_context`` so cookies and local-storage
    survive across runs.  The context is cached at module level so only one
    browser is launched per ``user_data_dir``.

    Parameters
    ----------
    user_data_dir:
        Path to the Firefox profile directory for persistent context.

    Returns
    -------
    BrowserContext
        A Playwright browser context with stealth settings applied.
    """
    global _browser_lock, _playwright_instance

    # Fast path: already cached
    if user_data_dir in _browser_cache:
        return _browser_cache[user_data_dir]

    async with _browser_lock:
        # Double-check after acquiring lock
        if user_data_dir in _browser_cache:
            return _browser_cache[user_data_dir]

        if _playwright_instance is None:
            _playwright_instance = await async_playwright().start()

        context = await _create_browser_context(_playwright_instance, user_data_dir)
        _browser_cache[user_data_dir] = context
        return context


async def enable_fast_mode(page: Any) -> None:
    """Block heavy resources (images, media, fonts) to speed up page loads."""
    blocked_types = {"image", "media", "font"}

    async def _route_handler(route: Any) -> None:
        try:
            if route.request.resource_type in blocked_types:
                await route.abort()
            else:
                await route.continue_()
        except Exception:
            pass

    await page.route("**/*", _route_handler)


# ---------------------------------------------------------------------------
# Single-URL scraping
# ---------------------------------------------------------------------------

async def scrape_url(
    url: str,
    title: str,
    browser_context: Any,
    html_dir: str = PAGES_DIR,
    text_dir: str = TEXT_DIR,
    click_xpath: str = "",
    scrolls: int = 0,
    scroll_div: str = "",
    initial_sleep: float = SLEEP_TIME,
    save_text: bool = True,
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
    click_xpath:
        Optional XPath to click before capturing content.
    scrolls:
        Number of times to scroll to bottom (for infinite-scroll pages).
    scroll_div:
        CSS selector for a scrollable container (scrolls window if empty).
    initial_sleep:
        Seconds to wait after page load before interacting.
    save_text:
        Whether to extract and save a plain-text version alongside HTML.
        Set to ``False`` for landing-page scrapes where text isn't useful.

    Returns
    -------
    ScrapeResult
        Contains paths, final URL, and a status string
        (``"success"``, ``"no_content"``, ``"blocked"``, ``"corrupt"``, or ``"error"``).
    """
    page = None
    try:
        page = await browser_context.new_page()

        # Block heavy resources for faster page loads
        await enable_fast_mode(page)

        # Navigate
        await page.goto(url, timeout=SHORT_REQUEST_TIMEOUT * 1000, wait_until="domcontentloaded")

        # Wait after load, then perform optional interactions
        sleep_time = initial_sleep + random.uniform(1, 3)
        await asyncio.sleep(sleep_time)

        if click_xpath:
            await asyncio.sleep(initial_sleep + random.uniform(1, 3))
            await page.wait_for_selector(f"xpath={click_xpath}")
            await page.click(f"xpath={click_xpath}")

        for _ in range(scrolls):
            await asyncio.sleep(random.uniform(1, 3))
            if scroll_div:
                await page.evaluate("""
                    const el = document.querySelector('%s');
                    if (el) { el.scrollTop = el.scrollHeight; }
                    else { window.scrollTo(0, document.body.scrollHeight); }
                """ % scroll_div)
            else:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")

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

        # Reject binary/compressed content before writing to disk
        if not is_valid_html(html_source):
            logger.warning("Corrupt/binary content from %s (len=%d)", url, len(html_source))
            return ScrapeResult(final_url=final_url, status="corrupt")

        # Ensure output directories exist
        os.makedirs(html_dir, exist_ok=True)

        safe_name = sanitize_filename(title)
        html_path = os.path.join(html_dir, f"{safe_name}.html")

        # Save raw HTML
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_source)

        # Try to extract last_updated from meta tags or JSON-LD
        last_updated = _extract_last_updated(html_source)

        text_path = ""
        status = "success"

        if save_text:
            os.makedirs(text_dir, exist_ok=True)
            text_path = os.path.join(text_dir, f"{safe_name}.txt")

            # Extract text via normalize_html (reads the file we just wrote)
            text = normalize_html(html_path)

            # Save extracted text
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            if not text.strip():
                status = "no_content"

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
                return str(tag["content"])

        # JSON-LD datePublished
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                if not script.string:
                    continue
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
    final: list[ScrapeResult] = []
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            logger.error("Task %d raised %s: %s", i, type(r).__name__, r)
            final.append(ScrapeResult(status="error"))
            continue
        final.append(r)  # type: ignore[arg-type]  # asyncio.gather narrowing

    return final
