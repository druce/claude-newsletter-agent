"""Web scraping utilities — sanitization, URL cleaning, HTML extraction, rate limiting."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from time import monotonic
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import trafilatura
import tldextract

from config import DOMAIN_DAILY_CAP

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
