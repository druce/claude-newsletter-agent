# Phase 3: Library Modules — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement five `lib/` modules (scrape, fetch, dedupe, rating, cluster) that provide the data-processing backbone for the newsletter agent.

**Architecture:** Five independent library modules under `lib/`, each a clean rewrite of its OpenAIAgentsSDK original. All share config.py constants, db.py models, llm.py agents, and prompts.py templates. Camoufox replaces playwright-stealth for browser automation. OpenAI text-embedding-3-large for embeddings. Bradley-Terry battles enabled.

**Tech Stack:** Python 3.11, Camoufox, trafilatura, feedparser, aiohttp, OpenAI embeddings, HDBSCAN, UMAP, Optuna, choix, tiktoken, pytest-asyncio

**Reference files:**
- Design doc: `docs/plans/2026-02-13-phase3-library-modules-design.md`
- Original source: `/Users/drucev/projects/OpenAIAgentsSDK/` (fetch.py, scrape.py, do_dedupe.py, do_rating.py, do_cluster.py)
- Prompts: `prompts.py` (already committed)

---

## Task 0: Config & Dependency Updates

**Files:**
- Modify: `config.py`
- Modify: `requirements.txt`
- Copy: `sources.yaml` from `/Users/drucev/projects/OpenAIAgentsSDK/sources.yaml`
- Copy: `umap_reducer.pkl` from `/Users/drucev/projects/OpenAIAgentsSDK/umap_reducer.pkl`

**Step 1: Add new constants to config.py**

Add after line 27 (`FIREFOX_PROFILE_PATH`):

```python
# --- Scraping ---
DOMAIN_DAILY_CAP = 50
SLEEP_TIME = 5

# --- Embeddings ---
EMBEDDING_MODEL = "text-embedding-3-large"
SIMILARITY_THRESHOLD = 0.925
MAX_EMBED_TOKENS = 8192

# --- Clustering ---
MIN_COMPONENTS = 20
RANDOM_STATE = 42
OPTUNA_TRIALS = 50
```

**Step 2: Update requirements.txt**

Replace `playwright-stealth` with `camoufox`. Add `beautifulsoup4`, `tiktoken`, `optuna`.

```
# Core
python-dotenv
pydantic>=2.0

# LLM
anthropic
openai
google-genai
tenacity

# Data processing
numpy
pandas
scipy
hdbscan
umap-learn
choix

# Web scraping
playwright>=1.43.0
camoufox
aiohttp
aiofiles
feedparser
trafilatura
beautifulsoup4

# Utilities
tldextract
pyyaml
tiktoken
optuna

# Testing
pytest
pytest-asyncio
pytest-mock
pytest-cov
pytest-timeout
```

**Step 3: Copy sources.yaml and umap_reducer.pkl**

```bash
cp /Users/drucev/projects/OpenAIAgentsSDK/sources.yaml sources.yaml
cp /Users/drucev/projects/OpenAIAgentsSDK/umap_reducer.pkl umap_reducer.pkl
```

**Step 4: Install new dependencies**

```bash
source .venv/bin/activate && pip install -r requirements.txt
```

**Step 5: Verify config imports**

```bash
python -c "from config import DOMAIN_DAILY_CAP, SLEEP_TIME, EMBEDDING_MODEL, SIMILARITY_THRESHOLD, MAX_EMBED_TOKENS, MIN_COMPONENTS, RANDOM_STATE, OPTUNA_TRIALS; print('OK')"
```

**Step 6: Commit**

```bash
git add config.py requirements.txt sources.yaml umap_reducer.pkl
git commit -m "chore: add Phase 3 config constants, dependencies, sources.yaml, UMAP reducer"
```

---

## Task 1: lib/scrape.py — Utilities & RateLimiter

**Files:**
- Create: `lib/scrape.py`
- Create: `tests/test_scrape.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/scrape.py` (lines 1-200 for utilities, RateLimiter)

### Step 1: Write failing tests for utility functions and RateLimiter

```python
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
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_scrape.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'lib.scrape'`

### Step 3: Implement utility functions, ScrapeResult, and RateLimiter

Create `lib/scrape.py`. Port from `/Users/drucev/projects/OpenAIAgentsSDK/scrape.py`:
- `sanitize_filename` (lines ~50-70) — regex replace non-alphanum with `_`, truncate to 200 chars, default "unnamed"
- `clean_url` (lines ~75-95) — parse with urllib, strip utm_* params and fragments
- `normalize_html` (lines ~100-130) — trafilatura.extract() with fallback to empty string
- `ScrapeResult` dataclass with defaults: `html_path=""`, `text_path=""`, `final_url=""`, `last_updated=""`, `status="pending"`
- `RateLimiter` class with `asyncio.Lock` per domain, `try_acquire(domain) → (bool, float)`, daily request counter per domain

Key implementation details:
```python
from __future__ import annotations
import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import trafilatura
import tldextract

from config import (
    PAGES_DIR, TEXT_DIR, DOMAIN_RATE_LIMIT,
    DOMAIN_DAILY_CAP, IGNORE_LIST, MIN_TITLE_LEN,
)

logger = logging.getLogger(__name__)
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_scrape.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/scrape.py tests/test_scrape.py
git commit -m "feat(lib): add scrape.py utilities, ScrapeResult, RateLimiter"
```

---

## Task 2: lib/scrape.py — Browser & Scraping Functions

**Files:**
- Modify: `lib/scrape.py`
- Modify: `tests/test_scrape.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/scrape.py` (lines 200-600 for get_browser, scrape_url; lines 600-900 for scrape_urls_concurrent)

### Step 1: Write failing tests for browser and scraping functions

Add to `tests/test_scrape.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch


class TestGetBrowser:
    @pytest.mark.asyncio
    @patch("lib.scrape.AsyncCamoufox")
    async def test_returns_browser_context(self, mock_camoufox):
        from lib.scrape import get_browser
        mock_context = AsyncMock()
        mock_camoufox.return_value.__aenter__ = AsyncMock(return_value=mock_context)
        ctx = await get_browser("/tmp/test_profile")
        assert ctx is not None

    @pytest.mark.asyncio
    @patch("lib.scrape.AsyncCamoufox")
    async def test_caches_browser_context(self, mock_camoufox):
        from lib.scrape import get_browser, _reset_browser_cache
        _reset_browser_cache()
        mock_context = AsyncMock()
        mock_camoufox.return_value.__aenter__ = AsyncMock(return_value=mock_context)
        ctx1 = await get_browser("/tmp/test_profile")
        ctx2 = await get_browser("/tmp/test_profile")
        # Should only create one browser
        assert mock_camoufox.call_count == 1


class TestScrapeUrl:
    @pytest.mark.asyncio
    async def test_saves_html_and_text(self, tmp_path):
        from lib.scrape import scrape_url, ScrapeResult
        mock_page = AsyncMock()
        mock_page.content.return_value = "<html><body><p>AI article content here.</p></body></html>"
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
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_scrape.py::TestGetBrowser -v
pytest tests/test_scrape.py::TestScrapeUrl -v
pytest tests/test_scrape.py::TestScrapeUrlsConcurrent -v
```
Expected: FAIL

### Step 3: Implement browser and scraping functions

Add to `lib/scrape.py`:
- `_browser_context_cache` dict and `_browser_lock` asyncio.Lock at module level
- `_reset_browser_cache()` — for testing
- `async get_browser(user_data_dir)` — uses `AsyncCamoufox(persistent_context=True, user_data_dir=...)`, caches result
- `async scrape_url(url, title, browser_context, html_dir=PAGES_DIR, text_dir=TEXT_DIR)`:
  1. Create new page, set random viewport
  2. Navigate with timeout (SHORT_REQUEST_TIMEOUT * 1000 ms)
  3. Wait for load, extract content
  4. Save HTML to `html_dir/{sanitize_filename(title)}.html`
  5. Run `normalize_html()` on saved file, save text to `text_dir/{sanitize_filename(title)}.txt`
  6. Extract `last_updated` from meta tags, JSON-LD, or `document.lastModified`
  7. Get canonical URL from `<link rel="canonical">`
  8. Return `ScrapeResult`
  9. On any exception: log, return ScrapeResult with `status="error"`
- `async scrape_urls_concurrent(urls, browser_context, concurrency=DEFAULT_CONCURRENCY, rate_limit_seconds=DOMAIN_RATE_LIMIT)`:
  1. Create semaphore and RateLimiter
  2. For each url dict, create async task that acquires semaphore, checks rate limiter, calls scrape_url
  3. Gather all tasks, return list of ScrapeResult

Port from original scrape.py lines 200-900, adapting:
- Replace `playwright.async_api` with `camoufox.async_api.AsyncCamoufox`
- Replace tuple returns with `ScrapeResult`
- Add bot-block detection: if page title contains "Access Denied", "Bot Detection", "Captcha", set status="blocked"

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_scrape.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/scrape.py tests/test_scrape.py
git commit -m "feat(lib): add scrape.py browser automation with Camoufox"
```

---

## Task 3: lib/fetch.py — Source Processor

**Files:**
- Create: `lib/fetch.py`
- Create: `tests/test_fetch.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/fetch.py` (587 lines)

### Step 1: Write failing tests

```python
# tests/test_fetch.py
"""Tests for lib/fetch.py — source fetching and link extraction."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestParseSourceLinks:
    def test_extracts_matching_links(self, tmp_path):
        from lib.fetch import parse_source_links
        html_file = tmp_path / "test.html"
        html_file.write_text("""
        <html><body>
            <a href="https://example.com/ai-article-1">AI Article 1</a>
            <a href="https://example.com/sports-news">Sports</a>
            <a href="https://example.com/ai-article-2">AI Article Two</a>
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
            <a href="https://example.com/good-article">Good Article Title Here</a>
            <a href="https://example.com/video/bad">Bad Video Link Here</a>
        </body></html>
        """)
        config = {
            "exclude": [r"/video/"],
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
    @patch("lib.fetch.feedparser.parse")
    async def test_parses_rss_entries(self, mock_parse):
        from lib.fetch import Fetcher
        mock_parse.return_value = MagicMock(
            entries=[
                MagicMock(
                    title="AI Breakthrough in Healthcare",
                    link="https://example.com/ai-health",
                    get=lambda k, d=None: {"published": "2026-02-13", "summary": "Summary text"}.get(k, d),
                ),
            ],
            bozo=False,
        )
        async with Fetcher.__new__(Fetcher) as f:
            pass
        # Test the RSS parsing directly
        fetcher = Fetcher.__new__(Fetcher)
        fetcher._session = None
        results = await fetcher.fetch_rss({"rss": "https://example.com/feed.xml"}, "TestSource")
        assert len(results) == 1
        assert results[0]["title"] == "AI Breakthrough in Healthcare"
        assert results[0]["source"] == "TestSource"


class TestFetcherFetchAll:
    @pytest.mark.asyncio
    @patch("lib.fetch.Fetcher.fetch_rss")
    @patch("lib.fetch.Fetcher.fetch_html")
    async def test_dispatches_by_type(self, mock_html, mock_rss):
        from lib.fetch import Fetcher
        mock_rss.return_value = [{"source": "RSS", "title": "T", "url": "u"}]
        mock_html.return_value = [{"source": "HTML", "title": "T", "url": "u"}]

        fetcher = Fetcher.__new__(Fetcher)
        fetcher.sources = {
            "RSSSource": {"type": "rss", "rss": "https://feed.xml"},
            "HTMLSource": {"type": "html", "url": "https://page.com"},
        }
        fetcher._session = AsyncMock()
        fetcher._browser_context = AsyncMock()
        fetcher._semaphore = asyncio.Semaphore(8)

        results = await fetcher.fetch_all()
        assert "RSSSource" in results
        assert "HTMLSource" in results
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_fetch.py -v
```
Expected: FAIL

### Step 3: Implement lib/fetch.py

Create `lib/fetch.py`. Port from `/Users/drucev/projects/OpenAIAgentsSDK/fetch.py`:

Key structure:
```python
from __future__ import annotations
import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

import aiohttp
import feedparser
import yaml
from bs4 import BeautifulSoup

from config import DOWNLOAD_DIR, MIN_TITLE_LEN, DEFAULT_CONCURRENCY
from lib.scrape import scrape_url, get_browser, clean_url

logger = logging.getLogger(__name__)


def parse_source_links(html_path: str, source_config: dict) -> List[dict]:
    """Extract headline links from a saved HTML file using include/exclude regex."""
    ...

def get_og_tags(html_path: str) -> dict:
    """Extract OpenGraph meta tags from HTML file."""
    ...

class Fetcher:
    """Async source processor. Loads sources.yaml and fetches headlines."""

    def __init__(self, sources_file: str = "sources.yaml", max_concurrent: int = DEFAULT_CONCURRENCY):
        with open(sources_file) as f:
            self.sources = yaml.safe_load(f)
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None
        self._browser_context = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def fetch_rss(self, source_config: dict, source_name: str) -> List[dict]: ...
    async def fetch_html(self, source_name: str, source_config: dict) -> List[dict]: ...
    async def fetch_api(self, source_name: str, source_config: dict) -> List[dict]: ...
    async def fetch_all(self) -> Dict[str, List[dict]]: ...
```

Port logic from original fetch.py:
- `fetch_rss`: feedparser.parse(url), extract title/link/published/summary per entry
- `fetch_html`: use scrape.py to download landing page, then parse_source_links to extract links
- `fetch_api`: NewsAPI integration via requests/aiohttp
- `fetch_all`: concurrent dispatch via asyncio.gather with semaphore
- `parse_source_links`: BeautifulSoup link extraction with include/exclude regex, min title length

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_fetch.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/fetch.py tests/test_fetch.py
git commit -m "feat(lib): add fetch.py source processor with RSS/HTML/API support"
```

---

## Task 4: lib/dedupe.py — Duplicate Detection

**Files:**
- Create: `lib/dedupe.py`
- Create: `tests/test_dedupe.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/do_dedupe.py` (lines 1-250 for dedup pipeline)

### Step 1: Write failing tests

```python
# tests/test_dedupe.py
"""Tests for lib/dedupe.py — embedding-based duplicate detection."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch


class TestCreateSimilarityMatrix:
    def test_identity_diagonal(self):
        from lib.dedupe import create_similarity_matrix
        embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        index = pd.Index([0, 1, 2])
        sim = create_similarity_matrix(embeddings, index)
        assert sim.shape == (3, 3)
        np.testing.assert_almost_equal(sim.iloc[0, 0], 1.0)
        np.testing.assert_almost_equal(sim.iloc[0, 1], 0.0)

    def test_high_similarity_detected(self):
        from lib.dedupe import create_similarity_matrix
        embeddings = np.array([[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0]])
        index = pd.Index([0, 1, 2])
        sim = create_similarity_matrix(embeddings, index)
        assert sim.iloc[0, 1] > 0.9


class TestFindDuplicatePairs:
    def test_finds_pairs_above_threshold(self):
        from lib.dedupe import find_duplicate_pairs
        sim_data = {0: {0: 1.0, 1: 0.96, 2: 0.3}, 1: {0: 0.96, 1: 1.0, 2: 0.2}, 2: {0: 0.3, 1: 0.2, 2: 1.0}}
        sim_df = pd.DataFrame(sim_data)
        pairs = find_duplicate_pairs(sim_df, threshold=0.925)
        assert len(pairs) == 1
        assert (0, 1) in pairs or (1, 0) in pairs

    def test_no_pairs_below_threshold(self):
        from lib.dedupe import find_duplicate_pairs
        sim_data = {0: {0: 1.0, 1: 0.5}, 1: {0: 0.5, 1: 1.0}}
        sim_df = pd.DataFrame(sim_data)
        pairs = find_duplicate_pairs(sim_df, threshold=0.925)
        assert len(pairs) == 0


class TestFilterDuplicates:
    def test_keeps_longer_article(self):
        from lib.dedupe import filter_duplicates
        df = pd.DataFrame({
            "title": ["Article A", "Article B", "Article C"],
            "content_length": [500, 1000, 200],
        })
        pairs = [(0, 1)]  # A and B are dupes
        result = filter_duplicates(df, pairs)
        assert len(result) == 2
        assert "Article B" in result["title"].values  # B has more content
        assert "Article A" not in result["title"].values

    def test_no_pairs_returns_unchanged(self):
        from lib.dedupe import filter_duplicates
        df = pd.DataFrame({"title": ["A", "B"], "content_length": [100, 200]})
        result = filter_duplicates(df, [])
        assert len(result) == 2


class TestReadAndTruncateFiles:
    def test_reads_text_files(self, tmp_path):
        from lib.dedupe import read_and_truncate_files
        text_file = tmp_path / "article.txt"
        text_file.write_text("This is a test article about artificial intelligence.")
        df = pd.DataFrame({"text_path": [str(text_file)]})
        result = read_and_truncate_files(df, max_tokens=100)
        assert "truncated_text" in result.columns
        assert "artificial intelligence" in result.iloc[0]["truncated_text"]

    def test_handles_missing_file(self, tmp_path):
        from lib.dedupe import read_and_truncate_files
        df = pd.DataFrame({"text_path": ["/nonexistent/file.txt"]})
        result = read_and_truncate_files(df, max_tokens=100)
        assert result.iloc[0]["truncated_text"] == ""


class TestGetEmbeddingsBatch:
    @pytest.mark.asyncio
    @patch("lib.dedupe.openai.AsyncOpenAI")
    async def test_returns_embeddings(self, mock_openai_cls):
        from lib.dedupe import get_embeddings_batch
        mock_client = AsyncMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        result = await get_embeddings_batch(["test text"], batch_size=10)
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]


class TestProcessDataframeWithFiltering:
    @pytest.mark.asyncio
    @patch("lib.dedupe.get_embeddings_batch")
    async def test_end_to_end_dedup(self, mock_embed):
        from lib.dedupe import process_dataframe_with_filtering
        # Two near-identical embeddings + one different
        mock_embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.999, 0.001, 0.0],
            [0.0, 1.0, 0.0],
        ]
        df = pd.DataFrame({
            "text_path": ["a.txt", "b.txt", "c.txt"],
            "content_length": [100, 200, 300],
            "truncated_text": ["text a", "text b", "text c"],
        })
        result = await process_dataframe_with_filtering(df, similarity_threshold=0.925)
        # Should drop one of the near-identical pair (keep longer)
        assert len(result) == 2
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_dedupe.py -v
```
Expected: FAIL

### Step 3: Implement lib/dedupe.py

Create `lib/dedupe.py`. Port from `/Users/drucev/projects/OpenAIAgentsSDK/do_dedupe.py` lines 1-250:

```python
from __future__ import annotations
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import openai
import pandas as pd
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL, SIMILARITY_THRESHOLD, MAX_EMBED_TOKENS

logger = logging.getLogger(__name__)


async def get_embeddings_batch(
    texts: List[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = 25,
) -> List[List[float]]:
    """Get embeddings from OpenAI in batches."""
    client = openai.AsyncOpenAI()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])
    return all_embeddings


def read_and_truncate_files(df: pd.DataFrame, max_tokens: int = MAX_EMBED_TOKENS) -> pd.DataFrame:
    """Read text files from text_path column, truncate to max_tokens."""
    enc = tiktoken.encoding_for_model("gpt-4")  # cl100k_base
    texts = []
    for path in df["text_path"]:
        try:
            with open(path) as f:
                text = f.read()
            tokens = enc.encode(text)[:max_tokens]
            texts.append(enc.decode(tokens))
        except (FileNotFoundError, OSError):
            texts.append("")
    result = df.copy()
    result["truncated_text"] = texts
    return result


def create_similarity_matrix(embeddings: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Compute pairwise cosine similarity matrix."""
    sim = cosine_similarity(embeddings)
    return pd.DataFrame(sim, index=index, columns=index)


def find_duplicate_pairs(
    similarity_df: pd.DataFrame, threshold: float = SIMILARITY_THRESHOLD
) -> List[Tuple[int, int]]:
    """Find pairs with similarity above threshold (upper triangle only)."""
    pairs = []
    n = len(similarity_df)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_df.iloc[i, j] > threshold:
                pairs.append((similarity_df.index[i], similarity_df.columns[j]))
    return pairs


def filter_duplicates(df: pd.DataFrame, pairs: List[Tuple]) -> pd.DataFrame:
    """For each duplicate pair, drop the article with less content."""
    to_drop = set()
    for idx_a, idx_b in pairs:
        if idx_a in to_drop or idx_b in to_drop:
            continue
        len_a = df.loc[idx_a, "content_length"] if idx_a in df.index else 0
        len_b = df.loc[idx_b, "content_length"] if idx_b in df.index else 0
        to_drop.add(idx_a if len_b >= len_a else idx_b)
    return df.drop(index=to_drop)


async def process_dataframe_with_filtering(
    df: pd.DataFrame, similarity_threshold: float = SIMILARITY_THRESHOLD
) -> pd.DataFrame:
    """Top-level dedup pipeline: embed → similarity → filter."""
    if len(df) < 2:
        return df
    texts = df.get("truncated_text", pd.Series([""] * len(df))).tolist()
    texts = [t if t else "empty" for t in texts]
    embeddings = await get_embeddings_batch(texts)
    emb_array = np.array(embeddings)
    sim_df = create_similarity_matrix(emb_array, df.index)
    pairs = find_duplicate_pairs(sim_df, similarity_threshold)
    return filter_duplicates(df, pairs)
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_dedupe.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/dedupe.py tests/test_dedupe.py
git commit -m "feat(lib): add dedupe.py embedding-based duplicate detection"
```

---

## Task 5: lib/rating.py — Scoring Functions (no LLM)

**Files:**
- Create: `lib/rating.py`
- Create: `tests/test_rating.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/do_rating.py` (lines 1-150 for scoring)

### Step 1: Write failing tests for pure scoring functions

```python
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
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_rating.py -v
```
Expected: FAIL

### Step 3: Implement pure scoring functions

Create `lib/rating.py`:

```python
from __future__ import annotations
import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DEFAULT_CONCURRENCY

logger = logging.getLogger(__name__)

LN2 = math.log(2)
MAX_ARTICLE_AGE_DAYS = 7


def compute_recency_score(published_date: Optional[datetime]) -> float:
    """Half-life of 1 day: 2 * exp(-ln2 * age_days) - 1. Zero for articles > 7 days."""
    if published_date is None:
        return 0.0
    age = (datetime.now() - published_date).total_seconds() / 86400
    if age > MAX_ARTICLE_AGE_DAYS:
        return 0.0
    return 2 * math.exp(-LN2 * age) - 1


def compute_length_score(content_length: int) -> float:
    """log10(content_length) - 3, clipped to [0, 2]."""
    if content_length <= 0:
        return 0.0
    raw = math.log10(content_length) - 3
    return max(0.0, min(2.0, raw))


def compute_composite_rating(
    reputation: float,
    length_score: float,
    on_topic: float,
    importance: float,
    low_quality: float,
    recency: float,
    bt_zscore: float,
) -> float:
    """Composite article rating formula."""
    return reputation + length_score + on_topic + importance - low_quality + recency + bt_zscore
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_rating.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/rating.py tests/test_rating.py
git commit -m "feat(lib): add rating.py scoring functions (recency, length, composite)"
```

---

## Task 6: lib/rating.py — LLM Assessments

**Files:**
- Modify: `lib/rating.py`
- Modify: `tests/test_rating.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/do_rating.py` (lines 150-350 for LLM assessment functions)

### Step 1: Write failing tests for LLM assessment wrappers

Add to `tests/test_rating.py`:

```python
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
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_rating.py::TestAssessQuality -v
```
Expected: FAIL

### Step 3: Implement LLM assessment functions

Add to `lib/rating.py`:

```python
from llm import create_agent, CLAUDE_SONNET_MODEL
from prompts import RATE_QUALITY, RATE_ON_TOPIC, RATE_IMPORTANCE


async def _assess_with_probs(df: pd.DataFrame, prompt_config, text_field: str = "input_str") -> pd.Series:
    """Generic LLM probability assessment. Iterates rows, gets P(token='1')."""
    agent = create_agent(
        model=prompt_config.model,
        system_prompt=prompt_config.system_prompt,
        user_prompt=prompt_config.user_prompt,
        reasoning_effort=prompt_config.reasoning_effort,
    )
    results = []
    for _, row in df.iterrows():
        input_text = f"Title: {row.get('title', '')}\nSummary: {row.get('summary', '')}"
        probs = await agent.run_prompt_with_probs(
            variables={"input_text": input_text},
            target_tokens=["1", "0"],
        )
        results.append(probs.get("1", 0.0))
    return pd.Series(results, index=df.index)


async def assess_quality(df: pd.DataFrame) -> pd.Series:
    """LLM probability of low quality."""
    return await _assess_with_probs(df, RATE_QUALITY)


async def assess_on_topic(df: pd.DataFrame) -> pd.Series:
    """LLM probability of AI-relevance."""
    return await _assess_with_probs(df, RATE_ON_TOPIC)


async def assess_importance(df: pd.DataFrame) -> pd.Series:
    """LLM probability of importance."""
    return await _assess_with_probs(df, RATE_IMPORTANCE)
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_rating.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/rating.py tests/test_rating.py
git commit -m "feat(lib): add rating.py LLM assessment functions (quality, on_topic, importance)"
```

---

## Task 7: lib/rating.py — Bradley-Terry Battle System

**Files:**
- Modify: `lib/rating.py`
- Modify: `tests/test_rating.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/do_rating.py` (lines 350-641 for BT system)

### Step 1: Write failing tests for BT system

Add to `tests/test_rating.py`:

```python
class TestSwissPairing:
    def test_pairs_all_articles(self):
        from lib.rating import swiss_pairing
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "rating": [5.0, 4.0, 3.0, 2.0],
        })
        history = set()
        pairs = swiss_pairing(df, history)
        assert len(pairs) >= 1
        # Each pair should be a tuple of two different IDs
        for a, b in pairs:
            assert a != b
            assert a in df["id"].values
            assert b in df["id"].values

    def test_avoids_repeat_battles(self):
        from lib.rating import swiss_pairing
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "rating": [5.0, 4.0, 3.0, 2.0],
        })
        history = {(1, 2), (2, 1)}
        pairs = swiss_pairing(df, history)
        for a, b in pairs:
            assert (a, b) not in history


class TestRunBradleyTerry:
    @pytest.mark.asyncio
    @patch("lib.rating.create_agent")
    async def test_returns_series(self, mock_create):
        from lib.rating import run_bradley_terry
        # Mock battle agent that returns items in order
        mock_agent = AsyncMock()
        mock_agent.prompt_list.return_value = [
            {"id": 1}, {"id": 2}, {"id": 3}
        ]
        mock_create.return_value = mock_agent

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "title": ["A", "B", "C"],
            "summary": ["a", "b", "c"],
            "rating": [0.0, 0.0, 0.0],
        })
        result = await run_bradley_terry(df, max_rounds=2, batch_size=3)
        assert isinstance(result, pd.Series)
        assert len(result) == 3


class TestRateArticles:
    @pytest.mark.asyncio
    @patch("lib.rating.run_bradley_terry")
    @patch("lib.rating.assess_importance")
    @patch("lib.rating.assess_on_topic")
    @patch("lib.rating.assess_quality")
    async def test_end_to_end(self, mock_q, mock_t, mock_i, mock_bt):
        from lib.rating import rate_articles
        mock_q.return_value = pd.Series([0.1, 0.2])
        mock_t.return_value = pd.Series([0.9, 0.8])
        mock_i.return_value = pd.Series([0.7, 0.6])
        mock_bt.return_value = pd.Series([0.0, 0.0])

        df = pd.DataFrame({
            "id": [1, 2],
            "title": ["A", "B"],
            "summary": ["a", "b"],
            "content_length": [5000, 10000],
            "published": [datetime.now(), datetime.now() - timedelta(hours=12)],
            "reputation": [1.0, 1.5],
        })
        result = await rate_articles(df)
        assert "rating" in result.columns
        assert len(result) == 2
        assert result.iloc[0]["rating"] != result.iloc[1]["rating"]
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_rating.py::TestSwissPairing -v
pytest tests/test_rating.py::TestRunBradleyTerry -v
```
Expected: FAIL

### Step 3: Implement BT battle system and rate_articles

Add to `lib/rating.py`. Port from original do_rating.py lines 350-641:

Key functions:
- `swiss_pairing(df, battle_history)` — pair articles by similar rank, avoid repeat battles
- `swiss_batching(df, battle_history, batch_size=6)` — group pairs into battle batches
- `process_battle_round(df, batches, agent)` — LLM ranks each batch, extract pairwise wins
- `run_bradley_terry(df, max_rounds=8, batch_size=6)` — iterative BT with convergence check
- `rate_articles(df, db_path=None)` — top-level: assess → compute composite → BT → final rating

BT convergence: stop when avg rank change < n_stories/100, min rounds = max_rounds//2.
Final BT scores: `choix.opt_pairwise()` → z-score normalize.

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_rating.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/rating.py tests/test_rating.py
git commit -m "feat(lib): add rating.py Bradley-Terry battle system and rate_articles pipeline"
```

---

## Task 8: lib/cluster.py — Embeddings & Summary Helpers

**Files:**
- Create: `lib/cluster.py`
- Create: `tests/test_cluster.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/do_cluster.py` (lines 1-200 for helpers)

### Step 1: Write failing tests

```python
# tests/test_cluster.py
"""Tests for lib/cluster.py — HDBSCAN clustering and topic naming."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestCreateExtendedSummary:
    def test_concatenates_fields(self):
        from lib.cluster import _create_extended_summary
        row = {
            "title": "AI Breakthrough",
            "description": "Major discovery",
            "topics": "AI, Research",
            "summary": "Scientists found new approach",
        }
        result = _create_extended_summary(row)
        assert "AI Breakthrough" in result
        assert "Major discovery" in result
        assert "AI, Research" in result
        assert "Scientists found" in result

    def test_handles_missing_fields(self):
        from lib.cluster import _create_extended_summary
        row = {"title": "AI Breakthrough"}
        result = _create_extended_summary(row)
        assert "AI Breakthrough" in result


class TestCreateShortSummary:
    def test_concatenates_fields(self):
        from lib.cluster import _create_short_summary
        row = {"short_summary": "Brief summary", "topics": "AI, ML"}
        result = _create_short_summary(row)
        assert "Brief summary" in result
        assert "AI, ML" in result


class TestLoadUmapReducer:
    def test_loads_pickle(self, tmp_path):
        from lib.cluster import load_umap_reducer
        import pickle
        # Create a mock reducer
        mock_reducer = MagicMock()
        mock_reducer.transform = MagicMock(return_value=np.zeros((5, 690)))
        pkl_path = tmp_path / "test_reducer.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(mock_reducer, f)

        reducer = load_umap_reducer(str(pkl_path))
        assert reducer is not None

    def test_raises_on_missing_file(self):
        from lib.cluster import load_umap_reducer
        with pytest.raises(FileNotFoundError):
            load_umap_reducer("/nonexistent/reducer.pkl")


class TestCalculateClusteringMetrics:
    def test_returns_metrics_dict(self):
        from lib.cluster import calculate_clustering_metrics
        embeddings = np.random.rand(20, 10)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, -1, -1, -1, -1, -1])
        metrics = calculate_clustering_metrics(embeddings, labels, clusterer=None)
        assert "silhouette" in metrics
        assert "noise_ratio" in metrics
        assert 0 <= metrics["noise_ratio"] <= 1

    def test_all_noise_returns_zero_silhouette(self):
        from lib.cluster import calculate_clustering_metrics
        embeddings = np.random.rand(10, 5)
        labels = np.array([-1] * 10)
        metrics = calculate_clustering_metrics(embeddings, labels, clusterer=None)
        assert metrics["silhouette"] == 0.0
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_cluster.py -v
```
Expected: FAIL

### Step 3: Implement helper functions

Create `lib/cluster.py`:

```python
from __future__ import annotations
import logging
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from config import EMBEDDING_MODEL, MIN_COMPONENTS, RANDOM_STATE, OPTUNA_TRIALS

logger = logging.getLogger(__name__)


def _create_extended_summary(row: dict) -> str:
    """Concatenate title + description + topics + summary."""
    parts = [
        row.get("title", ""),
        row.get("description", ""),
        row.get("topics", ""),
        row.get("summary", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _create_short_summary(row: dict) -> str:
    """Concatenate short_summary + topics."""
    parts = [row.get("short_summary", ""), row.get("topics", "")]
    return " ".join(p for p in parts if p).strip()


def load_umap_reducer(path: str):
    """Load pretrained UMAP reducer from pickle file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"UMAP reducer not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def calculate_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray, clusterer: Any
) -> dict:
    """Calculate clustering quality metrics."""
    mask = labels >= 0
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    noise_ratio = (~mask).sum() / len(labels)

    if n_clusters < 2 or mask.sum() < 2:
        return {
            "silhouette": 0.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": float("inf"),
            "noise_ratio": noise_ratio,
            "n_clusters": n_clusters,
            "validity_index": 0.0,
        }

    sil = silhouette_score(embeddings[mask], labels[mask])
    ch = calinski_harabasz_score(embeddings[mask], labels[mask])
    db = davies_bouldin_score(embeddings[mask], labels[mask])
    validity = clusterer.relative_validity_ if clusterer and hasattr(clusterer, "relative_validity_") else 0.0

    return {
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
        "noise_ratio": noise_ratio,
        "n_clusters": n_clusters,
        "validity_index": validity,
    }
```

(Add `import os` to imports.)

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_cluster.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/cluster.py tests/test_cluster.py
git commit -m "feat(lib): add cluster.py helpers (summaries, UMAP loader, metrics)"
```

---

## Task 9: lib/cluster.py — Optuna Optimization & HDBSCAN

**Files:**
- Modify: `lib/cluster.py`
- Modify: `tests/test_cluster.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/do_cluster.py` (lines 200-500 for Optuna + HDBSCAN)

### Step 1: Write failing tests

Add to `tests/test_cluster.py`:

```python
class TestOptimizeHdbscan:
    def test_returns_best_params(self):
        from lib.cluster import optimize_hdbscan
        np.random.seed(42)
        # Create data with 3 clear clusters
        cluster1 = np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(20, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(20, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([cluster1, cluster2, cluster3])

        result = optimize_hdbscan(embeddings, n_trials=5)
        assert "min_cluster_size" in result
        assert "min_samples" in result
        assert "labels" in result
        assert "score" in result

    def test_handles_small_dataset(self):
        from lib.cluster import optimize_hdbscan
        embeddings = np.random.randn(5, 10)
        result = optimize_hdbscan(embeddings, n_trials=3)
        assert "labels" in result


class TestObjective:
    def test_returns_float_score(self):
        from lib.cluster import objective
        import optuna
        np.random.seed(42)
        embeddings = np.vstack([
            np.random.randn(15, 5) + [3, 0, 0, 0, 0],
            np.random.randn(15, 5) + [0, 3, 0, 0, 0],
        ])
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        score = objective(trial, embeddings)
        assert isinstance(score, float)
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_cluster.py::TestOptimizeHdbscan -v
```
Expected: FAIL

### Step 3: Implement Optuna optimization and HDBSCAN

Add to `lib/cluster.py`:

```python
import hdbscan
import optuna

# Silence Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, embeddings_array: np.ndarray) -> float:
    """Optuna objective: optimize HDBSCAN params. Returns composite score."""
    min_cluster_size = trial.suggest_int("min_cluster_size", 3, min(50, len(embeddings_array) // 3))
    min_samples = trial.suggest_int("min_samples", 2, min(30, min_cluster_size))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings_array)

    mask = labels >= 0
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2 or mask.sum() < n_clusters + 1:
        return 0.0

    sil = silhouette_score(embeddings_array[mask], labels[mask])
    validity = clusterer.relative_validity_ if hasattr(clusterer, "relative_validity_") else 0.0
    return 0.5 * sil + 0.5 * validity


def optimize_hdbscan(embeddings_array: np.ndarray, n_trials: int = OPTUNA_TRIALS) -> dict:
    """Run Optuna to find best HDBSCAN params."""
    if len(embeddings_array) < 6:
        labels = np.array([-1] * len(embeddings_array))
        return {"min_cluster_size": 3, "min_samples": 2, "labels": labels, "score": 0.0}

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, embeddings_array), n_trials=n_trials)

    best = study.best_params
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best["min_cluster_size"],
        min_samples=best["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings_array)

    return {
        "min_cluster_size": best["min_cluster_size"],
        "min_samples": best["min_samples"],
        "labels": labels,
        "score": study.best_value,
        "clusterer": clusterer,
    }
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_cluster.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/cluster.py tests/test_cluster.py
git commit -m "feat(lib): add cluster.py Optuna HDBSCAN optimization"
```

---

## Task 10: lib/cluster.py — Cluster Naming & Pipeline

**Files:**
- Modify: `lib/cluster.py`
- Modify: `tests/test_cluster.py`

**Reference:** `/Users/drucev/projects/OpenAIAgentsSDK/do_cluster.py` (lines 500-750 for naming + pipeline)

### Step 1: Write failing tests

Add to `tests/test_cluster.py`:

```python
class TestNameClusters:
    @pytest.mark.asyncio
    @patch("lib.cluster.create_agent")
    async def test_names_each_cluster(self, mock_create):
        from lib.cluster import name_clusters
        mock_agent = AsyncMock()
        mock_result = MagicMock()
        mock_result.topic = "AI Healthcare"
        mock_agent.prompt_dict.return_value = mock_result
        mock_create.return_value = mock_agent

        df = pd.DataFrame({
            "title": ["AI in hospitals", "Medical AI", "AI doctor", "Sports news"],
            "cluster_label": [0, 0, 0, -1],
        })
        result = await name_clusters(df)
        assert "cluster_name" in result.columns
        assert result[result["cluster_label"] == 0].iloc[0]["cluster_name"] == "AI Healthcare"
        assert result[result["cluster_label"] == -1].iloc[0]["cluster_name"] == "Other"


class TestDoClustering:
    @pytest.mark.asyncio
    @patch("lib.cluster.name_clusters")
    @patch("lib.cluster.optimize_hdbscan")
    @patch("lib.cluster.load_umap_reducer")
    @patch("lib.cluster.get_embeddings_df")
    async def test_end_to_end_pipeline(self, mock_embed, mock_umap, mock_opt, mock_name):
        from lib.cluster import do_clustering

        # Mock embeddings
        mock_embed.return_value = pd.DataFrame(
            np.random.randn(10, 50),
            index=range(10),
        )

        # Mock UMAP reducer
        mock_reducer = MagicMock()
        mock_reducer.transform.return_value = np.random.randn(10, 20)
        mock_umap.return_value = mock_reducer

        # Mock HDBSCAN
        mock_opt.return_value = {
            "labels": np.array([0, 0, 0, 1, 1, 1, 2, 2, -1, -1]),
            "score": 0.5,
            "min_cluster_size": 3,
            "min_samples": 2,
            "clusterer": MagicMock(),
        }

        # Mock naming
        async def fake_name(df):
            df["cluster_name"] = df["cluster_label"].apply(lambda x: f"Cluster {x}" if x >= 0 else "Other")
            return df
        mock_name.side_effect = fake_name

        df = pd.DataFrame({
            "title": [f"Article {i}" for i in range(10)],
            "summary": [f"Summary {i}" for i in range(10)],
            "description": ["desc"] * 10,
            "topics": ["ai"] * 10,
        })

        result = await do_clustering(df, umap_reducer_path="fake.pkl", n_trials=3)
        assert "cluster_label" in result.columns
        assert "cluster_name" in result.columns
        assert len(result) == 10
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_cluster.py::TestNameClusters -v
pytest tests/test_cluster.py::TestDoClustering -v
```
Expected: FAIL

### Step 3: Implement cluster naming and main pipeline

Add to `lib/cluster.py`:

```python
import openai as openai_client
from llm import create_agent
from prompts import TOPIC_WRITER
from pydantic import BaseModel


class TopicText(BaseModel):
    topic: str


async def get_embeddings_df(
    df: pd.DataFrame, model: str = EMBEDDING_MODEL, batch_size: int = 100
) -> pd.DataFrame:
    """Generate embeddings for extended summaries."""
    client = openai_client.AsyncOpenAI()
    texts = [_create_extended_summary(row) for _, row in df.iterrows()]
    texts = [t if t else "empty" for t in texts]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])

    return pd.DataFrame(all_embeddings, index=df.index)


async def name_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Name each cluster using Claude via topic_writer prompt."""
    result = df.copy()
    result["cluster_name"] = "Other"

    agent = create_agent(
        model=TOPIC_WRITER.model,
        system_prompt=TOPIC_WRITER.system_prompt,
        user_prompt=TOPIC_WRITER.user_prompt,
        output_type=TopicText,
        reasoning_effort=TOPIC_WRITER.reasoning_effort,
    )

    unique_labels = sorted(set(df["cluster_label"]))
    for label in unique_labels:
        if label < 0:
            continue
        cluster_df = df[df["cluster_label"] == label]
        if len(cluster_df) < 2:
            continue
        titles = cluster_df["title"].tolist()
        input_text = "\n".join(f"- {t}" for t in titles)
        parsed = await agent.prompt_dict({"input_text": input_text})
        result.loc[cluster_df.index, "cluster_name"] = parsed.topic

    return result


async def do_clustering(
    df: pd.DataFrame,
    umap_reducer_path: str = "umap_reducer.pkl",
    n_trials: int = OPTUNA_TRIALS,
) -> pd.DataFrame:
    """Top-level clustering pipeline: embed → reduce → optimize → cluster → name."""
    # 1. Get embeddings
    embeddings_df = await get_embeddings_df(df)
    embeddings_array = embeddings_df.values

    # 2. Reduce with pretrained UMAP
    reducer = load_umap_reducer(umap_reducer_path)
    reduced = reducer.transform(embeddings_array)

    # 3. Optimize HDBSCAN
    opt_result = optimize_hdbscan(reduced, n_trials=n_trials)

    # 4. Assign labels
    result = df.copy()
    result["cluster_label"] = opt_result["labels"]

    # 5. Name clusters
    result = await name_clusters(result)

    return result
```

### Step 4: Run tests to verify they pass

```bash
pytest tests/test_cluster.py -v
```
Expected: All PASS

### Step 5: Commit

```bash
git add lib/cluster.py tests/test_cluster.py
git commit -m "feat(lib): add cluster.py naming pipeline and do_clustering entry point"
```

---

## Task 11: Update CLAUDE.md & Final Verification

**Files:**
- Modify: `CLAUDE.md`

### Step 1: Run all tests

```bash
pytest tests/ -v
```
Expected: All PASS

### Step 2: Update CLAUDE.md project status

Change Phase 3 status from "not started" to "COMPLETE" and add lib/ files to Key Files:

```markdown
**Phase 3: Library modules — COMPLETE** (scrape, fetch, dedupe, rating, cluster)
```

Add to Key Files:
```markdown
lib/scrape.py  — Camoufox browser automation, rate limiting, text extraction
lib/fetch.py   — Source processor (RSS/HTML/API from sources.yaml)
lib/dedupe.py  — Embedding-based duplicate detection (cosine similarity)
lib/rating.py  — Composite rating formula + Bradley-Terry battles
lib/cluster.py — HDBSCAN + Optuna + UMAP + Claude cluster naming
prompts.py     — All 23 LLM prompt templates (system/user/model/reasoning_effort)
```

### Step 3: Run final check

```bash
pytest tests/ -v --tb=short
python -c "from lib.scrape import ScrapeResult, RateLimiter; print('scrape OK')"
python -c "from lib.fetch import Fetcher, parse_source_links; print('fetch OK')"
python -c "from lib.dedupe import process_dataframe_with_filtering; print('dedupe OK')"
python -c "from lib.rating import rate_articles, run_bradley_terry; print('rating OK')"
python -c "from lib.cluster import do_clustering; print('cluster OK')"
python -c "from prompts import ALL_PROMPTS; print(f'{len(ALL_PROMPTS)} prompts OK')"
```

### Step 4: Commit

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md — Phase 3 complete"
```

---

## Summary

| Task | Module | Description | Depends On |
|------|--------|-------------|------------|
| 0 | config | New constants, deps, sources.yaml, UMAP pkl | — |
| 1 | lib/scrape.py | Utilities, ScrapeResult, RateLimiter | Task 0 |
| 2 | lib/scrape.py | Browser automation, scrape_url, concurrent | Task 1 |
| 3 | lib/fetch.py | Source processor, RSS/HTML/API | Task 2 |
| 4 | lib/dedupe.py | Embedding dedup pipeline | Task 0 |
| 5 | lib/rating.py | Scoring functions (no LLM) | Task 0 |
| 6 | lib/rating.py | LLM assessments | Task 5 |
| 7 | lib/rating.py | Bradley-Terry battles + rate_articles | Task 6 |
| 8 | lib/cluster.py | Helpers, metrics, UMAP loader | Task 0 |
| 9 | lib/cluster.py | Optuna + HDBSCAN | Task 8 |
| 10 | lib/cluster.py | Cluster naming + do_clustering | Task 9 |
| 11 | CLAUDE.md | Final verification + docs update | All |

**Parallelizable groups:**
- Tasks 1-2 (scrape) can run in parallel with Tasks 4 (dedupe), 5-7 (rating), 8-10 (cluster)
- Task 3 (fetch) depends on Task 2 (scrape)
- Task 11 depends on all others
