# Phase 4: Bash-Based Step Scripts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create 5 CLI step scripts in `steps/` that orchestrate the lib/ modules into the data-heavy pipeline stages of the newsletter agent workflow.

**Architecture:** Each step script is a standalone CLI tool invoked as `python steps/<name>.py --session SESSION_ID [--db DB_PATH]`. Scripts load workflow state from SQLite, call lib/ module functions, persist results back to SQLite, checkpoint state, and print a JSON summary to stdout for the agent orchestrator. A shared `run_step()` helper in `steps/__init__.py` eliminates boilerplate for state lifecycle (load → start → complete/error → checkpoint → print).

**Tech Stack:** Python 3.11+, asyncio, argparse, pandas, existing lib/ modules (fetch, scrape, dedupe, rating, cluster), existing infrastructure (state.py, db.py, config.py, llm.py, prompts.py). Tests use pytest + pytest-asyncio + unittest.mock.

---

## Task 1: Step Runner Helper

Create the shared `run_step()` utility that handles state loading, step lifecycle, checkpointing, and JSON output for all step scripts.

**Files:**
- Modify: `steps/__init__.py`
- Test: `tests/test_steps.py`

### Step 1: Write the failing test

```python
# tests/test_steps.py
"""Tests for steps/ — step runner utility and CLI scripts."""
import json
import os
import pytest


TEST_DB = "test_steps.db"


@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestRunStep:
    def test_creates_fresh_state_and_checkpoints(self):
        from db import AgentState
        from state import NewsletterAgentState
        from steps import run_step

        AgentState.create_table(TEST_DB)

        def action(state: NewsletterAgentState) -> str:
            state.headline_data.append({"url": "https://test.com", "title": "Test"})
            return "Processed 1 item"

        result = run_step(
            step_name="gather_urls",
            session_id="test_session",
            db_path=TEST_DB,
            action=action,
        )
        assert result["status"] == "success"
        assert result["message"] == "Processed 1 item"

        # Verify checkpoint was saved
        loaded = NewsletterAgentState.load_latest_from_db("test_session", db_path=TEST_DB)
        assert loaded is not None
        assert len(loaded.headline_data) == 1
        assert loaded.is_step_complete("gather_urls")

    def test_loads_existing_state(self):
        from db import AgentState
        from state import NewsletterAgentState
        from steps import run_step

        AgentState.create_table(TEST_DB)

        # Create initial state with some data
        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data.append({"url": "https://existing.com", "title": "Existing"})
        state.complete_step("gather_urls", message="Initial gather")
        state.save_checkpoint("gather_urls")

        def action(state: NewsletterAgentState) -> str:
            # Should see the existing data
            assert len(state.headline_data) == 1
            return "Filtered"

        result = run_step(
            step_name="filter_urls",
            session_id="test_session",
            db_path=TEST_DB,
            action=action,
        )
        assert result["status"] == "success"

    def test_handles_action_error(self):
        from db import AgentState
        from state import NewsletterAgentState
        from steps import run_step

        AgentState.create_table(TEST_DB)

        def bad_action(state: NewsletterAgentState) -> str:
            raise ValueError("Something went wrong")

        result = run_step(
            step_name="gather_urls",
            session_id="test_session",
            db_path=TEST_DB,
            action=bad_action,
        )
        assert result["status"] == "error"
        assert "Something went wrong" in result["error"]

        # Step should be marked as error, not complete
        loaded = NewsletterAgentState.load_latest_from_db("test_session", db_path=TEST_DB)
        assert loaded is not None
        assert not loaded.is_step_complete("gather_urls")

    def test_async_action(self):
        from db import AgentState
        from state import NewsletterAgentState
        from steps import run_step

        AgentState.create_table(TEST_DB)

        async def async_action(state: NewsletterAgentState) -> str:
            return "Async done"

        result = run_step(
            step_name="gather_urls",
            session_id="test_session",
            db_path=TEST_DB,
            action=async_action,
        )
        assert result["status"] == "success"
        assert result["message"] == "Async done"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_steps.py::TestRunStep -v`
Expected: FAIL — `run_step` not importable from `steps`

### Step 3: Write minimal implementation

```python
# steps/__init__.py
"""Step runner utility for newsletter agent CLI scripts."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
from typing import Any, Callable, Dict, Union

from state import NewsletterAgentState

logger = logging.getLogger(__name__)


def run_step(
    step_name: str,
    session_id: str,
    db_path: str = "newsletter_agent.db",
    action: Callable[[NewsletterAgentState], Union[str, Any]] = lambda s: "",
) -> Dict[str, Any]:
    """Execute a workflow step with state lifecycle management.

    Loads (or creates) state, marks the step as started, runs the action,
    checkpoints, and returns a JSON-serializable result dict.

    The action callable receives the state and returns a status message string.
    It may be a sync or async function.

    Returns:
        Dict with "status" ("success" or "error") and "message" or "error".
    """
    from db import AgentState as AgentStateDB

    # Ensure table exists
    AgentStateDB.create_table(db_path)

    # Load existing state or create fresh
    state = NewsletterAgentState.load_latest_from_db(session_id, db_path=db_path)
    if state is None:
        state = NewsletterAgentState(session_id=session_id, db_path=db_path)

    state.start_step(step_name)

    try:
        if inspect.iscoroutinefunction(action):
            message = asyncio.run(action(state))
        else:
            message = action(state)

        state.complete_step(step_name, message=str(message))
        state.save_checkpoint(step_name)

        result = {"status": "success", "message": str(message)}
        return result

    except Exception as e:
        logger.error("Step %s failed: %s", step_name, e)
        state.error_step(step_name, str(e))
        state.save_checkpoint(step_name)

        result = {"status": "error", "error": str(e)}
        return result
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_steps.py::TestRunStep -v`
Expected: All 4 tests PASS

### Step 5: Commit

```bash
git add steps/__init__.py tests/test_steps.py
git commit -m "feat(steps): add run_step helper for CLI step lifecycle"
```

---

## Task 2: steps/gather_urls.py

Fetches headlines from all sources in `sources.yaml` using `Fetcher.fetch_all()`, inserts URLs into the `urls` table, and adds headlines to state.

**Files:**
- Create: `steps/gather_urls.py`
- Modify: `tests/test_steps.py` (add `TestGatherUrls` class)

### Step 1: Write the failing test

```python
# append to tests/test_steps.py

class TestGatherUrls:
    @pytest.mark.asyncio
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Url, AgentState
        AgentState.create_table(TEST_DB)
        Url.create_table(TEST_DB)
        yield

    @patch("steps.gather_urls.Fetcher")
    def test_gathers_and_persists_urls(self, mock_fetcher_cls):
        from steps.gather_urls import gather_urls_action
        from state import NewsletterAgentState
        from db import Url

        # Mock Fetcher context manager
        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "source": "TestSource",
                "results": [
                    {"title": "AI Article One", "url": "https://example.com/1", "source": "TestSource"},
                    {"title": "AI Article Two", "url": "https://example.com/2", "source": "TestSource"},
                ],
                "status": "success",
                "metadata": {"entry_count": 2},
            },
        ]
        mock_fetcher.__aenter__ = AsyncMock(return_value=mock_fetcher)
        mock_fetcher.__aexit__ = AsyncMock(return_value=False)
        mock_fetcher_cls.return_value = mock_fetcher

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(gather_urls_action(state))

        assert "2" in result  # message mentions count
        assert len(state.headline_data) == 2
        # URLs persisted to DB
        all_urls = Url.get_all(TEST_DB)
        assert len(all_urls) == 2

    @patch("steps.gather_urls.Fetcher")
    def test_deduplicates_across_sources(self, mock_fetcher_cls):
        from steps.gather_urls import gather_urls_action
        from state import NewsletterAgentState

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "source": "Source1",
                "results": [{"title": "Shared Article", "url": "https://shared.com/1", "source": "Source1"}],
                "status": "success",
                "metadata": {},
            },
            {
                "source": "Source2",
                "results": [{"title": "Shared Article", "url": "https://shared.com/1", "source": "Source2"}],
                "status": "success",
                "metadata": {},
            },
        ]
        mock_fetcher.__aenter__ = AsyncMock(return_value=mock_fetcher)
        mock_fetcher.__aexit__ = AsyncMock(return_value=False)
        mock_fetcher_cls.return_value = mock_fetcher

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(gather_urls_action(state))

        # Should deduplicate the shared URL
        assert len(state.headline_data) == 1

    @patch("steps.gather_urls.Fetcher")
    def test_skips_failed_sources(self, mock_fetcher_cls):
        from steps.gather_urls import gather_urls_action
        from state import NewsletterAgentState

        mock_fetcher = AsyncMock()
        mock_fetcher.fetch_all.return_value = [
            {
                "source": "GoodSource",
                "results": [{"title": "Good Article Here", "url": "https://good.com/1", "source": "GoodSource"}],
                "status": "success",
                "metadata": {},
            },
            {
                "source": "BadSource",
                "results": [],
                "status": "error",
                "metadata": {"error": "Connection failed"},
            },
        ]
        mock_fetcher.__aenter__ = AsyncMock(return_value=mock_fetcher)
        mock_fetcher.__aexit__ = AsyncMock(return_value=False)
        mock_fetcher_cls.return_value = mock_fetcher

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(gather_urls_action(state))

        assert len(state.headline_data) == 1
```

**Note:** Add these imports at the top of `tests/test_steps.py`:
```python
import asyncio
from unittest.mock import AsyncMock, patch
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_steps.py::TestGatherUrls -v`
Expected: FAIL — `steps.gather_urls` not found

### Step 3: Write implementation

```python
# steps/gather_urls.py
#!/usr/bin/env python3
"""Step 1: Gather URLs from all sources.

Run via: python steps/gather_urls.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime

from config import NEWSAGENTDB
from db import Url
from lib.fetch import Fetcher
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


async def gather_urls_action(state: NewsletterAgentState) -> str:
    """Fetch all sources and persist URLs to DB and state."""
    Url.create_table(state.db_path)

    async with Fetcher(sources_file=state.sources_file) as fetcher:
        results = await fetcher.fetch_all()

    total_urls = 0
    total_sources = 0
    failed_sources = 0

    for source_result in results:
        source_name = source_result["source"]
        if source_result["status"] != "success":
            failed_sources += 1
            logger.warning("Source %s failed: %s", source_name, source_result.get("metadata", {}))
            continue

        total_sources += 1
        for item in source_result["results"]:
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title:
                continue

            # Persist to Url table (skip duplicates)
            try:
                record = Url(
                    initial_url=url,
                    final_url=url,
                    title=title,
                    source=item.get("source", source_name),
                    created_at=datetime.now(),
                )
                record.insert(state.db_path)
            except sqlite3.IntegrityError:
                pass  # URL already exists

            total_urls += 1

    # Add to state (handles dedup internally)
    all_headlines = []
    for source_result in results:
        if source_result["status"] == "success":
            all_headlines.extend(source_result["results"])
    state.add_headlines(all_headlines)

    return f"Gathered {total_urls} URLs from {total_sources} sources ({failed_sources} failed)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: Gather URLs from sources")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="gather_urls",
        session_id=args.session,
        db_path=args.db,
        action=gather_urls_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_steps.py::TestGatherUrls -v`
Expected: All 3 tests PASS

### Step 5: Commit

```bash
git add steps/gather_urls.py tests/test_steps.py
git commit -m "feat(steps): add gather_urls step — fetches from all sources"
```

---

## Task 3: steps/filter_urls.py

Filters headlines by domain skiplist, then uses LLM to classify AI-relevance. Updates Url records and removes non-AI headlines from state.

**Files:**
- Create: `steps/filter_urls.py`
- Modify: `tests/test_steps.py` (add `TestFilterUrls` class)

### Step 1: Write the failing test

```python
# append to tests/test_steps.py

class TestFilterUrls:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Url, AgentState
        AgentState.create_table(TEST_DB)
        Url.create_table(TEST_DB)
        yield

    def test_filters_skiplist_domains(self):
        from steps.filter_urls import _filter_skiplist
        from state import NewsletterAgentState

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data = [
            {"url": "https://example.com/good", "title": "Good Article"},
            {"url": "https://finbold.com/bad", "title": "Bad Domain"},
            {"url": "https://www.cnn.com/news", "title": "Ignored Domain"},
        ]
        filtered = _filter_skiplist(state.headline_data)
        assert len(filtered) == 1
        assert filtered[0]["url"] == "https://example.com/good"

    @patch("steps.filter_urls.create_agent")
    def test_classifies_ai_relevance(self, mock_create):
        from steps.filter_urls import filter_urls_action
        from state import NewsletterAgentState
        from pydantic import BaseModel

        mock_agent = AsyncMock()
        # The filter agent returns a list of {id, value} dicts
        mock_agent.filter_dataframe.return_value = pd.Series([True, False], index=[0, 1])
        mock_create.return_value = mock_agent

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data = [
            {"url": "https://example.com/ai", "title": "AI Research Breakthrough"},
            {"url": "https://example.com/sports", "title": "Sports Final Scores"},
        ]

        result = asyncio.run(filter_urls_action(state))
        # Only AI-relevant article should remain
        assert len(state.headline_data) == 1
        assert "ai" in state.headline_data[0]["url"]
```

**Note:** Add `import pandas as pd` at the top of `tests/test_steps.py`.

### Step 2: Run test to verify it fails

Run: `pytest tests/test_steps.py::TestFilterUrls -v`
Expected: FAIL — `steps.filter_urls` not found

### Step 3: Write implementation

```python
# steps/filter_urls.py
#!/usr/bin/env python3
"""Step 2: Filter URLs — domain skiplist + LLM AI-relevance classification.

Run via: python steps/filter_urls.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd
from pydantic import BaseModel

from config import DOMAIN_SKIPLIST, IGNORE_LIST, NEWSAGENTDB
from db import Url
from llm import create_agent
from prompts import FILTER_URLS
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


class FilterResult(BaseModel):
    """Schema for FILTER_URLS structured output."""
    id: int
    value: bool


class FilterResultList(BaseModel):
    results_list: List[FilterResult]


def _filter_skiplist(headlines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove headlines from domains in DOMAIN_SKIPLIST and IGNORE_LIST."""
    blocked = set(DOMAIN_SKIPLIST) | set(IGNORE_LIST)
    filtered = []
    for h in headlines:
        url = h.get("url", "")
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Check both with and without www prefix
        bare = hostname.removeprefix("www.")
        if hostname in blocked or bare in blocked:
            continue
        filtered.append(h)
    return filtered


async def filter_urls_action(state: NewsletterAgentState) -> str:
    """Filter headlines by domain skiplist, then classify AI-relevance via LLM."""
    initial_count = len(state.headline_data)

    # Step 1: Domain skiplist
    state.headline_data = _filter_skiplist(state.headline_data)
    after_skiplist = len(state.headline_data)

    # Step 2: LLM classification
    if not state.headline_data:
        return f"No headlines to filter (started with {initial_count})"

    df = pd.DataFrame(state.headline_data).reset_index(drop=True)
    df["id"] = df.index

    agent = create_agent(
        model=FILTER_URLS.model,
        system_prompt=FILTER_URLS.system_prompt,
        user_prompt=FILTER_URLS.user_prompt,
        output_type=FilterResultList,
        reasoning_effort=FILTER_URLS.reasoning_effort,
    )

    is_ai = await agent.filter_dataframe(
        df=df[["id", "title"]],
        chunk_size=25,
        value_field="value",
        item_list_field="results_list",
        item_id_field="id",
    )

    # Keep only AI-relevant headlines
    ai_mask = is_ai.astype(bool)
    ai_urls = set(df.loc[ai_mask, "url"].tolist()) if "url" in df.columns else set()
    state.headline_data = [h for h in state.headline_data if h.get("url") in ai_urls]

    # Update Url records
    Url.create_table(state.db_path)
    for _, row in df.iterrows():
        url_str = row.get("url", "")
        is_ai_val = bool(ai_mask.get(row.name, False))
        # Best-effort update of existing records
        try:
            from db import _connect
            with _connect(state.db_path) as conn:
                conn.execute(
                    "UPDATE urls SET isAI = ? WHERE initial_url = ?",
                    (int(is_ai_val), url_str),
                )
        except Exception:
            pass

    after_filter = len(state.headline_data)
    return (
        f"Filtered {initial_count} -> {after_skiplist} (skiplist) -> "
        f"{after_filter} AI-relevant headlines"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Filter URLs")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="filter_urls",
        session_id=args.session,
        db_path=args.db,
        action=filter_urls_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_steps.py::TestFilterUrls -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add steps/filter_urls.py tests/test_steps.py
git commit -m "feat(steps): add filter_urls step — skiplist + LLM AI classification"
```

---

## Task 4: steps/download_articles.py

Downloads article content for filtered URLs using `scrape_urls_concurrent()`. Creates Article records with HTML/text paths and content metadata. Resolves domains and creates Site records.

**Files:**
- Create: `steps/download_articles.py`
- Modify: `tests/test_steps.py` (add `TestDownloadArticles` class)

### Step 1: Write the failing test

```python
# append to tests/test_steps.py

class TestDownloadArticles:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Article, Site, AgentState
        AgentState.create_table(TEST_DB)
        Article.create_table(TEST_DB)
        Site.create_table(TEST_DB)
        yield

    @patch("steps.download_articles.get_browser")
    @patch("steps.download_articles.scrape_urls_concurrent")
    def test_downloads_and_creates_articles(self, mock_scrape, mock_browser):
        from steps.download_articles import download_articles_action
        from state import NewsletterAgentState
        from db import Article
        from lib.scrape import ScrapeResult

        mock_browser.return_value = AsyncMock()
        mock_scrape.return_value = [
            ScrapeResult(
                html_path="/tmp/test.html",
                text_path="/tmp/test.txt",
                final_url="https://example.com/article-1",
                last_updated="2026-02-13",
                status="success",
            ),
        ]

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data = [
            {"url": "https://example.com/article-1", "title": "Test Article", "source": "TestSource"},
        ]

        # Create a minimal text file for content_length
        with open("/tmp/test.txt", "w") as f:
            f.write("This is test content." * 50)

        result = asyncio.run(download_articles_action(state))

        assert "1" in result  # mentions count
        articles = Article.get_all(TEST_DB)
        assert len(articles) == 1
        assert articles[0].title == "Test Article"
        assert articles[0].status == "success"

    @patch("steps.download_articles.get_browser")
    @patch("steps.download_articles.scrape_urls_concurrent")
    def test_skips_failed_scrapes(self, mock_scrape, mock_browser):
        from steps.download_articles import download_articles_action
        from state import NewsletterAgentState
        from db import Article
        from lib.scrape import ScrapeResult

        mock_browser.return_value = AsyncMock()
        mock_scrape.return_value = [
            ScrapeResult(status="error"),
            ScrapeResult(
                html_path="/tmp/ok.html",
                text_path="/tmp/ok.txt",
                final_url="https://example.com/ok",
                status="success",
            ),
        ]

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        state.headline_data = [
            {"url": "https://bad.com/fail", "title": "Failed", "source": "S1"},
            {"url": "https://example.com/ok", "title": "OK Article", "source": "S2"},
        ]

        with open("/tmp/ok.txt", "w") as f:
            f.write("OK content here." * 50)

        result = asyncio.run(download_articles_action(state))
        articles = Article.get_all(TEST_DB)
        # Only the successful scrape creates an article
        assert len(articles) == 1
        assert articles[0].status == "success"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_steps.py::TestDownloadArticles -v`
Expected: FAIL — `steps.download_articles` not found

### Step 3: Write implementation

```python
# steps/download_articles.py
#!/usr/bin/env python3
"""Step 3: Download article content via browser scraping.

Run via: python steps/download_articles.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from typing import Any, Dict, List

import tldextract

from config import FIREFOX_PROFILE_PATH, NEWSAGENTDB, PAGES_DIR, TEXT_DIR
from db import Article, Site
from lib.scrape import get_browser, scrape_urls_concurrent
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


def _get_content_length(text_path: str) -> int:
    """Read text file and return character count."""
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            return len(f.read())
    except (FileNotFoundError, OSError):
        return 0


def _extract_domain(url: str) -> str:
    """Extract registerable domain from URL."""
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


async def download_articles_action(state: NewsletterAgentState) -> str:
    """Scrape article content for all headlines and create Article records."""
    Article.create_table(state.db_path)
    Site.create_table(state.db_path)

    if not state.headline_data:
        return "No headlines to download"

    # Prepare URL list for scraper
    urls_to_scrape: List[Dict[str, str]] = [
        {"url": h["url"], "title": h.get("title", "")}
        for h in state.headline_data
        if h.get("url")
    ]

    browser = await get_browser(FIREFOX_PROFILE_PATH)
    results = await scrape_urls_concurrent(
        urls=urls_to_scrape,
        browser_context=browser,
        html_dir=PAGES_DIR,
        text_dir=TEXT_DIR,
    )

    success_count = 0
    error_count = 0

    for headline, scrape_result in zip(state.headline_data, results):
        url = headline.get("url", "")
        title = headline.get("title", "")
        source = headline.get("source", "")
        domain = _extract_domain(url)

        if scrape_result.status not in ("success", "no_content"):
            error_count += 1
            continue

        content_length = _get_content_length(scrape_result.text_path)

        # Upsert Site record
        try:
            site = Site(domain_name=domain, site_name=source)
            site.insert(state.db_path)
        except sqlite3.IntegrityError:
            pass

        # Create Article record
        try:
            article = Article(
                final_url=scrape_result.final_url or url,
                url=url,
                source=source,
                title=title,
                html_path=scrape_result.html_path,
                text_path=scrape_result.text_path,
                content_length=content_length,
                domain=domain,
                site_name=source,
                status=scrape_result.status,
                last_updated=scrape_result.last_updated or None,
                rss_summary=headline.get("summary"),
                published=headline.get("published"),
            )
            article.insert(state.db_path)
            success_count += 1
        except sqlite3.IntegrityError:
            # Article already exists (same final_url) — skip
            success_count += 1

    return f"Downloaded {success_count} articles ({error_count} failed)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Download articles")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="download_articles",
        session_id=args.session,
        db_path=args.db,
        action=download_articles_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_steps.py::TestDownloadArticles -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add steps/download_articles.py tests/test_steps.py
git commit -m "feat(steps): add download_articles step — browser scraping + Article persistence"
```

---

## Task 5: steps/rate_articles.py

Loads articles from DB into a DataFrame, runs the full rating pipeline from `lib/rating.py` (LLM quality/topic/importance assessments + Bradley-Terry battles), and updates Article records with ratings.

**Files:**
- Create: `steps/rate_articles.py`
- Modify: `tests/test_steps.py` (add `TestRateArticles` class)

### Step 1: Write the failing test

```python
# append to tests/test_steps.py

class TestRateArticles:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Article, Site, AgentState
        AgentState.create_table(TEST_DB)
        Article.create_table(TEST_DB)
        Site.create_table(TEST_DB)
        yield

    @patch("steps.rate_articles.rate_articles_lib")
    def test_rates_and_updates_articles(self, mock_rate):
        from steps.rate_articles import rate_articles_action
        from state import NewsletterAgentState
        from db import Article
        from datetime import datetime

        # Insert test articles
        for i in range(3):
            Article(
                final_url=f"https://example.com/{i}",
                url=f"https://example.com/{i}",
                title=f"Article {i}",
                source="TestSource",
                content_length=5000,
                status="success",
                domain="example.com",
                site_name="TestSource",
            ).insert(TEST_DB)

        # Mock the rating pipeline to return a DataFrame with rating columns
        async def fake_rate(df):
            df = df.copy()
            df["low_quality"] = 0.1
            df["on_topic"] = 0.9
            df["importance"] = 0.8
            df["recency"] = 0.5
            df["length_score"] = 1.0
            df["bt_zscore"] = 0.0
            df["rating"] = 4.2
            return df
        mock_rate.side_effect = fake_rate

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(rate_articles_action(state))

        assert "3" in result  # rated 3 articles
        # Verify articles were updated in DB
        articles = Article.get_all(TEST_DB)
        for a in articles:
            assert a.rating == pytest.approx(4.2, abs=0.01)

    def test_handles_no_articles(self):
        from steps.rate_articles import rate_articles_action
        from state import NewsletterAgentState

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(rate_articles_action(state))
        assert "0" in result or "No" in result
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_steps.py::TestRateArticles -v`
Expected: FAIL — `steps.rate_articles` not found

### Step 3: Write implementation

```python
# steps/rate_articles.py
#!/usr/bin/env python3
"""Step 5: Rate articles — LLM assessments + Bradley-Terry battles.

Run via: python steps/rate_articles.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from config import NEWSAGENTDB
from db import Article
from lib.rating import rate_articles as rate_articles_lib
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


def _articles_to_dataframe(db_path: str) -> pd.DataFrame:
    """Load all successful articles from DB into a DataFrame."""
    articles = Article.get_all(db_path)
    if not articles:
        return pd.DataFrame()

    rows = []
    for a in articles:
        if a.status != "success":
            continue
        rows.append({
            "db_id": a.id,
            "id": a.id,
            "final_url": a.final_url,
            "title": a.title,
            "summary": a.summary or a.rss_summary or "",
            "content_length": a.content_length,
            "published": a.published,
            "reputation": a.reputation or 0.0,
            "source": a.source,
            "domain": a.domain,
        })
    return pd.DataFrame(rows)


async def rate_articles_action(state: NewsletterAgentState) -> str:
    """Load articles, run rating pipeline, update DB."""
    Article.create_table(state.db_path)

    df = _articles_to_dataframe(state.db_path)
    if df.empty:
        return "No articles to rate"

    rated_df = await rate_articles_lib(df)

    # Update Article records with rating data
    rating_cols = ["low_quality", "on_topic", "importance", "rating"]
    for _, row in rated_df.iterrows():
        db_id = int(row["db_id"])
        article = Article.get(state.db_path, db_id)
        if article is None:
            continue
        article.rating = float(row["rating"])
        article.update(state.db_path)

    return f"Rated {len(rated_df)} articles"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: Rate articles")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="rate_articles",
        session_id=args.session,
        db_path=args.db,
        action=rate_articles_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_steps.py::TestRateArticles -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add steps/rate_articles.py tests/test_steps.py
git commit -m "feat(steps): add rate_articles step — LLM assessments + Bradley-Terry scoring"
```

---

## Task 6: steps/cluster_topics.py

Loads rated articles from DB, runs the clustering pipeline from `lib/cluster.py` (embed → UMAP → HDBSCAN → name), and updates Article records with cluster labels and names. Updates state with cluster metadata.

**Files:**
- Create: `steps/cluster_topics.py`
- Modify: `tests/test_steps.py` (add `TestClusterTopics` class)

### Step 1: Write the failing test

```python
# append to tests/test_steps.py

class TestClusterTopics:
    @pytest.fixture(autouse=True)
    def setup_tables(self):
        from db import Article, AgentState
        AgentState.create_table(TEST_DB)
        Article.create_table(TEST_DB)
        yield

    @patch("steps.cluster_topics.do_clustering")
    def test_clusters_and_updates_articles(self, mock_cluster):
        from steps.cluster_topics import cluster_topics_action
        from state import NewsletterAgentState
        from db import Article

        # Insert test articles
        for i in range(4):
            Article(
                final_url=f"https://example.com/{i}",
                url=f"https://example.com/{i}",
                title=f"Article {i}",
                source="TestSource",
                content_length=5000,
                status="success",
                rating=float(i),
                domain="example.com",
                site_name="TestSource",
            ).insert(TEST_DB)

        # Mock clustering to assign labels and names
        async def fake_cluster(df, **kwargs):
            df = df.copy()
            df["cluster_label"] = [0, 0, 1, 1]
            df["cluster_name"] = ["AI Policy", "AI Policy", "LLM Updates", "LLM Updates"]
            return df
        mock_cluster.side_effect = fake_cluster

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(cluster_topics_action(state))

        assert "4" in result  # clustered 4 articles
        assert "2" in result  # 2 clusters
        # Verify state has cluster names
        assert len(state.cluster_names) == 2
        assert "AI Policy" in state.cluster_names

    def test_handles_no_articles(self):
        from steps.cluster_topics import cluster_topics_action
        from state import NewsletterAgentState

        state = NewsletterAgentState(session_id="test_session", db_path=TEST_DB)
        result = asyncio.run(cluster_topics_action(state))
        assert "No" in result or "0" in result
```

### Step 2: Run test to verify it fails

Run: `pytest tests/test_steps.py::TestClusterTopics -v`
Expected: FAIL — `steps.cluster_topics` not found

### Step 3: Write implementation

```python
# steps/cluster_topics.py
#!/usr/bin/env python3
"""Step 6: Cluster topics — HDBSCAN clustering + Claude-powered topic naming.

Run via: python steps/cluster_topics.py --session SESSION_ID [--db DB_PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from config import NEWSAGENTDB
from db import Article
from lib.cluster import do_clustering
from state import NewsletterAgentState
from steps import run_step

logger = logging.getLogger(__name__)


def _articles_to_dataframe(db_path: str) -> pd.DataFrame:
    """Load all successful articles from DB into a DataFrame for clustering."""
    articles = Article.get_all(db_path)
    if not articles:
        return pd.DataFrame()

    rows = []
    for a in articles:
        if a.status != "success":
            continue
        rows.append({
            "db_id": a.id,
            "final_url": a.final_url,
            "title": a.title,
            "summary": a.summary or a.rss_summary or "",
            "short_summary": a.short_summary or "",
            "description": a.description or "",
            "topics": a.topics or "",
            "rating": a.rating,
            "source": a.source,
        })
    return pd.DataFrame(rows)


async def cluster_topics_action(state: NewsletterAgentState) -> str:
    """Load articles, run clustering pipeline, update DB and state."""
    Article.create_table(state.db_path)

    df = _articles_to_dataframe(state.db_path)
    if df.empty:
        return "No articles to cluster"

    clustered_df = await do_clustering(df)

    # Update Article records with cluster data
    for _, row in clustered_df.iterrows():
        db_id = int(row["db_id"])
        article = Article.get(state.db_path, db_id)
        if article is None:
            continue
        article.cluster_label = str(row.get("cluster_label", -1))
        article.update(state.db_path)

    # Update state with cluster metadata
    if "cluster_name" in clustered_df.columns:
        unique_names = sorted(
            clustered_df.loc[
                clustered_df["cluster_name"] != "Other", "cluster_name"
            ].unique().tolist()
        )
        state.cluster_names = unique_names

        # Build clusters dict: {cluster_name: [url1, url2, ...]}
        clusters = {}
        for name in unique_names:
            mask = clustered_df["cluster_name"] == name
            clusters[name] = clustered_df.loc[mask, "final_url"].tolist()
        state.clusters = clusters

    n_clusters = len(state.cluster_names)
    return f"Clustered {len(clustered_df)} articles into {n_clusters} topics"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Cluster topics")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--db", default=NEWSAGENTDB, help="Database path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_step(
        step_name="cluster_topics",
        session_id=args.session,
        db_path=args.db,
        action=cluster_topics_action,
    )
    print(json.dumps(result))
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 4: Run test to verify it passes

Run: `pytest tests/test_steps.py::TestClusterTopics -v`
Expected: All tests PASS

### Step 5: Commit

```bash
git add steps/cluster_topics.py tests/test_steps.py
git commit -m "feat(steps): add cluster_topics step — HDBSCAN clustering + topic naming"
```

---

## Task 7: Integration Test + Docs Update

Run all step tests together, verify they all pass. Update CLAUDE.md to reflect Phase 4 completion.

**Files:**
- Modify: `CLAUDE.md`

### Step 1: Run all step tests

Run: `pytest tests/test_steps.py -v`
Expected: All tests PASS

### Step 2: Run full test suite

Run: `pytest tests/ -v`
Expected: All tests PASS (no regressions)

### Step 3: Update CLAUDE.md

Change Phase 4 status from "not started" to "COMPLETE":

```
**Phase 4: Bash-based steps (steps/) — COMPLETE** (gather_urls, filter_urls, download_articles, rate_articles, cluster_topics)
```

Add to Key Files:
```
steps/gather_urls.py      — Step 1: fetch all sources → Url table
steps/filter_urls.py      — Step 2: domain skiplist + LLM AI classification
steps/download_articles.py — Step 3: browser scraping → Article records
steps/rate_articles.py    — Step 5: LLM assessments + Bradley-Terry → ratings
steps/cluster_topics.py   — Step 6: HDBSCAN + UMAP + Claude → topic clusters
```

### Step 4: Commit

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md — Phase 4 complete"
```

---

## Data Flow Summary

```
sources.yaml
     │
     ▼
[gather_urls] ──→ Url table + state.headline_data
     │
     ▼
[filter_urls] ──→ Remove skiplist domains, LLM AI-classification
     │                Updates Url.isAI, filters state.headline_data
     ▼
[download_articles] ──→ Browser scrape → HTML/text files
     │                    Creates Article records + Site records
     ▼
[extract_summaries] ──→ (Phase 5 MCP tool — Claude summarizes articles)
     │                    Updates Article.summary, Article.short_summary
     ▼
[rate_articles] ──→ LLM quality/topic/importance + Bradley-Terry
     │                Updates Article.rating
     ▼
[cluster_topics] ──→ Embed → UMAP → HDBSCAN → Claude naming
                      Updates Article.cluster_label, state.cluster_names
```

## Usage

Each step script can be run standalone:

```bash
source .venv/bin/activate

# Step 1: Gather URLs
python steps/gather_urls.py --session newsletter_20260213

# Step 2: Filter URLs
python steps/filter_urls.py --session newsletter_20260213

# Step 3: Download articles
python steps/download_articles.py --session newsletter_20260213

# Step 5: Rate articles (step 4 is an MCP tool in Phase 5)
python steps/rate_articles.py --session newsletter_20260213

# Step 6: Cluster topics
python steps/cluster_topics.py --session newsletter_20260213
```

Each outputs JSON: `{"status": "success", "message": "..."}` or `{"status": "error", "error": "..."}`.
