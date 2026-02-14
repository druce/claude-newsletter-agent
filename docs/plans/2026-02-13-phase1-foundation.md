# Phase 1: Foundation (state, db, config) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the foundational modules (config, state, db) and repo skeleton for the Claude Agent SDK newsletter rewrite.

**Architecture:** Clean rewrite of the OpenAI Agents SDK newsletter system. Phase 1 establishes the data layer: configuration constants, Pydantic-based workflow state management with 9-step tracking, and SQLite persistence via dataclass-based CRUD models. No ORM — raw sqlite3 with a cleaner base class pattern than the original.

**Tech Stack:** Python 3.11+, Pydantic v2, sqlite3, pytest, pytest-asyncio

---

### Task 1: Create repo skeleton

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `steps/__init__.py`
- Create: `tools/__init__.py`
- Create: `lib/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Initialize git repo**

```bash
cd /Users/drucev/projects/ClaudeAgentSDK
git init
```

**Step 2: Create directory structure**

```bash
mkdir -p steps tools lib tests docs
```

**Step 3: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[codz]
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
env/

# Testing
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
coverage/
htmlcov/

# IDE
.idea/
.vscode/

# Environment
.env

# Data (generated at runtime)
download/
data/
*.db

# OS
.DS_Store
```

**Step 4: Create `requirements.txt`**

```
# Core
python-dotenv
pydantic>=2.0

# LLM
anthropic
openai
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
playwright-stealth
aiohttp
aiofiles
feedparser
trafilatura

# Utilities
tldextract
pyyaml

# Testing
pytest
pytest-asyncio
pytest-mock
pytest-cov
pytest-timeout
```

**Step 5: Create `.env.example`**

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...  # Used for embeddings only

# Browser
FIREFOX_PROFILE_PATH=/path/to/firefox/profile

# Optional news APIs
BRAVE_API_KEY=
NEWSCATCHER_API_KEY=
NEWSAPI_API_KEY=
GNEWS_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USERNAME=
REDDIT_PASSWORD=
```

**Step 6: Create empty `__init__.py` files**

Create empty files at:
- `steps/__init__.py`
- `tools/__init__.py`
- `lib/__init__.py`
- `tests/__init__.py`

**Step 7: Commit**

```bash
git add .gitignore requirements.txt .env.example steps/__init__.py tools/__init__.py lib/__init__.py tests/__init__.py CC.md docs/
git commit -m "chore: initialize repo skeleton with directory structure"
```

---

### Task 2: Write config.py — constants and topics

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Tests for config.py constants and configuration."""
import os


def test_canonical_topics_is_nonempty_list():
    from config import CANONICAL_TOPICS
    assert isinstance(CANONICAL_TOPICS, list)
    assert len(CANONICAL_TOPICS) > 100


def test_canonical_topics_contains_key_entries():
    from config import CANONICAL_TOPICS
    for expected in ["Agents", "Language Models", "OpenAI", "Anthropic", "Safety & Alignment"]:
        assert expected in CANONICAL_TOPICS, f"Missing topic: {expected}"


def test_download_paths_are_strings():
    from config import DOWNLOAD_ROOT, DOWNLOAD_DIR, PAGES_DIR, TEXT_DIR
    for path in [DOWNLOAD_ROOT, DOWNLOAD_DIR, PAGES_DIR, TEXT_DIR]:
        assert isinstance(path, str)
        assert len(path) > 0


def test_timeouts_are_positive():
    from config import REQUEST_TIMEOUT, SHORT_REQUEST_TIMEOUT, DOMAIN_RATE_LIMIT
    assert REQUEST_TIMEOUT > 0
    assert SHORT_REQUEST_TIMEOUT > 0
    assert DOMAIN_RATE_LIMIT > 0


def test_domain_skiplist_and_ignore_list():
    from config import DOMAIN_SKIPLIST, IGNORE_LIST
    assert isinstance(DOMAIN_SKIPLIST, list)
    assert isinstance(IGNORE_LIST, list)


def test_model_constants():
    from config import CLAUDE_SONNET, CLAUDE_HAIKU, DEFAULT_MODEL
    assert "sonnet" in CLAUDE_SONNET.lower() or "claude" in CLAUDE_SONNET.lower()
    assert "haiku" in CLAUDE_HAIKU.lower() or "claude" in CLAUDE_HAIKU.lower()
    assert DEFAULT_MODEL in (CLAUDE_SONNET, CLAUDE_HAIKU)


def test_default_concurrency():
    from config import DEFAULT_CONCURRENCY
    assert isinstance(DEFAULT_CONCURRENCY, int)
    assert DEFAULT_CONCURRENCY > 0


def test_newsagentdb_constant():
    from config import NEWSAGENTDB
    assert NEWSAGENTDB.endswith(".db")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

**Step 3: Write `config.py`**

```python
"""Configuration constants for the newsletter agent."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Timeouts & Rate Limits ---
REQUEST_TIMEOUT = 900  # 15 minutes
SHORT_REQUEST_TIMEOUT = 60
DOMAIN_RATE_LIMIT = 5.0  # seconds between requests to same domain

# --- Concurrency ---
DEFAULT_CONCURRENCY = 12
MAX_CRITIQUE_ITERATIONS = 2

# --- File Paths ---
DOWNLOAD_ROOT = "download"
DOWNLOAD_DIR = os.path.join(DOWNLOAD_ROOT, "sources")
PAGES_DIR = os.path.join(DOWNLOAD_ROOT, "html")
TEXT_DIR = os.path.join(DOWNLOAD_ROOT, "text")
SCREENSHOT_DIR = os.path.join(DOWNLOAD_ROOT, "screenshots")

# --- Database ---
NEWSAGENTDB = "newsletter_agent.db"

# --- Browser ---
FIREFOX_PROFILE_PATH = os.environ.get("FIREFOX_PROFILE_PATH", "")

# --- Content Filtering ---
MIN_TITLE_LEN = 28
DOMAIN_SKIPLIST = ["finbold.com", "philarchive.org"]
IGNORE_LIST = ["Bloomberg", "CNN", "WSJ"]

# --- Model Constants ---
CLAUDE_SONNET = "claude-sonnet-4-5-20250929"
CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
DEFAULT_MODEL = CLAUDE_SONNET

# --- Canonical Topics ---
# 185 topics covering AI/ML domains, companies, people, geographies
CANONICAL_TOPICS = [
    # Policy & Regulation
    "Policy & Regulation",
    "AI Regulation",
    "Executive Orders",
    "Legislation",
    "Copyright",
    "Intellectual Property",
    "Privacy",
    "Surveillance",
    "Antitrust",
    "Export Controls",
    "Open Source Policy",
    # Economics
    "Economics",
    "Labor Market",
    "Automation",
    "Productivity",
    "GDP Impact",
    "Income Inequality",
    "Trade",
    # Safety & Alignment
    "Safety & Alignment",
    "AI Safety",
    "Alignment Research",
    "Interpretability",
    "Red Teaming",
    "Jailbreaking",
    "Guardrails",
    "Bias & Fairness",
    "Hallucination",
    "Existential Risk",
    "AI Ethics",
    # AI/ML Core
    "Language Models",
    "Transformers",
    "Diffusion Models",
    "Reinforcement Learning",
    "Computer Vision",
    "Speech & Audio",
    "Multimodal",
    "Embeddings",
    "Fine-Tuning",
    "Prompt Engineering",
    "RAG",
    "Agents",
    "Tool Use",
    "Function Calling",
    "Code Generation",
    "Reasoning",
    "Chain of Thought",
    "Planning",
    "Memory",
    "Context Windows",
    "Tokenization",
    "Quantization",
    "Distillation",
    "Synthetic Data",
    "Benchmarks",
    "Evaluation",
    "Training",
    "Inference",
    "Scaling Laws",
    "Emergent Abilities",
    "World Models",
    "Robotics",
    "Autonomous Vehicles",
    "Drones",
    # Applications
    "Healthcare",
    "Drug Discovery",
    "Medical Imaging",
    "Clinical AI",
    "Education",
    "Legal AI",
    "Finance AI",
    "Trading",
    "Fraud Detection",
    "Customer Service",
    "Marketing",
    "Content Creation",
    "Gaming",
    "Music",
    "Art & Design",
    "Video Generation",
    "Image Generation",
    "Search",
    "Recommendation Systems",
    "Translation",
    "Summarization",
    "Writing Assistance",
    "Coding Assistants",
    "DevOps",
    "Cybersecurity",
    "Military & Defense",
    "Climate & Energy",
    "Agriculture",
    "Manufacturing",
    "Supply Chain",
    "Scientific Research",
    "Materials Science",
    "Protein Folding",
    "Weather & Climate Modeling",
    # Infrastructure
    "Cloud Computing",
    "Edge Computing",
    "Data Centers",
    "GPUs",
    "TPUs",
    "Custom Silicon",
    "Networking",
    "Semiconductors",
    "NVIDIA",
    "AMD",
    "Intel",
    "TSMC",
    "Qualcomm",
    # Business & Funding
    "Funding",
    "Venture Capital",
    "Mergers & Acquisitions",
    "IPOs",
    "Valuations",
    "Revenue",
    "Partnerships",
    "Enterprise AI",
    "Startups",
    "Layoffs",
    "Hiring",
    # Companies
    "OpenAI",
    "Anthropic",
    "Google",
    "Google DeepMind",
    "Meta",
    "Microsoft",
    "Apple",
    "Amazon",
    "xAI",
    "Perplexity",
    "Mistral",
    "Cohere",
    "Stability AI",
    "Midjourney",
    "Hugging Face",
    "Scale AI",
    "Databricks",
    "Snowflake",
    "Palantir",
    "Tesla",
    "Samsung",
    "ByteDance",
    "Baidu",
    "Alibaba",
    "Tencent",
    # People
    "Sam Altman",
    "Elon Musk",
    "Demis Hassabis",
    "Dario Amodei",
    "Jensen Huang",
    "Satya Nadella",
    "Sundar Pichai",
    "Mark Zuckerberg",
    "Tim Cook",
    "Mustafa Suleyman",
    "Yann LeCun",
    "Ilya Sutskever",
    "Andrej Karpathy",
    # Geographies
    "China",
    "European Union",
    "United Kingdom",
    "Japan",
    "South Korea",
    "India",
    "Taiwan",
    "Israel",
    "Canada",
    "Russia",
    "Middle East",
    "Africa",
    "Latin America",
    "Singapore",
    "Australia",
    # Media & Society
    "Deepfakes",
    "Misinformation",
    "Content Moderation",
    "Social Media",
    "Journalism",
    "Public Opinion",
    "Workforce",
    "Digital Divide",
    "Accessibility",
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add config.py with constants, topics, and model config"
```

---

### Task 3: Write state.py — StepStatus enum and WorkflowStep

**Files:**
- Create: `state.py`
- Create: `tests/test_state.py`

This task builds the first half of state.py: the enum and WorkflowStep dataclass. Task 4 adds WorkflowState and Task 5 adds NewsletterAgentState.

**Step 1: Write the failing test**

Create `tests/test_state.py`:

```python
"""Tests for state.py workflow state management."""
from datetime import datetime
import pytest


class TestStepStatus:
    def test_enum_values(self):
        from state import StepStatus
        assert StepStatus.NOT_STARTED.value == "not_started"
        assert StepStatus.STARTED.value == "started"
        assert StepStatus.COMPLETE.value == "complete"
        assert StepStatus.ERROR.value == "error"
        assert StepStatus.SKIPPED.value == "skipped"


class TestWorkflowStep:
    def test_create_step(self):
        from state import WorkflowStep, StepStatus
        step = WorkflowStep(id="gather_urls", name="Gather URLs", description="Fetch headlines")
        assert step.id == "gather_urls"
        assert step.name == "Gather URLs"
        assert step.status == StepStatus.NOT_STARTED
        assert step.retry_count == 0

    def test_start_step(self):
        from state import WorkflowStep
        step = WorkflowStep(id="test", name="Test", description="desc")
        step.start()
        from state import StepStatus
        assert step.status == StepStatus.STARTED
        assert step.started_at is not None

    def test_complete_step(self):
        from state import WorkflowStep, StepStatus
        step = WorkflowStep(id="test", name="Test", description="desc")
        step.start()
        step.complete("Done")
        assert step.status == StepStatus.COMPLETE
        assert step.completed_at is not None
        assert step.status_message == "Done"

    def test_error_step_increments_retry(self):
        from state import WorkflowStep, StepStatus
        step = WorkflowStep(id="test", name="Test", description="desc")
        step.start()
        step.error("Failed")
        assert step.status == StepStatus.ERROR
        assert step.error_message == "Failed"
        assert step.retry_count == 1
        step.error("Failed again")
        assert step.retry_count == 2

    def test_step_serialization(self):
        from state import WorkflowStep
        step = WorkflowStep(id="test", name="Test", description="desc")
        step.start()
        d = step.model_dump()
        assert d["id"] == "test"
        assert d["status"] == "started"
        restored = WorkflowStep.model_validate(d)
        assert restored.id == step.id
        assert restored.status == step.status
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'state'`

**Step 3: Write the first part of `state.py`**

```python
"""Workflow state management for the newsletter agent."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status of a workflow step."""
    NOT_STARTED = "not_started"
    STARTED = "started"
    COMPLETE = "complete"
    ERROR = "error"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    """A single step in the workflow pipeline."""
    id: str
    name: str
    description: str = ""
    status: StepStatus = StepStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""
    status_message: str = ""
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def start(self) -> None:
        self.status = StepStatus.STARTED
        self.started_at = datetime.now()
        self.error_message = ""

    def complete(self, message: str = "") -> None:
        self.status = StepStatus.COMPLETE
        self.completed_at = datetime.now()
        self.status_message = message

    def error(self, message: str) -> None:
        self.status = StepStatus.ERROR
        self.error_message = message
        self.retry_count += 1

    def __str__(self) -> str:
        elapsed = ""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            elapsed = f" ({delta.total_seconds():.1f}s)"
        return f"[{self.status.value}] {self.name}{elapsed}"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add StepStatus enum and WorkflowStep model"
```

---

### Task 4: Write state.py — WorkflowState base class

**Files:**
- Modify: `state.py` (append)
- Modify: `tests/test_state.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_state.py`:

```python
class TestWorkflowState:
    def _make_state(self):
        from state import WorkflowState
        ws = WorkflowState()
        ws.add_step("step_a", "Step A", "First step")
        ws.add_step("step_b", "Step B", "Second step")
        ws.add_step("step_c", "Step C", "Third step")
        return ws

    def test_add_step_returns_self(self):
        from state import WorkflowState
        ws = WorkflowState()
        result = ws.add_step("s1", "S1", "desc")
        assert result is ws

    def test_get_step(self):
        ws = self._make_state()
        step = ws.get_step("step_b")
        assert step is not None
        assert step.name == "Step B"

    def test_get_step_missing_returns_none(self):
        ws = self._make_state()
        assert ws.get_step("nonexistent") is None

    def test_start_and_complete_step(self):
        from state import StepStatus
        ws = self._make_state()
        ws.start_step("step_a")
        assert ws.get_step("step_a").status == StepStatus.STARTED
        ws.complete_step("step_a", "Done A")
        assert ws.get_step("step_a").status == StepStatus.COMPLETE

    def test_error_step(self):
        from state import StepStatus
        ws = self._make_state()
        ws.start_step("step_a")
        ws.error_step("step_a", "Boom")
        assert ws.get_step("step_a").status == StepStatus.ERROR
        assert ws.get_step("step_a").error_message == "Boom"

    def test_get_current_step(self):
        from state import StepStatus
        ws = self._make_state()
        assert ws.get_current_step().id == "step_a"
        ws.start_step("step_a")
        ws.complete_step("step_a")
        assert ws.get_current_step().id == "step_b"

    def test_all_complete(self):
        ws = self._make_state()
        assert ws.all_complete() is False
        for sid in ["step_a", "step_b", "step_c"]:
            ws.start_step(sid)
            ws.complete_step(sid)
        assert ws.all_complete() is True

    def test_has_errors(self):
        ws = self._make_state()
        assert ws.has_errors() is False
        ws.start_step("step_a")
        ws.error_step("step_a", "fail")
        assert ws.has_errors() is True

    def test_get_completed_steps(self):
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a")
        assert ws.get_completed_steps() == ["step_a"]

    def test_get_progress_percentage(self):
        ws = self._make_state()
        assert ws.get_progress_percentage() == 0.0
        ws.start_step("step_a")
        ws.complete_step("step_a")
        assert abs(ws.get_progress_percentage() - 33.33) < 1.0

    def test_clear_errors(self):
        from state import StepStatus
        ws = self._make_state()
        ws.start_step("step_a")
        ws.error_step("step_a", "fail")
        ws.clear_errors()
        assert ws.get_step("step_a").status == StepStatus.NOT_STARTED

    def test_reset(self):
        from state import StepStatus
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a")
        ws.reset()
        assert all(s.status == StepStatus.NOT_STARTED for s in ws.steps)

    def test_get_status_summary(self):
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a")
        summary = ws.get_status_summary()
        assert summary["complete"] == 1
        assert summary["not_started"] == 2

    def test_serialization_roundtrip(self):
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a", "ok")
        d = ws.model_dump()
        from state import WorkflowState
        restored = WorkflowState.model_validate(d)
        assert len(restored.steps) == 3
        assert restored.get_step("step_a").status_message == "ok"

    def test_get_workflow_status_report(self):
        ws = self._make_state()
        ws.start_step("step_a")
        ws.complete_step("step_a")
        report = ws.get_workflow_status_report("Test Workflow")
        assert "Test Workflow" in report
        assert "complete" in report.lower()
```

**Step 2: Run test to verify new tests fail**

Run: `pytest tests/test_state.py::TestWorkflowState -v`
Expected: FAIL — `ImportError` (WorkflowState not defined yet)

**Step 3: Append WorkflowState to `state.py`**

Append to `state.py`:

```python


class WorkflowState(BaseModel):
    """Base class for managing a multi-step workflow."""
    steps: List[WorkflowStep] = Field(default_factory=list)
    current_step_name: str = ""

    def add_step(self, step_id: str, name: str, description: str = "") -> "WorkflowState":
        self.steps.append(WorkflowStep(id=step_id, name=name, description=description))
        return self

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def start_step(self, step_id: str) -> None:
        step = self.get_step(step_id)
        if step:
            step.start()
            self.current_step_name = step.name

    def complete_step(self, step_id: str, message: str = "") -> None:
        step = self.get_step(step_id)
        if step:
            step.complete(message)

    def error_step(self, step_id: str, error_message: str) -> None:
        step = self.get_step(step_id)
        if step:
            step.error(error_message)

    def get_current_step(self) -> Optional[WorkflowStep]:
        """Return the first step that is not COMPLETE."""
        for step in self.steps:
            if step.status != StepStatus.COMPLETE:
                return step
        return None

    def all_complete(self) -> bool:
        return all(s.status == StepStatus.COMPLETE for s in self.steps)

    def has_errors(self) -> bool:
        return any(s.status == StepStatus.ERROR for s in self.steps)

    def get_completed_steps(self) -> List[str]:
        return [s.id for s in self.steps if s.status == StepStatus.COMPLETE]

    def get_failed_steps(self) -> List[str]:
        return [s.id for s in self.steps if s.status == StepStatus.ERROR]

    def get_progress_percentage(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status == StepStatus.COMPLETE)
        return (done / len(self.steps)) * 100

    def clear_errors(self) -> None:
        for step in self.steps:
            if step.status == StepStatus.ERROR:
                step.status = StepStatus.NOT_STARTED
                step.error_message = ""

    def reset(self) -> None:
        for step in self.steps:
            step.status = StepStatus.NOT_STARTED
            step.started_at = None
            step.completed_at = None
            step.error_message = ""
            step.status_message = ""

    def get_status_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for step in self.steps:
            key = step.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_workflow_status_report(self, title: str = "Workflow") -> str:
        lines = [f"=== {title} ===", f"Progress: {self.get_progress_percentage():.1f}%", ""]
        for i, step in enumerate(self.steps):
            lines.append(f"  Step {i + 1}: {step}")
        summary = self.get_status_summary()
        lines.append("")
        lines.append(f"Summary: {summary}")
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add WorkflowState base class with step management"
```

---

### Task 5: Write state.py — NewsletterAgentState

**Files:**
- Modify: `state.py` (append)
- Modify: `tests/test_state.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_state.py`:

```python
class TestNewsletterAgentState:
    def test_create_with_session_id(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test_20260213")
        assert s.session_id == "test_20260213"

    def test_nine_workflow_steps(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        assert len(s.steps) == 9
        step_ids = [step.id for step in s.steps]
        assert step_ids == [
            "gather_urls",
            "filter_urls",
            "download_articles",
            "extract_summaries",
            "rate_articles",
            "cluster_topics",
            "select_sections",
            "draft_sections",
            "finalize_newsletter",
        ]

    def test_headline_data_starts_empty(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        assert s.headline_data == []

    def test_add_headlines_deduplicates(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        s.add_headlines([
            {"url": "http://a.com", "title": "A"},
            {"url": "http://b.com", "title": "B"},
        ])
        assert len(s.headline_data) == 2
        s.add_headlines([
            {"url": "http://a.com", "title": "A duplicate"},
            {"url": "http://c.com", "title": "C"},
        ])
        assert len(s.headline_data) == 3

    def test_newsletter_section_data_starts_empty(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        assert s.newsletter_section_data == []

    def test_defaults(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        assert s.db_path == "newsletter_agent.db"
        assert s.sources_file == "sources.yaml"
        assert s.max_edits == 2
        assert s.do_download is True

    def test_get_status_returns_dict(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        status = s.get_status()
        assert "headlines" in status
        assert "workflow" in status
        assert status["headlines"] == 0

    def test_serialization_roundtrip(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test_rt")
        s.add_headlines([{"url": "http://x.com", "title": "X"}])
        s.start_step("gather_urls")
        s.complete_step("gather_urls", "Got 1 url")
        d = s.model_dump()
        restored = NewsletterAgentState.model_validate(d)
        assert restored.session_id == "test_rt"
        assert len(restored.headline_data) == 1
        assert restored.get_step("gather_urls").status_message == "Got 1 url"

    def test_get_unique_sources(self):
        from state import NewsletterAgentState
        s = NewsletterAgentState(session_id="test")
        s.add_headlines([
            {"url": "http://a.com", "title": "A", "source": "TechCrunch"},
            {"url": "http://b.com", "title": "B", "source": "TechCrunch"},
            {"url": "http://c.com", "title": "C", "source": "Ars Technica"},
        ])
        sources = s.get_unique_sources()
        assert sources["TechCrunch"] == 2
        assert sources["Ars Technica"] == 1
```

**Step 2: Run test to verify new tests fail**

Run: `pytest tests/test_state.py::TestNewsletterAgentState -v`
Expected: FAIL — `ImportError` (NewsletterAgentState not defined)

**Step 3: Append NewsletterAgentState to `state.py`**

Append to `state.py`:

```python


class NewsletterAgentState(WorkflowState):
    """State for the 9-step newsletter generation workflow."""

    # Session
    session_id: str
    db_path: str = "newsletter_agent.db"
    sources_file: str = "sources.yaml"

    # Data
    headline_data: List[Dict[str, Any]] = Field(default_factory=list)
    newsletter_section_data: List[Dict[str, Any]] = Field(default_factory=list)
    newsletter_title: str = ""
    final_newsletter: str = ""

    # Config
    max_edits: int = 2
    concurrency: int = 12
    do_download: bool = True
    reprocess_since: Optional[datetime] = None

    # Sources
    sources: Dict[str, Any] = Field(default_factory=dict)

    # Topics & Clustering
    cluster_names: List[str] = Field(default_factory=list)
    clusters: Dict[str, List[str]] = Field(default_factory=dict)
    common_topics: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.steps:
            self._initialize_workflow()

    def _initialize_workflow(self) -> None:
        self.add_step("gather_urls", "Gather URLs", "Collect headlines from sources")
        self.add_step("filter_urls", "Filter URLs", "Deduplicate and filter AI-relevant content")
        self.add_step("download_articles", "Download Articles", "Fetch full article content")
        self.add_step("extract_summaries", "Extract Summaries", "Create bullet-point summaries")
        self.add_step("rate_articles", "Rate Articles", "Evaluate quality and importance")
        self.add_step("cluster_topics", "Cluster Topics", "Group articles by theme")
        self.add_step("select_sections", "Select Sections", "Organize into newsletter sections")
        self.add_step("draft_sections", "Draft Sections", "Write section content")
        self.add_step("finalize_newsletter", "Finalize Newsletter", "Combine into final output")

    def add_headlines(self, new_headlines: List[Dict[str, Any]]) -> None:
        """Add headlines with URL-based deduplication."""
        existing_urls = {h.get("url") for h in self.headline_data}
        for h in new_headlines:
            if h.get("url") not in existing_urls:
                self.headline_data.append(h)
                existing_urls.add(h.get("url"))

    def get_unique_sources(self) -> Dict[str, int]:
        """Return source names with article counts."""
        counts: Dict[str, int] = {}
        for h in self.headline_data:
            src = h.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def get_status(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "headlines": len(self.headline_data),
            "sources": len(self.get_unique_sources()),
            "topics": len(self.cluster_names),
            "sections": len(self.newsletter_section_data),
            "workflow": {
                "progress": self.get_progress_percentage(),
                "summary": self.get_status_summary(),
            },
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add NewsletterAgentState with 9-step workflow"
```

---

### Task 6: Write db.py — SQLiteModel base class

**Files:**
- Create: `db.py`
- Create: `tests/test_db.py`

The original db.py has repetitive CRUD boilerplate across 5 models. We'll introduce a `SQLiteModel` base class that handles create_table, insert, update, delete, get, get_all, and upsert generically.

**Step 1: Write the failing test**

Create `tests/test_db.py`:

```python
"""Tests for db.py SQLite models."""
import os
import sqlite3
import pytest

TEST_DB = "test_newsletter.db"


@pytest.fixture(autouse=True)
def clean_db():
    """Remove test DB before and after each test."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


class TestSQLiteModel:
    def test_create_table_and_insert(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://example.com", final_url="http://example.com/final", title="Test", source="rss")
        u.insert(TEST_DB)
        assert u.id is not None

    def test_get_by_id(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="html")
        u.insert(TEST_DB)
        fetched = Url.get(TEST_DB, u.id)
        assert fetched is not None
        assert fetched.initial_url == "http://a.com"

    def test_get_all(self):
        from db import Url
        Url.create_table(TEST_DB)
        Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="s").insert(TEST_DB)
        Url(initial_url="http://b.com", final_url="http://b.com", title="B", source="s").insert(TEST_DB)
        all_urls = Url.get_all(TEST_DB)
        assert len(all_urls) == 2

    def test_update(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="s")
        u.insert(TEST_DB)
        u.title = "Updated"
        u.update(TEST_DB)
        fetched = Url.get(TEST_DB, u.id)
        assert fetched.title == "Updated"

    def test_delete(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="s")
        u.insert(TEST_DB)
        rid = u.id
        u.delete(TEST_DB)
        assert Url.get(TEST_DB, rid) is None

    def test_upsert_inserts_then_updates(self):
        from db import Url
        Url.create_table(TEST_DB)
        u = Url(initial_url="http://a.com", final_url="http://a.com", title="A", source="s")
        u.upsert(TEST_DB)
        assert u.id is not None
        first_id = u.id
        u.title = "Updated"
        u.upsert(TEST_DB)
        assert u.id == first_id
        fetched = Url.get(TEST_DB, first_id)
        assert fetched.title == "Updated"

    def test_unique_constraint_on_initial_url(self):
        from db import Url
        Url.create_table(TEST_DB)
        Url(initial_url="http://same.com", final_url="http://same.com", title="A", source="s").insert(TEST_DB)
        with pytest.raises(sqlite3.IntegrityError):
            Url(initial_url="http://same.com", final_url="http://same.com", title="B", source="s").insert(TEST_DB)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db'`

**Step 3: Write `db.py` with SQLiteModel base and Url model**

```python
"""SQLite persistence layer for the newsletter agent.

Uses dataclasses + raw sqlite3. A SQLiteModel base class eliminates
repetitive CRUD boilerplate across all models.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar

T = TypeVar("T", bound="SQLiteModel")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@dataclass
class SQLiteModel:
    """Base class providing generic CRUD for sqlite3-backed dataclasses.

    Subclasses must define:
        _table_name: ClassVar[str]
        _create_sql: ClassVar[str]  — full CREATE TABLE statement
        _unique_columns: ClassVar[Tuple[str, ...]]  — for upsert conflict detection
    """
    _table_name: ClassVar[str]
    _create_sql: ClassVar[str]
    _unique_columns: ClassVar[Tuple[str, ...]] = ()
    _indexes_sql: ClassVar[List[str]] = []

    id: Optional[int] = field(default=None, repr=False)

    @classmethod
    def create_table(cls, db_path: str) -> None:
        with _connect(db_path) as conn:
            conn.execute(cls._create_sql)
            for idx_sql in cls._indexes_sql:
                conn.execute(idx_sql)

    def _data_columns(self) -> List[str]:
        """Return field names excluding 'id'."""
        return [f.name for f in fields(self) if f.name != "id"]

    def _data_values(self) -> List[Any]:
        """Return field values excluding 'id', converting datetimes to ISO strings."""
        vals = []
        for col in self._data_columns():
            v = getattr(self, col)
            if isinstance(v, datetime):
                v = v.isoformat()
            vals.append(v)
        return vals

    def insert(self, db_path: str) -> None:
        cols = self._data_columns()
        placeholders = ", ".join("?" for _ in cols)
        sql = f"INSERT INTO {self._table_name} ({', '.join(cols)}) VALUES ({placeholders})"
        with _connect(db_path) as conn:
            cur = conn.execute(sql, self._data_values())
            self.id = cur.lastrowid

    def update(self, db_path: str) -> None:
        if self.id is None:
            raise ValueError("Cannot update a record without an id")
        cols = self._data_columns()
        set_clause = ", ".join(f"{c} = ?" for c in cols)
        sql = f"UPDATE {self._table_name} SET {set_clause} WHERE id = ?"
        with _connect(db_path) as conn:
            conn.execute(sql, self._data_values() + [self.id])

    def delete(self, db_path: str) -> None:
        if self.id is None:
            raise ValueError("Cannot delete a record without an id")
        with _connect(db_path) as conn:
            conn.execute(f"DELETE FROM {self._table_name} WHERE id = ?", (self.id,))

    def upsert(self, db_path: str) -> None:
        """Insert or update. If id is set, updates. Otherwise inserts."""
        if self.id is not None:
            self.update(db_path)
        else:
            self.insert(db_path)

    @classmethod
    def _row_to_instance(cls: Type[T], row: sqlite3.Row) -> T:
        col_names = [f.name for f in fields(cls)]
        kwargs = {}
        for col in col_names:
            if col in row.keys():
                kwargs[col] = row[col]
        return cls(**kwargs)

    @classmethod
    def get(cls: Type[T], db_path: str, record_id: int) -> Optional[T]:
        with _connect(db_path) as conn:
            row = conn.execute(
                f"SELECT * FROM {cls._table_name} WHERE id = ?", (record_id,)
            ).fetchone()
        if row is None:
            return None
        return cls._row_to_instance(row)

    @classmethod
    def get_all(cls: Type[T], db_path: str) -> List[T]:
        with _connect(db_path) as conn:
            rows = conn.execute(f"SELECT * FROM {cls._table_name}").fetchall()
        return [cls._row_to_instance(r) for r in rows]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py::TestSQLiteModel -v`
Expected: FAIL — Url is not yet defined. Continue to step 5.

**Step 5: Add Url model to `db.py`**

Append to `db.py`:

```python


@dataclass
class Url(SQLiteModel):
    """URL and headline tracking."""
    _table_name: ClassVar[str] = "urls"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            initial_url TEXT UNIQUE NOT NULL,
            final_url TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            isAI INTEGER NOT NULL DEFAULT 0,
            created_at TEXT
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("initial_url",)

    initial_url: str = ""
    final_url: str = ""
    title: str = ""
    source: str = ""
    isAI: bool = False
    created_at: Optional[datetime] = None
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: add SQLiteModel base class and Url model"
```

---

### Task 7: Write db.py — Article model

**Files:**
- Modify: `db.py` (append)
- Modify: `tests/test_db.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_db.py`:

```python
class TestArticle:
    def test_create_and_retrieve(self):
        from db import Article
        Article.create_table(TEST_DB)
        a = Article(
            final_url="http://example.com/article",
            url="http://example.com/article",
            source="TechCrunch",
            title="AI News",
            domain="example.com",
        )
        a.insert(TEST_DB)
        fetched = Article.get(TEST_DB, a.id)
        assert fetched.title == "AI News"
        assert fetched.domain == "example.com"

    def test_unique_on_final_url(self):
        from db import Article
        Article.create_table(TEST_DB)
        Article(final_url="http://same.com", url="http://same.com", source="s", title="A", domain="same.com").insert(TEST_DB)
        with pytest.raises(sqlite3.IntegrityError):
            Article(final_url="http://same.com", url="http://same.com", source="s", title="B", domain="same.com").insert(TEST_DB)

    def test_topics_conversion(self):
        from db import Article
        assert Article.topics_list_to_string(["AI", "ML", "NLP"]) == "AI,ML,NLP"
        assert Article.topics_string_to_list("AI,ML,NLP") == ["AI", "ML", "NLP"]
        assert Article.topics_string_to_list("") == []
        assert Article.topics_string_to_list(None) == []

    def test_article_defaults(self):
        from db import Article
        a = Article(final_url="http://x.com", url="http://x.com", source="s", title="T", domain="x.com")
        assert a.isAI is False
        assert a.rating == 0.0
        assert a.content_length == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py::TestArticle -v`
Expected: FAIL — `ImportError` (Article not defined)

**Step 3: Append Article to `db.py`**

```python


@dataclass
class Article(SQLiteModel):
    """Full article content and metadata."""
    _table_name: ClassVar[str] = "articles"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            final_url TEXT UNIQUE NOT NULL,
            url TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            published TEXT,
            date TEXT,
            rss_summary TEXT,
            description TEXT,
            summary TEXT,
            short_summary TEXT,
            isAI INTEGER NOT NULL DEFAULT 0,
            status TEXT,
            html_path TEXT,
            text_path TEXT,
            content_length INTEGER NOT NULL DEFAULT 0,
            rating REAL NOT NULL DEFAULT 0.0,
            cluster_label TEXT,
            topics TEXT,
            domain TEXT NOT NULL DEFAULT '',
            site_name TEXT NOT NULL DEFAULT '',
            reputation REAL,
            last_updated TEXT
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("final_url",)

    final_url: str = ""
    url: str = ""
    source: str = ""
    title: str = ""
    published: Optional[datetime] = None
    date: Optional[datetime] = None
    rss_summary: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    short_summary: Optional[str] = None
    isAI: bool = False
    status: Optional[str] = None
    html_path: Optional[str] = None
    text_path: Optional[str] = None
    content_length: int = 0
    rating: float = 0.0
    cluster_label: Optional[str] = None
    topics: Optional[str] = None
    domain: str = ""
    site_name: str = ""
    reputation: Optional[float] = None
    last_updated: Optional[datetime] = None

    @staticmethod
    def topics_list_to_string(topics: List[str]) -> str:
        return ",".join(topics)

    @staticmethod
    def topics_string_to_list(topics_str: Optional[str]) -> List[str]:
        if not topics_str:
            return []
        return [t.strip() for t in topics_str.split(",") if t.strip()]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: add Article model with 22-field schema"
```

---

### Task 8: Write db.py — Site, Newsletter, AgentState models

**Files:**
- Modify: `db.py` (append)
- Modify: `tests/test_db.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_db.py`:

```python
class TestSite:
    def test_create_and_retrieve(self):
        from db import Site
        Site.create_table(TEST_DB)
        s = Site(domain_name="example.com", site_name="Example", reputation=0.8)
        s.insert(TEST_DB)
        fetched = Site.get(TEST_DB, s.id)
        assert fetched.reputation == 0.8

    def test_unique_on_domain(self):
        from db import Site
        Site.create_table(TEST_DB)
        Site(domain_name="x.com", site_name="X", reputation=0.5).insert(TEST_DB)
        with pytest.raises(sqlite3.IntegrityError):
            Site(domain_name="x.com", site_name="X2", reputation=0.6).insert(TEST_DB)


class TestNewsletter:
    def test_create_and_retrieve(self):
        from db import Newsletter
        Newsletter.create_table(TEST_DB)
        n = Newsletter(session_id="s1", date=datetime.now(), final_newsletter="# Hello")
        n.insert(TEST_DB)
        fetched = Newsletter.get(TEST_DB, n.id)
        assert fetched.final_newsletter == "# Hello"


class TestAgentState:
    def test_create_and_retrieve(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        a = AgentState(session_id="s1", step_name="gather_urls", state_data='{"key": "val"}')
        a.insert(TEST_DB)
        fetched = AgentState.get(TEST_DB, a.id)
        assert fetched.state_data == '{"key": "val"}'

    def test_get_by_session_and_step(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="gather_urls", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s1", step_name="filter_urls", state_data='{"n":1}').insert(TEST_DB)
        fetched = AgentState.get_by_session_and_step(TEST_DB, "s1", "filter_urls")
        assert fetched is not None
        assert fetched.state_data == '{"n":1}'

    def test_get_latest_by_session(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="step_a", state_data="a").insert(TEST_DB)
        AgentState(session_id="s1", step_name="step_b", state_data="b").insert(TEST_DB)
        latest = AgentState.get_latest_by_session(TEST_DB, "s1")
        assert latest is not None
        # Should return the one with highest id (most recently inserted)
        assert latest.step_name == "step_b"

    def test_get_all_by_session(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="a", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s1", step_name="b", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s2", step_name="a", state_data="{}").insert(TEST_DB)
        results = AgentState.get_all_by_session(TEST_DB, "s1")
        assert len(results) == 2

    def test_list_sessions(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="a", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s2", step_name="a", state_data="{}").insert(TEST_DB)
        sessions = AgentState.list_sessions(TEST_DB)
        assert set(sessions) == {"s1", "s2"}

    def test_delete_session(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="a", state_data="{}").insert(TEST_DB)
        AgentState(session_id="s1", step_name="b", state_data="{}").insert(TEST_DB)
        count = AgentState.delete_session(TEST_DB, "s1")
        assert count == 2
        assert AgentState.get_all_by_session(TEST_DB, "s1") == []

    def test_unique_session_step_constraint(self):
        from db import AgentState
        AgentState.create_table(TEST_DB)
        AgentState(session_id="s1", step_name="same", state_data="{}").insert(TEST_DB)
        with pytest.raises(sqlite3.IntegrityError):
            AgentState(session_id="s1", step_name="same", state_data="{}").insert(TEST_DB)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py::TestSite tests/test_db.py::TestNewsletter tests/test_db.py::TestAgentState -v`
Expected: FAIL — ImportError

**Step 3: Append Site, Newsletter, AgentState to `db.py`**

```python


@dataclass
class Site(SQLiteModel):
    """Source site reputation tracking."""
    _table_name: ClassVar[str] = "sites"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS sites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_name TEXT UNIQUE NOT NULL,
            site_name TEXT NOT NULL DEFAULT '',
            reputation REAL NOT NULL DEFAULT 0.0
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("domain_name",)

    domain_name: str = ""
    site_name: str = ""
    reputation: float = 0.0


@dataclass
class Newsletter(SQLiteModel):
    """Finalized newsletter storage."""
    _table_name: ClassVar[str] = "newsletters"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS newsletters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            date TEXT NOT NULL,
            final_newsletter TEXT NOT NULL DEFAULT ''
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("session_id",)

    session_id: str = ""
    date: Optional[datetime] = None
    final_newsletter: str = ""


@dataclass
class AgentState(SQLiteModel):
    """Workflow state persistence with session checkpointing."""
    _table_name: ClassVar[str] = "agent_state"
    _create_sql: ClassVar[str] = """
        CREATE TABLE IF NOT EXISTS agent_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            step_name TEXT NOT NULL,
            state_data TEXT NOT NULL DEFAULT '',
            updated_at TEXT,
            UNIQUE(session_id, step_name)
        )
    """
    _unique_columns: ClassVar[Tuple[str, ...]] = ("session_id", "step_name")
    _indexes_sql: ClassVar[List[str]] = [
        "CREATE INDEX IF NOT EXISTS idx_agent_state_session_id ON agent_state(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_agent_state_updated_at ON agent_state(updated_at)",
    ]

    session_id: str = ""
    step_name: str = ""
    state_data: str = ""
    updated_at: Optional[datetime] = None

    @classmethod
    def get_by_session_and_step(cls, db_path: str, session_id: str, step_name: str) -> Optional["AgentState"]:
        with _connect(db_path) as conn:
            row = conn.execute(
                f"SELECT * FROM {cls._table_name} WHERE session_id = ? AND step_name = ?",
                (session_id, step_name),
            ).fetchone()
        if row is None:
            return None
        return cls._row_to_instance(row)

    @classmethod
    def get_latest_by_session(cls, db_path: str, session_id: str) -> Optional["AgentState"]:
        with _connect(db_path) as conn:
            row = conn.execute(
                f"SELECT * FROM {cls._table_name} WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return cls._row_to_instance(row)

    @classmethod
    def get_all_by_session(cls, db_path: str, session_id: str) -> List["AgentState"]:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT * FROM {cls._table_name} WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [cls._row_to_instance(r) for r in rows]

    @classmethod
    def list_sessions(cls, db_path: str) -> List[str]:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT session_id FROM {cls._table_name} ORDER BY session_id"
            ).fetchall()
        return [r["session_id"] for r in rows]

    @classmethod
    def delete_session(cls, db_path: str, session_id: str) -> int:
        with _connect(db_path) as conn:
            cur = conn.execute(
                f"DELETE FROM {cls._table_name} WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_db.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add db.py tests/test_db.py
git commit -m "feat: add Site, Newsletter, and AgentState models"
```

---

### Task 9: Integration — state + db persistence

**Files:**
- Modify: `state.py` (add serialize/load methods)
- Modify: `tests/test_state.py` (add DB integration tests)

This task wires up NewsletterAgentState to persist via AgentState in SQLite.

**Step 1: Write the failing tests**

Append to `tests/test_state.py`:

```python
import os

DB_PATH = "test_state_integration.db"


@pytest.fixture(autouse=True)
def clean_integration_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    yield
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


class TestStateDBIntegration:
    def test_save_and_load_checkpoint(self):
        from state import NewsletterAgentState
        from db import AgentState as AgentStateDB
        AgentStateDB.create_table(DB_PATH)

        s = NewsletterAgentState(session_id="int_test", db_path=DB_PATH)
        s.add_headlines([{"url": "http://a.com", "title": "A", "source": "s"}])
        s.start_step("gather_urls")
        s.complete_step("gather_urls", "Got 1")
        s.save_checkpoint("gather_urls")

        loaded = NewsletterAgentState.load_latest_from_db("int_test", DB_PATH)
        assert loaded is not None
        assert loaded.session_id == "int_test"
        assert len(loaded.headline_data) == 1
        assert loaded.get_step("gather_urls").status_message == "Got 1"

    def test_save_multiple_checkpoints(self):
        from state import NewsletterAgentState
        from db import AgentState as AgentStateDB
        AgentStateDB.create_table(DB_PATH)

        s = NewsletterAgentState(session_id="multi", db_path=DB_PATH)
        s.start_step("gather_urls")
        s.complete_step("gather_urls")
        s.save_checkpoint("gather_urls")

        s.start_step("filter_urls")
        s.complete_step("filter_urls")
        s.save_checkpoint("filter_urls")

        loaded = NewsletterAgentState.load_latest_from_db("multi", DB_PATH)
        assert loaded.is_step_complete("gather_urls")
        assert loaded.is_step_complete("filter_urls")

    def test_load_specific_step(self):
        from state import NewsletterAgentState
        from db import AgentState as AgentStateDB
        AgentStateDB.create_table(DB_PATH)

        s = NewsletterAgentState(session_id="specific", db_path=DB_PATH)
        s.start_step("gather_urls")
        s.complete_step("gather_urls", "step 1 done")
        s.save_checkpoint("gather_urls")

        s.start_step("filter_urls")
        s.complete_step("filter_urls", "step 2 done")
        s.save_checkpoint("filter_urls")

        loaded = NewsletterAgentState.load_from_db("specific", "gather_urls", DB_PATH)
        assert loaded is not None
        assert loaded.get_step("gather_urls").status_message == "step 1 done"
        # filter_urls should NOT be complete in this checkpoint
        from state import StepStatus
        assert loaded.get_step("filter_urls").status != StepStatus.COMPLETE

    def test_list_session_steps(self):
        from state import NewsletterAgentState
        from db import AgentState as AgentStateDB
        AgentStateDB.create_table(DB_PATH)

        s = NewsletterAgentState(session_id="list_test", db_path=DB_PATH)
        s.save_checkpoint("gather_urls")
        s.save_checkpoint("filter_urls")

        steps = NewsletterAgentState.list_session_steps("list_test", DB_PATH)
        assert len(steps) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py::TestStateDBIntegration -v`
Expected: FAIL — `save_checkpoint` not defined

**Step 3: Add persistence methods to `state.py` NewsletterAgentState**

Add these methods to the `NewsletterAgentState` class in `state.py`:

```python
    def is_step_complete(self, step_id: str) -> bool:
        step = self.get_step(step_id)
        return step is not None and step.status == StepStatus.COMPLETE

    def save_checkpoint(self, step_name: str) -> None:
        """Serialize current state and save to DB."""
        import json
        from db import AgentState as AgentStateDB
        from datetime import datetime

        state_json = self.model_dump_json()
        existing = AgentStateDB.get_by_session_and_step(self.db_path, self.session_id, step_name)
        if existing:
            existing.state_data = state_json
            existing.updated_at = datetime.now()
            existing.update(self.db_path)
        else:
            record = AgentStateDB(
                session_id=self.session_id,
                step_name=step_name,
                state_data=state_json,
                updated_at=datetime.now(),
            )
            record.insert(self.db_path)

    @classmethod
    def load_from_db(cls, session_id: str, step_name: str, db_path: str = "newsletter_agent.db") -> Optional["NewsletterAgentState"]:
        """Load state from a specific checkpoint."""
        import json
        from db import AgentState as AgentStateDB

        record = AgentStateDB.get_by_session_and_step(db_path, session_id, step_name)
        if record is None:
            return None
        return cls.model_validate_json(record.state_data)

    @classmethod
    def load_latest_from_db(cls, session_id: str, db_path: str = "newsletter_agent.db") -> Optional["NewsletterAgentState"]:
        """Load the most recent checkpoint for a session."""
        import json
        from db import AgentState as AgentStateDB

        record = AgentStateDB.get_latest_by_session(db_path, session_id)
        if record is None:
            return None
        return cls.model_validate_json(record.state_data)

    @classmethod
    def list_session_steps(cls, session_id: str, db_path: str = "newsletter_agent.db") -> List[Dict[str, Any]]:
        """List all checkpoints for a session."""
        from db import AgentState as AgentStateDB

        records = AgentStateDB.get_all_by_session(db_path, session_id)
        return [
            {"step_name": r.step_name, "updated_at": r.updated_at, "id": r.id}
            for r in records
        ]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add state<->db persistence with checkpoint save/load"
```

---

### Task 10: Final verification and cleanup

**Files:**
- All files created in Tasks 1-9

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (approximately 35-40 tests)

**Step 2: Verify imports work cleanly**

Run: `python -c "from config import CANONICAL_TOPICS, CLAUDE_SONNET; print(f'Topics: {len(CANONICAL_TOPICS)}, Model: {CLAUDE_SONNET}')"`
Expected: `Topics: 185, Model: claude-sonnet-4-5-20250929`

Run: `python -c "from state import NewsletterAgentState; s = NewsletterAgentState(session_id='test'); print(s.get_status())"`
Expected: Dict with headlines=0, workflow progress=0

Run: `python -c "from db import Url, Article, Site, Newsletter, AgentState; print('All models imported OK')"`
Expected: `All models imported OK`

**Step 3: Commit final state**

```bash
git add -A
git commit -m "chore: phase 1 complete — foundation (config, state, db)"
```

---

## Summary

| Task | Description | Tests | Files |
|------|-------------|-------|-------|
| 1 | Repo skeleton | — | .gitignore, requirements.txt, .env.example, __init__.py files |
| 2 | config.py | 8 | config.py, tests/test_config.py |
| 3 | StepStatus + WorkflowStep | 5 | state.py, tests/test_state.py |
| 4 | WorkflowState base | 15 | state.py, tests/test_state.py |
| 5 | NewsletterAgentState | 9 | state.py, tests/test_state.py |
| 6 | SQLiteModel base + Url | 7 | db.py, tests/test_db.py |
| 7 | Article model | 4 | db.py, tests/test_db.py |
| 8 | Site, Newsletter, AgentState | 9 | db.py, tests/test_db.py |
| 9 | State-DB integration | 4 | state.py, tests/test_state.py |
| 10 | Final verification | — | — |
| **Total** | | **~61 tests** | **8 source files** |
