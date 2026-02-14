# ClaudeAgentSDK — Newsletter Agent

Clean rewrite of the newsletter agent (from OpenAIAgentsSDK) using Claude models. See `CC.md` for the full migration plan.

## Project Status

**Phase 1: Foundation — COMPLETE** (config, state, db)
**Phase 2: LLM Layer — COMPLETE** (llm.py with multi-vendor support)
**Phase 3: Library modules — COMPLETE** (scrape, fetch, dedupe, rating, cluster)
**Phase 4: Bash-based steps — COMPLETE** (gather_urls, filter_urls, download_articles, rate_articles, cluster_topics)
**Phase 5: MCP tools — COMPLETE** (summarize, select_sections, draft_sections, finalize, FastMCP server)
- Phase 6: Agent orchestrator — not started

## Architecture

Hybrid tool pattern: heavy data-processing runs as CLI scripts (steps/), LLM-centric work runs as MCP tools (tools/). All state lives in SQLite. The agent never holds article data in its context window.

## Key Files

```
config.py      — Constants, 126 canonical topics, model IDs, paths, timeouts
state.py       — StepStatus, WorkflowStep, WorkflowState, NewsletterAgentState (Pydantic v2)
db.py          — SQLiteModel base class + 5 models: Url, Article, Site, Newsletter, AgentState
llm.py         — Multi-vendor LLM wrapper (Anthropic, OpenAI, Gemini) with structured output, batch processing, retry logic
prompts.py     — All 23 LLM prompt templates (system/user/model/reasoning_effort)
lib/scrape.py  — Camoufox browser automation, rate limiting, text extraction
lib/fetch.py   — Source processor (RSS/HTML/API from sources.yaml)
lib/dedupe.py  — Embedding-based duplicate detection (cosine similarity)
lib/rating.py  — Composite rating formula + Bradley-Terry battles
lib/cluster.py — HDBSCAN + Optuna + UMAP + Claude cluster naming
steps/         — CLI step scripts (gather_urls, filter_urls, download_articles, rate_articles, cluster_topics)
tools/server.py      — FastMCP server registering 4 MCP tools
tools/summarize.py   — Step 4: Extract article summaries via LLM
tools/select_sections.py — Step 7: Organize articles into newsletter sections
tools/draft_sections.py  — Step 8: Draft sections with critique-optimize loop
tools/finalize.py    — Step 9: Assemble, polish, deliver final newsletter
tools/models.py      — Pydantic output models for structured LLM responses
tools/email_sender.py — Gmail delivery with HTML conversion
CC.md          — Full migration plan with all phases
```

## Running Tests

```bash
# Activate venv
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_state.py -v

# Run a specific test class or method
pytest tests/test_db.py::TestAgentState -v
pytest tests/test_state.py::TestNewsletterAgentState::test_nine_workflow_steps -v
```

All tests run from the project root. The venv at `.venv/` has all deps installed.

## Writing Tests

Tests live in `tests/` and follow these conventions:

- **File naming**: `tests/test_<module>.py` mirrors `<module>.py`
- **Class grouping**: Group related tests in classes (`class TestWorkflowStep:`, `class TestArticle:`)
- **Imports inside test functions**: Use `from state import ...` inside each test method, not at module level. This matches the existing pattern and avoids import-order issues.
- **DB tests use fixtures**: Any test touching SQLite uses an `autouse` fixture that creates/removes a temp DB:

```python
TEST_DB = "test_<purpose>.db"

@pytest.fixture(autouse=True)
def clean_db():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
```

- **DB tests must call `create_table()` first**: Every test that uses a model must call `Model.create_table(TEST_DB)` before inserting.
- **Test SQLite integrity**: Use `pytest.raises(sqlite3.IntegrityError)` to verify unique constraints.
- **TDD flow**: Write failing test → implement → verify pass → commit.

## Dependencies

Python 3.11+. Key packages: pydantic>=2.0, anthropic, openai (embeddings only), tenacity, pytest. Full list in `requirements.txt`.

## Models

- `CLAUDE_SONNET = "claude-sonnet-4-5-20250929"` — batch work (default)
- `CLAUDE_HAIKU = "claude-haiku-4-5-20251001"` — fast tasks
- OpenAI `text-embedding-3-large` — embeddings only

## Database

SQLite with WAL mode. 5 tables: `urls`, `articles`, `sites`, `newsletters`, `agent_state`. All models inherit from `SQLiteModel` which provides generic CRUD (insert, update, delete, get, get_all, upsert). `AgentState` has additional query methods for session management.

## State Management

`NewsletterAgentState` (Pydantic BaseModel) tracks the 9-step workflow:
1. gather_urls → 2. filter_urls → 3. download_articles → 4. extract_summaries → 5. rate_articles → 6. cluster_topics → 7. select_sections → 8. draft_sections → 9. finalize_newsletter

Checkpointed to SQLite via `save_checkpoint(step_name)` / `load_latest_from_db(session_id)`.
