# Newsletter Agent: Claude Agent SDK Migration Plan

## Context

Rewrite the newsletter agent system (currently built on OpenAI Agents SDK in `/Users/drucev/projects/OpenAIAgentsSDK`) using the Claude Agent SDK. The new repo lives at `/Users/drucev/projects/ClaudeAgentSDK`. This is a **clean rewrite** with the same 9-step workflow but cleaner architecture, using Claude models for all LLM calls.

## Architecture Overview

**Hybrid tool pattern**: Heavy data-processing steps run as standalone Python CLI scripts (invoked via Bash tool), while LLM-centric steps run as in-process MCP tools. This keeps the agent's context window lean — Bash scripts return short status summaries, not full article data.

```
┌─────────────────────────────────────────────────────┐
│  Claude Agent (orchestrator)                        │
│  - System prompt describes 9-step workflow          │
│  - Calls Bash scripts for data-heavy steps          │
│  - Calls MCP tools for LLM-centric steps            │
│  - Reads workflow status from SQLite between steps   │
├─────────────────────────────────────────────────────┤
│  Bash-based steps (data-heavy, context-efficient):  │
│    1. gather_urls    - fetch RSS/HTML/API sources    │
│    2. filter_urls    - dedupe, filter AI-relevant    │
│    3. download_articles - Playwright article scrape  │
│    5. rate_articles  - Bradley-Terry scoring         │
│    6. cluster_topics - HDBSCAN + embeddings          │
├─────────────────────────────────────────────────────┤
│  MCP tools (LLM-centric, inline reasoning):         │
│    4. extract_summaries - Claude summarizes articles │
│    7. select_sections   - Claude organizes sections  │
│    8. draft_sections    - Claude writes newsletter   │
│    9. finalize_newsletter - Claude produces output   │
├─────────────────────────────────────────────────────┤
│  Shared infrastructure:                             │
│    - SQLite state DB (session_id + step checkpoints) │
│    - sources.yaml (same format as current)           │
│    - Anthropic Python SDK for all LLM calls          │
└─────────────────────────────────────────────────────┘
```

## File Structure

```
/Users/drucev/projects/ClaudeAgentSDK/
├── agent.py                  # Main orchestrator - ClaudeSDKClient setup + run
├── config.py                 # Constants, topics, model config
├── state.py                  # WorkflowState + NewsletterState (Pydantic)
├── db.py                     # SQLite ORM (AgentState, Article, Url, Newsletter)
├── llm.py                    # Anthropic API wrapper (replaces OpenAI calls)
│
├── steps/                    # Bash-based CLI step scripts
│   ├── __init__.py
│   ├── gather_urls.py        # Step 1: python steps/gather_urls.py --session X
│   ├── filter_urls.py        # Step 2: python steps/filter_urls.py --session X
│   ├── download_articles.py  # Step 3: python steps/download_articles.py --session X
│   ├── rate_articles.py      # Step 5: python steps/rate_articles.py --session X
│   └── cluster_topics.py     # Step 6: python steps/cluster_topics.py --session X
│
├── tools/                    # MCP tools (in-process)
│   ├── __init__.py
│   ├── server.py             # MCP server setup with @tool decorators
│   ├── summarize.py          # Step 4: extract_summaries tool
│   ├── select_sections.py    # Step 7: select_sections tool
│   ├── draft_sections.py     # Step 8: draft_sections tool
│   └── finalize.py           # Step 9: finalize_newsletter tool
│
├── lib/                      # Shared library code
│   ├── __init__.py
│   ├── fetch.py              # RSS/HTML/API fetcher (async, Playwright)
│   ├── scrape.py             # Playwright HTML scraping
│   ├── cluster.py            # HDBSCAN clustering + embeddings
│   ├── rating.py             # Bradley-Terry article scoring
│   └── dedupe.py             # Duplicate detection
│
├── sources.yaml              # Source configs (copy from current repo)
├── requirements.txt          # Dependencies
├── .env.example              # Environment variable template
└── tests/
    ├── test_state.py
    ├── test_fetch.py
    └── test_steps.py
```

## Implementation Steps

### Phase 1: Foundation (state, db, config)

**1.1 Create repo skeleton**
- `mkdir -p` all directories
- `git init`
- Create `.gitignore`, `requirements.txt`

**1.2 `config.py`** — Clean rewrite of current `config.py`
- Keep: CANONICAL_TOPICS, directory constants, domain lists, timeouts
- Remove: Langfuse config, OpenAI model family mapping
- Add: Anthropic model constants (`claude-sonnet-4-5-20250929` for batch work, `claude-haiku-4-5-20251001` for fast tasks)

**1.3 `state.py`** — Clean rewrite of `newsletter_state.py`
- Keep the WorkflowState/WorkflowStep/StepStatus pattern (it's clean)
- Keep NewsletterAgentState fields and serialize/load methods
- Remove: migration code, emoji formatting, pandas dependency in state
- Store headline_data as list of dicts only (DataFrame conversion happens in processing code, not state)

**1.4 `db.py`** — Simplify current `db.py`
- Keep: AgentState (essential for checkpointing)
- Keep: Article, Url tables (needed for dedup and tracking)
- Simplify: Remove verbose boilerplate, use a base class pattern

### Phase 2: LLM Layer

**2.1 `llm.py`** — New LLM wrapper

- should support multiple LLM vendors more cleanly than existing llm.py, while retaining existing functionality

- in most cases we will construct an agent class and call it on a dataframe:
  `df["ai_related"] = await agent.filter_dataframe(df[['headline']]`
  we should also support calling on a list of strings or dicts, substituting them into the prompt and getting a structured output with a defined pydantic class

  `retlist = await agent.filter_list(l)`
  and support a simple call on a 
  `retstr = await agent.prompt_dict(mydict)`

- should have an llm model class supporting e.g. Claude Sonnet 4.5, GPT-5-2-mini etc. The model class should have sufficient information to resolve to an API call. it should indicate if the model supports e.g. reasoning effort, returning logprobs. To make it easy to switch between models, calls should accept a numeric value for the reasoning effort from 1 to 10. 
  0   none , or 0 tokens

  2. Minimal or 1000
     4 low or 2,000
     6 medium or 4000
     8 high or 8000
     10 xhigh or 16000

- for anthropic Use `anthropic.AsyncAnthropic` client . use a reasonable inheritence methodology to allow the a virtual base class to work with multiple models via polymorphism in sub classes 

- Structured output via tool_use (Anthropic's approach to structured output)

- Batch processing with async semaphore concurrency

- Retry logic with tenacity . In the base class, should support fatal exceptions and temporary exceptions. LLM-specific derived class should map relevant exception types to fatal exceptions like malformed queries, and temporary exceptions like exceeded rate limit and need to slow down and retry

### Phase 3: Library modules (lib/)

**3.1 `lib/fetch.py`** — Rewrite of `fetch.py`
- high level class to process sources in sources.yaml and save to download/sources using scrape.py
- Same async fetcher pattern with aiohttp + Playwright
- Same source type routing (rss/html/rest)
- Cleaner class interface, better error handling

**3.2 `lib/scrape.py`** — Rewrite of `scrape.py`
- Low level Playwright browser automation - get a remote url and save 
- when downloading individual news pages we save html in download/html, and normalize to download/text . we support returning meta data .
- Same configurable scrolling, delays, patterns
- Firefox profile support
- consider ways to make scraping more robust and less likely to be blocked, proxy services etc.

**3.3 `lib/cluster.py`** — Rewrite of `do_cluster.py`
- HDBSCAN clustering + UMAP dimensionality reduction
- Replace OpenAI embeddings with Anthropic's voyage embeddings (or keep OpenAI embeddings as they're best-in-class — use `voyageai` or `openai` for embeddings only)
- Topic naming via Claude instead of GPT

**3.4 `lib/rating.py`** — Rewrite of `do_rating.py`
- Bradley-Terry algorithm (choix library)
- Swiss pairing for efficient comparisons
- Claude for pairwise article comparison instead of GPT

**3.5 `lib/dedupe.py`** — Rewrite of `do_dedupe.py`
- URL normalization and deduplication
- Title similarity matching

### Phase 4: Bash-based steps (steps/)

Each script follows this pattern:
```python
#!/usr/bin/env python3
"""Step N: Description. Run via: python steps/step_name.py --session SESSION_ID"""
import argparse, sys, json
from state import NewsletterAgentState

def main(session_id: str, db_path: str = "newsletter_agent.db"):
    state = NewsletterAgentState(session_id=session_id, db_path=db_path)
    loaded = state.load_latest_from_db()
    if loaded:
        state = loaded

    state.start_step("step_name")
    try:
        # ... do work, modify state ...
        state.complete_step("step_name", message="Processed N articles")
    except Exception as e:
        state.error_step("step_name", str(e))
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

    state.save_checkpoint("step_name")
    # Print JSON summary for agent to read
    print(json.dumps({"status": "success", "articles": N, "message": "..."}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--db", default="newsletter_agent.db")
    args = parser.parse_args()
    main(args.session, args.db)
```

**4.1 `steps/gather_urls.py`** — Fetches from all sources in sources.yaml
**4.2 `steps/filter_urls.py`** — Deduplication + AI relevance filtering
**4.3 `steps/download_articles.py`** — Playwright article content download
**4.4 `steps/rate_articles.py`** — Bradley-Terry scoring via Claude
**4.5 `steps/cluster_topics.py`** — HDBSCAN clustering + topic naming

### Phase 5: MCP tools (tools/)

**5.1 `tools/server.py`** — MCP server setup
```python
from claude_agent_sdk import create_sdk_mcp_server, tool
# Register all tools, export server object
```

**5.2 `tools/summarize.py`** — Extract summaries (Step 4)
- Takes session_id, loads state from DB
- For each article without summary, calls Claude to generate 3 bullet points
- Writes results back to DB
- Returns summary count to agent

**5.3 `tools/select_sections.py`** — Select sections (Step 7)
- Loads clustered + rated articles from DB
- Claude decides which clusters become newsletter sections
- Assigns articles to sections, prunes low-quality ones

**5.4 `tools/draft_sections.py`** — Draft sections (Step 8)
- For each section, Claude writes newsletter copy
- Uses article summaries, ratings, and cluster context

**5.5 `tools/finalize.py`** — Finalize newsletter (Step 9)
- Claude combines sections into final newsletter
- Generates title, intro, section transitions
- Saves to DB and returns markdown

### Phase 6: Agent orchestrator

**6.1 `agent.py`** — Main entry point
```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    allowed_tools=["Bash", "Read"],  # + MCP tools
    mcp_servers={"newsletter": tools_server},
    system_prompt=SYSTEM_PROMPT,
    permission_mode="acceptEdits",
    max_budget_usd=5.0,
)

# System prompt tells Claude:
# 1. You are a newsletter agent running a 9-step workflow
# 2. Check workflow status: python steps/check_status.py --session X
# 3. For each incomplete step, run the appropriate script/tool
# 4. After each step, verify success from JSON output
# 5. Steps must run in order: gather -> filter -> download -> summarize -> rate -> cluster -> select -> draft -> finalize
```

### Phase 7: Testing & verification

**7.1** Unit tests for state serialization
**7.2** Unit tests for individual step scripts (with mock data)
**7.3** Integration test: run full workflow end-to-end

## Key Design Decisions

1. **Embeddings**: Keep OpenAI `text-embedding-3-large` for embeddings (best quality). Only switch LLM calls to Claude. This means `openai` stays as a dependency but only for embeddings.

2. **State flow**: All state lives in SQLite. Bash scripts read state -> process -> write state -> print JSON summary. MCP tools do the same but in-process. The agent never holds article data in its context window.

3. **Session management**: Each run gets a `session_id` (e.g., `newsletter_20260213`). All scripts accept `--session` flag. Agent creates session at start and passes it to every step.

4. **Error recovery**: If a step fails, the agent sees the error in JSON output and can retry or skip. State checkpoints after each step enable resume.

5. **No Langfuse initially**: Skip tracing/prompt management. Can add later as a hook.

## Verification

1. **Unit**: `pytest tests/` — state serialization, DB operations, step script parsing
2. **Individual steps**: Run each step script manually with test data
3. **End-to-end**: `python agent.py` — full newsletter generation
4. **Context efficiency**: Verify agent context stays small by checking that Bash outputs are concise JSON summaries
