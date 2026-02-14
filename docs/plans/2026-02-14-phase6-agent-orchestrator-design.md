# Phase 6: Agent Orchestrator — Design Document

## Overview

Phase 6 adds the agent orchestrator that ties together the 5 Bash-based steps (Phase 4) and 4 MCP tools (Phase 5) into a fully autonomous newsletter generation pipeline. Claude Code serves as the agent runtime, invoked by a thin Python launcher.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  agent.py (Python launcher)                              │
│  - Parses CLI args (--session, --resume, --model, etc.)  │
│  - Generates session_id if not resuming                  │
│  - Ensures MCP server is registered                      │
│  - Invokes: claude -p <prompt> --append-system-prompt    │
│      --dangerously-skip-permissions                      │
│      --model sonnet --max-turns 50                       │
└────────────────────┬─────────────────────────────────────┘
                     │ subprocess
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Claude Code (single agentic session)                    │
│  - System prompt: 9-step workflow, tool usage guide      │
│  - User prompt: "Run newsletter workflow for session X"  │
│                                                          │
│  Tools available:                                        │
│  ┌────────────────┐  ┌─────────────────────────────────┐ │
│  │ Bash tool       │  │ MCP tools (via FastMCP server)  │ │
│  │                 │  │                                 │ │
│  │ Step 1: gather  │  │ Step 4: extract_summaries       │ │
│  │ Step 2: filter  │  │ Step 7: select_sections         │ │
│  │ Step 3: download│  │ Step 8: draft_sections          │ │
│  │ Step 5: rate    │  │ Step 9: finalize_newsletter     │ │
│  │ Step 6: cluster │  │                                 │ │
│  │                 │  │ + check_workflow_status          │ │
│  └────────────────┘  └─────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  SQLite (newsletter_agent.db)                            │
│  - agent_state: workflow checkpoints per session/step    │
│  - urls, articles, sites, newsletters                    │
└──────────────────────────────────────────────────────────┘
```

## Approach: Single Agentic Session

One `claude -p` invocation runs the entire 9-step workflow. Claude Code autonomously calls Bash for data-heavy steps and MCP tools for LLM-centric steps, checking workflow state between steps.

**Why this approach:**
- Simplest implementation — the system prompt does the heavy lifting
- Claude Code's built-in Bash + MCP support handles everything
- Token cost for a single run is manageable (~$2-5 with Sonnet)
- SQLite checkpointing handles the failure/resume case

**Alternatives considered:**
- Step-per-session orchestration (cheaper but loses cross-step reasoning)
- Claude Agent SDK Python loop (more control but more complex)

## System Prompt

The system prompt is interpolated with `{session_id}` by agent.py before passing to Claude Code via `--append-system-prompt`.

```
You are a newsletter agent that generates a weekly AI news digest.
You execute a 9-step workflow, checking status between steps.

## Workflow Steps

1. gather_urls     — Bash: python steps/gather_urls.py --session {session_id}
2. filter_urls     — Bash: python steps/filter_urls.py --session {session_id}
3. download_articles — Bash: python steps/download_articles.py --session {session_id}
4. extract_summaries — MCP tool: extract_summaries(session_id="{session_id}")
5. rate_articles   — Bash: python steps/rate_articles.py --session {session_id}
6. cluster_topics  — Bash: python steps/cluster_topics.py --session {session_id}
7. select_sections — MCP tool: select_sections(session_id="{session_id}")
8. draft_sections  — MCP tool: draft_sections(session_id="{session_id}")
9. finalize_newsletter — MCP tool: finalize_newsletter(session_id="{session_id}")

## Execution Rules

1. First, call the check_workflow_status MCP tool to see which steps are complete.
2. Execute steps in order. Skip any step already marked "complete".
3. After each Bash step, check the JSON output. If "status": "error", STOP
   and report the error. Do not retry.
4. After each MCP tool, check the returned JSON. If "status": "error", STOP
   and report the error.
5. After all 9 steps complete, report the final status.

## Important

- Always pass --session {session_id} to Bash steps.
- Always pass session_id="{session_id}" to MCP tools.
- Bash steps may take several minutes. Be patient.
- Do NOT read or modify article data directly. All data flows through
  the DB and tools.
- Do NOT attempt to fix errors yourself. Stop and report them.
```

## Python Launcher (`agent.py`)

Thin launcher (~80 lines) that:

1. Parses CLI arguments
2. Generates or validates session_id
3. Registers MCP server with Claude Code (idempotent)
4. Interpolates session_id into system prompt
5. Invokes `claude -p` via subprocess

### CLI Interface

```bash
# Fresh run (session_id = newsletter_YYYYMMDD)
python agent.py

# Explicit session
python agent.py --session newsletter_20260214

# Resume a failed run
python agent.py --resume --session newsletter_20260214

# Different model or turn limit
python agent.py --model opus --max-turns 100
```

### Key Flags Passed to Claude Code

- `--append-system-prompt` — preserves Claude Code's built-in capabilities
- `--dangerously-skip-permissions` — fully autonomous, no approval prompts
- `--model sonnet` — configurable, defaults to Sonnet
- `--max-turns 50` — enough for 9 steps plus status checks (~20 tool calls)

### Resume Flow

- `--resume` requires `--session`
- Changes user message to: "Resume the newsletter workflow for session X. Check status and continue from the next incomplete step."
- `check_workflow_status` tool returns step completion state from SQLite
- Claude Code skips completed steps and continues from the next incomplete one

## MCP Server Configuration

### Registration

agent.py registers the FastMCP server automatically before each run:

```python
subprocess.run([
    "claude", "mcp", "add",
    "--transport", "stdio",
    "newsletter", "--",
    "python", "tools/server.py"
], check=True)
```

`claude mcp add` is idempotent — re-adding overwrites the existing config.

### New Tool: `check_workflow_status`

Added to `tools/server.py`. Returns JSON with each step's status and overall progress:

```json
{
  "status": "ok",
  "progress": "44%",
  "next_step": "extract_summaries",
  "steps": {
    "gather_urls": {"status": "complete", "message": "Gathered 234 URLs"},
    "filter_urls": {"status": "complete", "message": "Filtered to 89 AI articles"},
    "download_articles": {"status": "complete", "message": "Downloaded 72 articles"},
    "extract_summaries": {"status": "not_started", "message": ""},
    ...
  }
}
```

For new sessions (no existing state):

```json
{
  "status": "new_session",
  "message": "No existing state. Start from step 1.",
  "steps": {}
}
```

## Error Handling

- **Manual intervention**: On error, Claude Code stops and reports. User investigates, then resumes with `--resume`.
- **State is always checkpointed**: Even failed steps save state to SQLite (with error status), so resume knows where to pick up.
- **No automatic retries**: The system prompt explicitly says "Do not retry."

## File Changes

```
Modified:
  tools/server.py    — Add check_workflow_status tool
  CLAUDE.md          — Update project status

New:
  agent.py           — Python launcher (~80 lines)
```

## Testing

- **Unit test for `check_workflow_status`**: Mock state, verify JSON output for new session, partial completion, and full completion cases
- **Unit test for `agent.py`**: Verify CLI argument parsing and correct `claude` command construction (mock subprocess.run)
- **End-to-end**: `python agent.py` with real data (manual verification)
