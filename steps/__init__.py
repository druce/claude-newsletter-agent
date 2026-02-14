"""Step runner utility for newsletter agent CLI scripts."""
from __future__ import annotations

import asyncio
import inspect
import logging
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

        return {"status": "success", "message": str(message)}

    except Exception as e:
        logger.error("Step %s failed: %s", step_name, e)
        state.error_step(step_name, str(e))
        state.save_checkpoint(step_name)

        return {"status": "error", "error": str(e)}
