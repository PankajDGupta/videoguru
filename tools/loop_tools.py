"""Loop Control Tools for VideoGuru (SPEC-011 & SPEC-012).

Provides ADK tools for managing the autonomous LoopAgent editing cycle:
- exit_loop: Breaks the autonomous loop when quality benchmarks are met, advancing
  the pipeline to Phase IV Human Review (sets tool_context.actions.escalate = True).
- append_to_state: Appends critical revision feedback to session state, allowing the
  Curation Agent to ingest feedback on subsequent iterations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def exit_loop(tool_context: Optional[Any] = None) -> str:
    """ADK Tool: Exit the autonomous editing loop when quality benchmarks are satisfied.

    Sets tool_context.actions.escalate = True to break the LoopAgent cycle and advances
    session state to Phase IV Human Review.

    Args:
        tool_context: ADK ToolContext providing access to state and actions.

    Returns:
        Confirmation message that the loop has been broken.
    """
    if tool_context is not None:
        # Signal ADK LoopAgent to break the loop
        if hasattr(tool_context, "actions") and tool_context.actions is not None:
            tool_context.actions.escalate = True
            tool_context.actions.skip_summarization = True

        # Update session state flags
        if hasattr(tool_context, "state") and tool_context.state is not None:
            try:
                tool_context.state["loop_status"] = "exited"
                tool_context.state["exit_loop_invoked"] = True
            except Exception as exc:
                logger.warning("Could not mutate tool_context.state directly: %s", exc)
            logger.info("exit_loop tool invoked: LoopAgent break requested via escalate=True.")
    else:
        logger.warning("exit_loop called without tool_context.")

    return (
        "Autonomous editing loop exited successfully. The Edit Decision List satisfies all "
        "algorithmic and human-centric standards and is ready for Phase IV Human Review."
    )


def append_to_state(
    key: str,
    value: Any,
    tool_context: Optional[Any] = None,
) -> str:
    """ADK Tool: Append a feedback item or record to a list in session state.

    Used by the Critic Agent to store actionable revision feedback (e.g. key='revision_feedback')
    so that CurationAgent can adjust the edit on subsequent iterations.

    Args:
        key: The state dictionary key to append to (e.g. 'revision_feedback').
        value: The feedback message or data object to append.
        tool_context: ADK ToolContext providing access to state.

    Returns:
        Status message confirming the append operation.
    """
    if tool_context is not None and hasattr(tool_context, "state") and tool_context.state is not None:
        state = tool_context.state
        try:
            current_val = state.get(key) if hasattr(state, "get") else None
            if current_val is None:
                state[key] = [value]
            elif isinstance(current_val, list):
                # Copy or append
                new_list = list(current_val)
                new_list.append(value)
                state[key] = new_list
            else:
                state[key] = [current_val, value]

            state["loop_status"] = "iterating"
            count = len(state[key]) if isinstance(state[key], list) else 1
            logger.info(
                "append_to_state: Appended revision entry to state['%s'] (total items: %d)",
                key,
                count,
            )
            return f"Successfully appended feedback to session state under '{key}' (entry #{count})."
        except Exception as exc:
            logger.warning("Failed to append to state['%s']: %s", key, exc)
            return f"Warning: Failed to append to state: {exc}"
    else:
        logger.warning("append_to_state called without tool_context. State not modified.")
        return f"Warning: append_to_state executed without tool_context. Feedback for '{key}' not persisted."
