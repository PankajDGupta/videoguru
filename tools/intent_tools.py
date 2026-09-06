"""Tools for intent capture and session theme management in VideoGuru."""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def record_theme(theme: str, tool_context: ToolContext) -> str:
    """Store the user's video theme or main highlight in session state and initiate handoff to the Ingestion Agent.

    Args:
        theme: The central theme, narrative focus, or main highlight described by the creator.
        tool_context: ADK ToolContext providing access to session state and agent routing actions.

    Returns:
        Confirmation message summarizing the recorded theme and indicating handoff to the Ingestion Agent.
    """
    if not isinstance(theme, str):
        theme = str(theme)

    cleaned_theme = theme.strip()
    if not cleaned_theme:
        return "Theme cannot be empty. Please provide a valid theme or highlight for the video."

    # Store in session state under key 'theme'
    tool_context.state["theme"] = cleaned_theme

    # Initiate handoff to Ingestion Agent
    tool_context.actions.transfer_to_agent = "ingestion_agent"

    logger.info("Recorded theme '%s' in session state and initiated handoff to ingestion_agent.", cleaned_theme)
    return (
        f"Theme successfully captured: '{cleaned_theme}'. "
        "Session state updated under key 'theme'. Initiating handoff to ingestion_agent (Ingestion Agent)."
    )


def get_theme_from_state(state: dict[str, Any]) -> Optional[str]:
    """Extract theme from session state dictionary if present.

    Args:
        state: State dictionary from session.

    Returns:
        Theme string if found, otherwise None.
    """
    return state.get("theme")
