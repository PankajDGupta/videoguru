"""VideoGuru Resilience and Graceful Degradation Helpers (SPEC-029).

Provides utilities for safe external call execution, recording non-fatal
warnings in session state, and executing fallback strategies.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Callable, Optional, TypeVar, Union

from services.exceptions import VideoGuruError

logger = logging.getLogger("videoguru.resilience")

T = TypeVar("T")


def record_session_warning(
    state: Any,
    warning_message: str,
    category: str = "general",
    details: Optional[dict[str, Any]] = None,
) -> None:
    """Record a non-fatal warning in session state for UI and user visibility.

    Args:
        state: ADK session state (dict-like or object with state attribute).
        warning_message: User-friendly or diagnostic warning message.
        category: Warning category ('rendering', 'captioning', 'ducking', 'api').
        details: Optional technical details.
    """
    warning_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "message": warning_message,
        "details": details or {},
    }

    target_state = state
    if hasattr(state, "state") and isinstance(state.state, dict):
        target_state = state.state

    if isinstance(target_state, dict):
        warnings_list = target_state.setdefault("warnings", [])
        if isinstance(warnings_list, list):
            warnings_list.append(warning_entry)
        logger.warning(
            "Recorded session warning [%s]: %s",
            category,
            warning_message,
            extra={"event": "session_warning", "warning_entry": warning_entry},
        )
    elif hasattr(target_state, "__getitem__") and hasattr(target_state, "__setitem__"):
        try:
            warnings_list = target_state.get("warnings", [])
            warnings_list.append(warning_entry)
            target_state["warnings"] = warnings_list
        except Exception as exc:
            logger.debug("Failed to record warning in state: %s", exc)


def record_session_error(
    state: Any,
    error: Union[Exception, str],
    category: str = "general",
) -> None:
    """Record a fatal or handled error diagnostic in session state.

    Args:
        state: ADK session state.
        error: Exception or error message string.
        category: Error classification category.
    """
    error_msg = str(error)
    user_msg = getattr(error, "user_message", error_msg)

    error_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "error_type": error.__class__.__name__ if isinstance(error, Exception) else "Error",
        "message": error_msg,
        "user_message": user_msg,
        "details": getattr(error, "details", {}),
    }

    target_state = state
    if hasattr(state, "state") and isinstance(state.state, dict):
        target_state = state.state

    if isinstance(target_state, dict):
        target_state["error"] = error_msg
        target_state["error_details"] = error_data
        logger.error(
            "Recorded session error [%s]: %s",
            category,
            error_msg,
            extra={"event": "session_error", "error_data": error_data},
        )


def safe_execute(
    action: Callable[[], T],
    fallback: Optional[Callable[[Exception], T]] = None,
    default_value: Optional[T] = None,
    category: str = "general",
    state: Optional[Any] = None,
    warning_message: Optional[str] = None,
) -> T:
    """Execute a callable with fallback and automatic session warning recording.

    Args:
        action: The primary function to execute.
        fallback: Optional callable receiving the exception to produce a fallback value.
        default_value: Constant fallback value if fallback callable is not provided.
        category: Error category for logging and warning tracking.
        state: Optional session state where non-fatal warning is recorded on fallback.
        warning_message: Optional custom user-friendly warning message.

    Returns:
        The result of action() or fallback/default_value on failure.
    """
    try:
        return action()
    except Exception as exc:
        msg = warning_message or str(exc)
        logger.warning(
            "Safe execute caught exception in '%s' (%s). Activating fallback.",
            category,
            exc,
            exc_info=True,
        )

        if state is not None:
            record_session_warning(
                state=state,
                warning_message=msg,
                category=category,
                details={"exception": str(exc), "type": type(exc).__name__},
            )

        if fallback is not None:
            return fallback(exc)

        return default_value  # type: ignore[return-value]
