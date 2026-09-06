"""VideoGuru Observability and Structured Logging (SPEC-028).

Provides structured JSON logging, contextual metadata enrichment,
and standardized event loggers for multi-agent transitions, tool invocations,
EDL iterations, and rendering progress.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Optional, Union

from config import settings


class JsonFormatter(logging.Formatter):
    """Logging formatter that serializes log records as structured JSON."""

    def __init__(
        self,
        include_traceback: bool = True,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the JSON formatter.

        Args:
            include_traceback: Whether to serialize exception tracebacks.
            extra_fields: Constant key-value pairs added to every log message.
        """
        super().__init__()
        self.include_traceback = include_traceback
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format a LogRecord as a single-line JSON string."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }

        # Include constant extra fields
        if self.extra_fields:
            log_entry.update(self.extra_fields)

        # Include custom attributes passed via `extra={...}`
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message",
        }
        for key, val in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                # Ensure value is JSON serializable
                try:
                    json.dumps(val)
                    log_entry[key] = val
                except (TypeError, ValueError):
                    log_entry[key] = str(val)

        # Format exception info if present
        if record.exc_info:
            if self.include_traceback:
                log_entry["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "traceback": traceback.format_exception(*record.exc_info),
                }
            else:
                log_entry["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                }
        elif record.exc_text:
            log_entry["exception_text"] = record.exc_text

        return json.dumps(log_entry, default=str)


def configure_logging(
    log_level: Optional[str] = None,
    json_format: Optional[bool] = None,
    log_file: Optional[Union[str, Path]] = None,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logging for VideoGuru.

    Args:
        log_level: Desired log level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
            Defaults to settings.LOG_LEVEL.
        json_format: If True, uses JsonFormatter. If False, standard readable format.
            Defaults to settings.LOG_JSON.
        log_file: Optional path to a file where logs will be written.
        logger_name: Specific logger to configure, or None to configure the root logger.

    Returns:
        The configured logger instance.
    """
    level_str = (log_level or getattr(settings, "LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    use_json = json_format if json_format is not None else getattr(settings, "LOG_JSON", True)

    target_logger = logging.getLogger(logger_name)
    target_logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    target_logger.handlers.clear()

    # Formatter selection
    if use_json:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console Handler (sys.stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    target_logger.addHandler(console_handler)

    # Optional File Handler
    target_file = log_file or getattr(settings, "LOG_FILE", None)
    if target_file:
        file_path = Path(target_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        target_logger.addHandler(file_handler)

    return target_logger


# Standardized Event Logging Helpers

_default_logger = logging.getLogger("videoguru.observability")


def log_agent_transition(
    from_agent: str,
    to_agent: str,
    session_id: Optional[str] = None,
    theme: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **extra: Any,
) -> None:
    """Log an agent transition event within the pipeline.

    Args:
        from_agent: Source agent name.
        to_agent: Target agent name.
        session_id: Active session ID.
        theme: Active theme or intent.
        logger: Logger to output through (defaults to observability logger).
        **extra: Additional structured metadata.
    """
    log = logger or _default_logger
    metadata: dict[str, Any] = {
        "event": "agent_transition",
        "from_agent": from_agent,
        "to_agent": to_agent,
        "session_id": session_id,
        "theme": theme,
        **extra,
    }
    log.info(
        f"Agent transition: '{from_agent}' -> '{to_agent}' (session_id={session_id})",
        extra=metadata,
    )


def log_tool_invocation(
    tool_name: str,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    status: str = "success",
    logger: Optional[logging.Logger] = None,
    **extra: Any,
) -> None:
    """Log a tool invocation event.

    Args:
        tool_name: Name of the invoked tool.
        session_id: Active session ID.
        agent_name: Invoking agent name.
        duration_seconds: Execution elapsed time in seconds.
        status: 'success', 'failed', or 'degraded'.
        logger: Logger to output through.
        **extra: Additional metadata (arguments, result summary, etc.).
    """
    log = logger or _default_logger
    metadata: dict[str, Any] = {
        "event": "tool_invocation",
        "tool_name": tool_name,
        "session_id": session_id,
        "agent_name": agent_name,
        "duration_seconds": duration_seconds,
        "status": status,
        **extra,
    }
    log.info(
        f"Tool '{tool_name}' completed with status '{status}' "
        f"(agent={agent_name}, session_id={session_id})",
        extra=metadata,
    )


def log_edl_iteration(
    iteration: int,
    cut_count: int,
    total_duration: float,
    score: Optional[float] = None,
    passed: Optional[bool] = None,
    feedback: Optional[str] = None,
    session_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **extra: Any,
) -> None:
    """Log an EDL curation/review iteration within the autonomous loop.

    Args:
        iteration: Current iteration number (1-based).
        cut_count: Total cuts in the EDL draft.
        total_duration: Total video timeline duration in seconds.
        score: Review or critic score (0.0 - 10.0).
        passed: Whether the draft passed review criteria.
        feedback: Review feedback or guidance text.
        session_id: Active session ID.
        logger: Logger to output through.
        **extra: Additional metadata.
    """
    log = logger or _default_logger
    metadata: dict[str, Any] = {
        "event": "edl_iteration",
        "iteration": iteration,
        "cut_count": cut_count,
        "total_duration": total_duration,
        "score": score,
        "passed": passed,
        "feedback_preview": feedback[:120] if feedback else None,
        "session_id": session_id,
        **extra,
    }
    status_str = "PASSED" if passed else ("REVISE" if passed is False else "EVALUATING")
    log.info(
        f"EDL Iteration #{iteration}: {cut_count} cuts, {total_duration:.2f}s, "
        f"status={status_str}, score={score} (session_id={session_id})",
        extra=metadata,
    )


def log_render_progress(
    stage: str,
    progress_percent: Optional[float] = None,
    output_path: Optional[str] = None,
    session_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **extra: Any,
) -> None:
    """Log rendering pipeline milestones and progress.

    Args:
        stage: Rendering stage (e.g., 'trimming', 'xfade_transitions', 'ducking', 'captions', 'completed').
        progress_percent: Estimated or exact completion percentage (0.0 - 100.0).
        output_path: Destination path for current stage artifact.
        session_id: Active session ID.
        logger: Logger to output through.
        **extra: Additional metadata.
    """
    log = logger or _default_logger
    metadata: dict[str, Any] = {
        "event": "render_progress",
        "stage": stage,
        "progress_percent": progress_percent,
        "output_path": output_path,
        "session_id": session_id,
        **extra,
    }
    progress_str = f" ({progress_percent:.0f}%)" if progress_percent is not None else ""
    log.info(
        f"Render milestone: stage='{stage}'{progress_str} -> {output_path} (session_id={session_id})",
        extra=metadata,
    )
