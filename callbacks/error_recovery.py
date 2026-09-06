"""Tool Error Recovery & Self-Correction Callback for VideoGuru (SPEC-023).

Provides after_tool_callback and on_tool_error_callback implementations to intercept
tool and FFmpeg errors, parse stderr for common failure patterns (corrupt media, filter
syntax, dimension mismatches, codec errors, timeouts), format actionable self-correction
guidance for the LLM, and provide retry logic with exponential backoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import time
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger("videoguru.security.error_recovery")

# Common FFmpeg and tool failure pattern classifiers
ERROR_PATTERNS: Sequence[tuple[str, str, re.Pattern[str], str, bool]] = [
    (
        "CORRUPT_MEDIA",
        "Source media stream or container appears corrupt or truncated.",
        re.compile(
            r"(?i)(invalid data found when processing input|moov atom not found|header missing|"
            r"error reading header|corrupt input packet|partial file|end of file)"
        ),
        "The source media file appears corrupt or incomplete. Re-ingest the file or adjust trim "
        "timecodes to exclude damaged frames.",
        False,  # Not retryable without changing timecodes/file
    ),
    (
        "FILTER_GRAPH_ERROR",
        "FFmpeg filter_complex syntax or stream connection error.",
        re.compile(
            r"(?i)(no such filter|cannot find a matching stream|invalid filter graph|"
            r"unconnected output|matches no streams|invalid stream index|does not exist in filtergraph)"
        ),
        "Filter graph syntax or stream routing error in FFmpeg. Verify filter names and pad labels match "
        "input stream indices and transition intent parameters.",
        False,
    ),
    (
        "DIMENSION_FORMAT_MISMATCH",
        "Video stream resolution, aspect ratio, or pixel format mismatch.",
        re.compile(
            r"(?i)(width not divisible by 2|height not divisible by 2|incompatible pixel format|"
            r"does not match input frame size|dimensions must be divisible|different aspect ratio)"
        ),
        "Video stream resolution or chroma subsampling mismatch across clips. Pre-scale and normalize "
        "all inputs to standard 1920x1080 yuv420p before transition crossfading.",
        False,
    ),
    (
        "CODEC_ERROR",
        "Requested encoder or codec is not supported or misconfigured.",
        re.compile(
            r"(?i)(unknown encoder|unsupported codec|codec .* not found|error initializing output stream|"
            r"could not find codec parameters|encoder not found)"
        ),
        "The requested encoder or codec is unavailable in this FFmpeg installation. Fall back to "
        "standard libx264 for video and aac for audio.",
        False,
    ),
    (
        "TIMEOUT_ERROR",
        "Tool or FFmpeg process execution timed out.",
        re.compile(
            r"(?i)(timed out after|timeoutexpired|command timed out|deadline exceeded)"
        ),
        "Execution exceeded the designated timeout. Consider reducing clip durations, lowering the "
        "encoding preset (e.g. -preset ultrafast), or processing in smaller batches.",
        True,  # Potentially retryable with higher timeout or retry
    ),
    (
        "FILE_IO_ERROR",
        "File access, permission, or missing resource error.",
        re.compile(
            r"(?i)(no such file or directory|permission denied|cannot open file|"
            r"file exists|access is denied|resource temporarily unavailable)"
        ),
        "File system access or missing path error. Verify that all input files exist in the designated "
        "staging/media directory and that output paths are writable.",
        True,  # Transient file lock may succeed on retry
    ),
]


@dataclass(frozen=True)
class ParsedToolError:
    """Structured representation of a parsed tool/FFmpeg error."""

    category: str
    root_cause: str
    recommendation: str
    raw_error: str
    is_retryable: bool

    def to_llm_feedback(self) -> str:
        """Format an actionable, LLM-friendly diagnostic message for self-correction."""
        return (
            f"[TOOL ERROR RECOVERY: {self.category}]\n"
            f"Root Cause: {self.root_cause}\n"
            f"Recommendation: {self.recommendation}\n"
            f"Diagnostic Snippet: {self.raw_error[:200].strip()}"
        )


def parse_tool_error(error_text: str) -> ParsedToolError:
    """Parse stderr or error text into a structured ParsedToolError.

    Args:
        error_text: Raw stderr or exception message.

    Returns:
        ParsedToolError with category, root cause, and recommendations.
    """
    clean_text = error_text.strip() if error_text else ""
    for category, root_cause, pattern, recommendation, is_retryable in ERROR_PATTERNS:
        if pattern.search(clean_text):
            return ParsedToolError(
                category=category,
                root_cause=root_cause,
                recommendation=recommendation,
                raw_error=clean_text,
                is_retryable=is_retryable,
            )

    return ParsedToolError(
        category="GENERAL_ERROR",
        root_cause="FFmpeg or tool execution failed with an unspecified error.",
        recommendation="Inspect the stderr trace and adjust tool arguments or timecode bounds.",
        raw_error=clean_text,
        is_retryable=False,
    )


def execute_with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 0.2,
    backoff_factor: float = 2.0,
    is_retryable_fn: Optional[Callable[[Exception], bool]] = None,
    **kwargs: Any,
) -> Any:
    """Execute a callable with exponential backoff retry logic (up to max_retries).

    Args:
        func: The function to execute.
        *args: Positional arguments for func.
        max_retries: Maximum number of retry attempts (default: 3).
        initial_delay: Initial sleep delay in seconds (default: 0.2).
        backoff_factor: Multiplier for sleep duration after each failure (default: 2.0).
        is_retryable_fn: Optional predicate checking if an exception is retryable.
        **kwargs: Keyword arguments for func.

    Returns:
        The result of func(*args, **kwargs).

    Raises:
        Exception: The last caught exception if all retries are exhausted or non-retryable.
    """
    last_error: Optional[Exception] = None
    delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            parsed = parse_tool_error(str(exc))
            can_retry = is_retryable_fn(exc) if is_retryable_fn else parsed.is_retryable

            logger.warning(
                "Attempt %d/%d for '%s' failed: %s (retryable: %s)",
                attempt,
                max_retries,
                getattr(func, "__name__", str(func)),
                exc,
                can_retry,
            )

            if attempt < max_retries and can_retry:
                time.sleep(delay)
                delay *= backoff_factor
            else:
                break

    if last_error:
        raise last_error


def after_tool_error_recovery_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any = None,
    tool_response: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Optional[dict[str, Any]]:
    """ADK after_tool_callback implementation evaluating tool output for errors (SPEC-023).

    Checks the tool_response dictionary for error flags or non-zero status. If an
    error is detected:
    - Parses stderr / message into a structured ParsedToolError.
    - Appends LLM-friendly self-correction guidance (`self_correction_guidance`,
      `error_category`, `recovery_recommendation`) to the tool response.
    - Logs the recovery event.

    Args:
        tool: The executed tool.
        args: The arguments passed to the tool.
        tool_context: ADK ToolContext (or None).
        tool_response: The dictionary returned by the tool.

    Returns:
        Enriched tool response dict if errors are found, or original response / None.
    """
    if not isinstance(tool_response, dict):
        return tool_response

    # Check for failure indicators
    is_error = (
        tool_response.get("status") == "error"
        or "error" in tool_response
        or (tool_response.get("returncode", 0) != 0 and "returncode" in tool_response)
    )

    if not is_error:
        return tool_response

    raw_err = str(
        tool_response.get("stderr")
        or tool_response.get("error")
        or tool_response.get("message")
        or ""
    )

    parsed = parse_tool_error(raw_err)
    logger.info(
        "after_tool_callback trapped tool error in '%s': category=%s",
        getattr(tool, "name", str(tool)),
        parsed.category,
    )

    updated_response = dict(tool_response)
    updated_response["parsed_error"] = {
        "category": parsed.category,
        "root_cause": parsed.root_cause,
        "recommendation": parsed.recommendation,
        "is_retryable": parsed.is_retryable,
    }
    updated_response["self_correction_guidance"] = parsed.to_llm_feedback()

    return updated_response


def on_tool_error_recovery_callback(
    tool: Any,
    args: dict[str, Any],
    tool_context: Any = None,
    error: Optional[Exception] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """ADK on_tool_error_callback implementation trapping unhandled tool exceptions (SPEC-023).

    Intercepts unhandled tool exceptions:
    - Parses the exception string for failure patterns.
    - Formats a structured recovery dictionary with actionable self-correction feedback.
    - Prevents pipeline abort, returning diagnostic guidance back to the LLM agent.

    Args:
        tool: The tool that raised an exception.
        args: Arguments that caused the error.
        tool_context: ADK ToolContext (or None).
        error: The caught exception.

    Returns:
        Structured error dict with self_correction_guidance for the agent.
    """
    tool_name = getattr(tool, "name", str(tool))
    err_str = str(error) if error else "Unknown tool execution error"
    parsed = parse_tool_error(err_str)

    logger.warning(
        "[TOOL ERROR RECOVERY HOOK] Trapped exception in tool '%s' (%s): %s",
        tool_name,
        parsed.category,
        err_str,
    )

    return {
        "status": "error",
        "tool_name": tool_name,
        "error_type": type(error).__name__ if error else "Exception",
        "raw_error": err_str,
        "parsed_error": {
            "category": parsed.category,
            "root_cause": parsed.root_cause,
            "recommendation": parsed.recommendation,
            "is_retryable": parsed.is_retryable,
        },
        "self_correction_guidance": parsed.to_llm_feedback(),
    }
