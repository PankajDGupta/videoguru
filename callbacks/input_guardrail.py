"""Input Guardrail Callback for VideoGuru (SPEC-021).

Provides before_model_callback implementation to sanitize user prompts, detect and
block prompt injection attempts, external URL payloads, system instruction overrides,
and sensitive file access attempts, logging security violations and short-circuiting
the LLM with an informative refusal response.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Sequence

from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

logger = logging.getLogger("videoguru.security.guardrail")

# Regexp patterns for prompt injection, jailbreaking, and instruction override
INJECTION_PATTERNS: Sequence[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            r"(?i)\b(ignore|disregard|forget|override|negate|bypass)\s+"
            r"(all\s+|any\s+)?(previous\s+|prior\s+|above\s+|system\s+|safety\s+|content\s+)?"
            r"(instructions|prompts|rules|guidelines|directions|commands|filters|guardrails)\b"
        ),
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"(?i)\b(you are now|pretend to be|act as)\s+"
            r"(DAN|an unrestricted|jailbroken|evil|unfiltered|developer mode)\b"
        ),
    ),
    (
        "developer_mode_toggle",
        re.compile(
            r"(?i)\b(developer mode|god mode|jailbreak mode)\s+(enabled|activate|on|engaged)\b"
        ),
    ),
    (
        "system_prompt_leak",
        re.compile(
            r"(?i)\b(reveal|show|print|dump|output|repeat)\s+"
            r"(your\s+|the\s+)?(system\s+prompt|initial\s+instructions|system\s+instructions)\b"
        ),
    ),
    (
        "remote_download_command",
        re.compile(
            r"(?i)\b(curl|wget|git\s+clone|invoke-webrequest|nc|netcat|ncat)\b"
        ),
    ),
    (
        "external_url_reference",
        re.compile(
            r"(?i)\b(https?|ftp|file|smb)://[^\s]+"
        ),
    ),
    (
        "system_command_injection",
        re.compile(
            r"(?i)(bash\s+-c|sh\s+-c|powershell(\.exe)?\s+|cmd(\.exe)?\s+/c|rm\s+-rf|del\s+/f|format\s+[a-z]:)"
        ),
    ),
    (
        "sensitive_file_probe",
        re.compile(
            r"(?i)(/etc/(passwd|shadow|hosts)|[a-z]:\\windows\\system32|\.env\b|id_rsa)"
        ),
    ),
]


def extract_prompt_text(llm_request: LlmRequest) -> str:
    """Extract raw user prompt text from an LlmRequest contents object.

    Args:
        llm_request: The incoming ADK LlmRequest.

    Returns:
        Concatenated text content found within the request.
    """
    if not llm_request or not llm_request.contents:
        return ""

    text_parts: list[str] = []
    for content in llm_request.contents:
        if hasattr(content, "parts") and content.parts:
            for part in content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
        elif isinstance(content, str):
            text_parts.append(content)

    return "\n".join(text_parts)


def check_prompt_safety(text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Evaluate text for prompt injections or security violations.

    Args:
        text: The prompt string to validate.

    Returns:
        Tuple of (is_safe, violation_category, violation_detail).
        If is_safe is True, category and detail are None.
    """
    if not text:
        return True, None, None

    for category, pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            matched_snippet = match.group(0)
            return False, category, f"Matched rule '{category}': '{matched_snippet}'"

    return True, None, None


def sanitize_prompt(text: str) -> str:
    """Sanitize prompt text by stripping null bytes and excessive whitespace.

    Args:
        text: The input prompt string.

    Returns:
        Sanitized prompt string.
    """
    if not text:
        return ""
    # Strip null bytes and control chars (except normal tabs and newlines)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned.strip()


def create_blocked_response(reason: str) -> LlmResponse:
    """Construct an ADK LlmResponse indicating the request was blocked by the guardrail.

    Args:
        reason: Explanation of the security block.

    Returns:
        An LlmResponse with explanatory message short-circuiting the LLM call.
    """
    guardrail_message = (
        f"[SECURITY GUARDRAIL] Request blocked: Potential prompt injection or policy violation detected.\n"
        f"Reason: {reason}\n"
        f"Please provide a valid video production theme or editorial feedback without external URLs, "
        f"system prompt overrides, or unauthorized commands."
    )
    return LlmResponse(
        content=types.Content(
            parts=[types.Part.from_text(text=guardrail_message)]
        ),
        finish_reason="STOP",
    )


def input_guardrail_callback(
    callback_context: Any = None,
    llm_request: Optional[LlmRequest] = None,
    *args: Any,
    **kwargs: Any,
) -> Optional[LlmResponse]:
    """ADK before_model_callback implementation enforcing input guardrails (SPEC-021).

    Inspects user prompt contents within the LlmRequest. If a prompt injection,
    external URL, system instruction override, or command injection is detected:
    - Logs the security incident with audit details.
    - Returns an LlmResponse with an explanation, short-circuiting the model.

    If the prompt is safe:
    - Returns None, allowing the agent to proceed to the model.

    Args:
        callback_context: ADK CallbackContext (or None).
        llm_request: The raw LlmRequest passed to the model.

    Returns:
        LlmResponse if blocked, or None to proceed.
    """
    # Accommodate flexible call conventions (positional or keyword)
    req = llm_request
    if req is None and args:
        for arg in args:
            if isinstance(arg, LlmRequest):
                req = arg
                break
    if req is None and "request" in kwargs:
        req = kwargs["request"]

    if req is None:
        return None

    prompt_text = extract_prompt_text(req)
    if not prompt_text:
        return None

    is_safe, category, detail = check_prompt_safety(prompt_text)
    if not is_safe:
        logger.warning(
            "[SECURITY GUARDRAIL BLOCKED] Prompt injection / policy violation detected! "
            "Category: %s | Detail: %s | Prompt snippet: %s",
            category,
            detail,
            prompt_text[:120].replace("\n", " "),
        )
        return create_blocked_response(detail or "Security violation")

    # Safe prompt proceeds
    return None
