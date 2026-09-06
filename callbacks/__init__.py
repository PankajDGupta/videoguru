"""VideoGuru Callbacks & Security Package (Phase VI).

Consolidates the four ADK lifecycle interception hooks:
1. before_model_callback: Input guardrails against prompt injection and policy breaches (SPEC-021).
2. before_tool_callback: Sandboxing FFmpeg and tools to authorized paths and flags (SPEC-022).
3. after_tool_callback & on_tool_error_callback: Tool failure parsing and self-correction (SPEC-023).
4. after_model_callback: Strict Pydantic EDL schema validation and regeneration requests (SPEC-024).
"""

from __future__ import annotations

from typing import Any, Optional

from callbacks.error_recovery import (
    ParsedToolError,
    after_tool_error_recovery_callback,
    execute_with_retry,
    on_tool_error_recovery_callback,
    parse_tool_error,
)
from callbacks.input_guardrail import (
    check_prompt_safety,
    create_blocked_response,
    extract_prompt_text,
    input_guardrail_callback,
    sanitize_prompt,
)
from callbacks.schema_validation import (
    SchemaValidationError,
    create_schema_rejection_response,
    extract_json_payload,
    schema_validation_callback,
    validate_edl_data,
)
from callbacks.tool_sandbox import (
    SecuritySandboxingError,
    before_tool_sandbox_callback,
    is_shell_safe,
    validate_ffmpeg_command,
    validate_path_confined,
)


def register_security_callbacks(
    agent: Any,
    enable_input_guardrail: bool = True,
    enable_tool_sandbox: bool = True,
    enable_error_recovery: bool = True,
    enable_schema_validation: bool = True,
) -> Any:
    """Register Phase VI security and validation callbacks onto an ADK Agent instance.

    Args:
        agent: The ADK Agent / LlmAgent instance or workflow container.
        enable_input_guardrail: Attach before_model_callback (SPEC-021).
        enable_tool_sandbox: Attach before_tool_callback (SPEC-022).
        enable_error_recovery: Attach after_tool_callback & on_tool_error_callback (SPEC-023).
        enable_schema_validation: Attach after_model_callback (SPEC-024).

    Returns:
        The configured agent instance.
    """
    # Recursively register on child sub-agents if present (e.g., SequentialAgent, LoopAgent)
    if hasattr(agent, "sub_agents") and agent.sub_agents:
        for sub in agent.sub_agents:
            register_security_callbacks(
                sub,
                enable_input_guardrail=enable_input_guardrail,
                enable_tool_sandbox=enable_tool_sandbox,
                enable_error_recovery=enable_error_recovery,
                enable_schema_validation=enable_schema_validation,
            )

    # Attach callbacks to LLM agents supporting ADK model/tool lifecycle hooks
    if hasattr(agent, "before_model_callback"):
        if enable_input_guardrail:
            current_bm = getattr(agent, "before_model_callback", None)
            callbacks = (
                current_bm if isinstance(current_bm, list) else ([current_bm] if current_bm else [])
            )
            if input_guardrail_callback not in callbacks:
                callbacks.append(input_guardrail_callback)
            agent.before_model_callback = callbacks

        if enable_tool_sandbox:
            current_bt = getattr(agent, "before_tool_callback", None)
            callbacks = (
                current_bt if isinstance(current_bt, list) else ([current_bt] if current_bt else [])
            )
            if before_tool_sandbox_callback not in callbacks:
                callbacks.append(before_tool_sandbox_callback)
            agent.before_tool_callback = callbacks

        if enable_error_recovery:
            current_at = getattr(agent, "after_tool_callback", None)
            callbacks = (
                current_at if isinstance(current_at, list) else ([current_at] if current_at else [])
            )
            if after_tool_error_recovery_callback not in callbacks:
                callbacks.append(after_tool_error_recovery_callback)
            agent.after_tool_callback = callbacks

            current_oe = getattr(agent, "on_tool_error_callback", None)
            oe_callbacks = (
                current_oe if isinstance(current_oe, list) else ([current_oe] if current_oe else [])
            )
            if on_tool_error_recovery_callback not in oe_callbacks:
                oe_callbacks.append(on_tool_error_recovery_callback)
            agent.on_tool_error_callback = oe_callbacks

        if enable_schema_validation:
            current_am = getattr(agent, "after_model_callback", None)
            callbacks = (
                current_am if isinstance(current_am, list) else ([current_am] if current_am else [])
            )
            if schema_validation_callback not in callbacks:
                callbacks.append(schema_validation_callback)
            agent.after_model_callback = callbacks

    return agent


__all__ = [
    # SPEC-021: Input Guardrail
    "input_guardrail_callback",
    "check_prompt_safety",
    "sanitize_prompt",
    "extract_prompt_text",
    "create_blocked_response",
    # SPEC-022: Tool Sandboxing
    "before_tool_sandbox_callback",
    "validate_ffmpeg_command",
    "validate_path_confined",
    "is_shell_safe",
    "SecuritySandboxingError",
    # SPEC-023: Error Recovery
    "after_tool_error_recovery_callback",
    "on_tool_error_recovery_callback",
    "parse_tool_error",
    "ParsedToolError",
    "execute_with_retry",
    # SPEC-024: Schema Validation
    "schema_validation_callback",
    "validate_edl_data",
    "extract_json_payload",
    "SchemaValidationError",
    "create_schema_rejection_response",
    # Orchestration helper
    "register_security_callbacks",
]
