"""EDL Schema Validation Callback for VideoGuru (SPEC-024).

Provides after_model_callback implementation to validate all model-generated JSON
structured outputs against the EDLEntry and EditDecisionList Pydantic schemas:
- Verifies timecode bounds (end_trim > start_trim >= 0.0).
- Verifies permitted TransitionIntent enum values.
- Validates clip references against session state clip manifest when available.
- Rejects malformed JSON and prompts the model for schema-compliant re-generation.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Sequence, Union

from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ValidationError

from schemas.media import ClipManifestEntry
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent

logger = logging.getLogger("videoguru.security.schema_validation")


class SchemaValidationError(ValueError):
    """Raised when model-generated output violates the EDL Pydantic schema."""
    pass


def extract_json_payload(text: str) -> Optional[Any]:
    """Extract and parse JSON content from model output string.

    Supports raw JSON strings as well as markdown fenced code blocks (```json ... ```).

    Args:
        text: Raw text response from the model.

    Returns:
        Parsed JSON data (list or dict), or None if no valid JSON structure found.
    """
    if not text:
        return None

    clean_text = text.strip()

    # 1. Match markdown code blocks: ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    candidate_str = code_block_match.group(1).strip() if code_block_match else clean_text

    # 2. Try parsing directly
    try:
        return json.loads(candidate_str)
    except json.JSONDecodeError:
        pass

    # 3. Look for outermost bracket/brace slice
    start_bracket = clean_text.find("[")
    end_bracket = clean_text.rfind("]")
    if start_bracket != -1 and end_bracket > start_bracket:
        try:
            return json.loads(clean_text[start_bracket : end_bracket + 1])
        except json.JSONDecodeError:
            pass

    start_brace = clean_text.find("{")
    end_brace = clean_text.rfind("}")
    if start_brace != -1 and end_brace > start_brace:
        try:
            return json.loads(clean_text[start_brace : end_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def validate_edl_data(
    data: Any,
    clip_manifest: Optional[Sequence[Union[ClipManifestEntry, dict[str, Any]]]] = None,
) -> EditDecisionList:
    """Validate that input data conforms strictly to the EditDecisionList Pydantic schema.

    Args:
        data: Parsed JSON structure (list of cuts or dict with 'cuts'/'entries').
        clip_manifest: Optional manifest to cross-validate clip_id and clip duration.

    Returns:
        Validated EditDecisionList instance.

    Raises:
        SchemaValidationError: If validation fails.
    """
    if data is None:
        raise SchemaValidationError("EDL data cannot be None.")

    raw_list: list[Any]
    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        raw_list = data.get("cuts") or data.get("entries") or data.get("edl") or []
        if not raw_list and "file_reference" in data:
            raw_list = [data]
    else:
        raise SchemaValidationError(
            f"Expected JSON list or dict for EDL, got {type(data).__name__}"
        )

    if not raw_list:
        raise SchemaValidationError("Edit Decision List is empty (0 cuts found).")

    # Validate against Pydantic schema
    try:
        edl = EditDecisionList.from_dict_list(raw_list)
    except (ValidationError, ValueError) as exc:
        raise SchemaValidationError(f"EDL schema validation failed: {exc}") from exc

    # Optional cross-validation against Clip Manifest
    if clip_manifest:
        manifest_map: dict[str, float] = {}
        for entry in clip_manifest:
            if isinstance(entry, ClipManifestEntry):
                manifest_map[entry.clip_id] = entry.duration_seconds
            elif isinstance(entry, dict) and "clip_id" in entry:
                manifest_map[entry["clip_id"]] = float(entry.get("duration_seconds", 0.0))

        if manifest_map:
            for i, cut in enumerate(edl):
                if cut.file_reference not in manifest_map:
                    raise SchemaValidationError(
                        f"Cut #{i+1} references unknown clip_id '{cut.file_reference}'. "
                        f"Available clips: {list(manifest_map.keys())}"
                    )
                clip_dur = manifest_map[cut.file_reference]
                if clip_dur > 0 and cut.end_trim > clip_dur + 0.05:
                    raise SchemaValidationError(
                        f"Cut #{i+1} ({cut.file_reference}): end_trim ({cut.end_trim}s) exceeds "
                        f"source clip duration ({clip_dur}s)."
                    )

    return edl


def create_schema_rejection_response(error_details: str) -> LlmResponse:
    """Create an ADK LlmResponse instructing the LLM to re-generate compliant EDL JSON.

    Args:
        error_details: Specific description of what failed schema validation.

    Returns:
        LlmResponse containing corrective instructions.
    """
    feedback = (
        f"[SCHEMA VALIDATION ERROR]\n"
        f"The generated Edit Decision List (EDL) failed schema verification:\n"
        f"{error_details}\n\n"
        f"Correction Instructions:\n"
        f"1. Return a valid JSON array of EDLEntry objects.\n"
        f"2. Ensure start_trim >= 0.0 and end_trim > start_trim.\n"
        f"3. Ensure transition_intent is one of: 'cut', 'fade', 'wipe', 'slide', 'dissolve'.\n"
        f"4. Ensure file_reference matches a valid clip_id from the manifest.\n"
        f"Please regenerate the complete Edit Decision List strictly conforming to the schema."
    )
    return LlmResponse(
        content=types.Content(
            parts=[types.Part.from_text(text=feedback)]
        ),
        finish_reason="STOP",
    )


def schema_validation_callback(
    callback_context: Any = None,
    llm_response: Optional[LlmResponse] = None,
    clip_manifest: Optional[Sequence[Any]] = None,
    raise_exception: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Optional[LlmResponse]:
    """ADK after_model_callback implementation validating EDL schema integrity (SPEC-024).

    Inspects the model's text response. If the response contains an EDL or structured
    decision list:
    - Extracts and parses JSON.
    - Validates against EDLEntry and EditDecisionList Pydantic models.
    - Cross-references clip_manifest from callback_context / session state if present.
    - If valid: returns None (or unmodified response), allowing workflow progression.
    - If invalid: rejects response and returns an LlmResponse prompting regeneration.

    Args:
        callback_context: ADK CallbackContext (or None).
        llm_response: The raw LlmResponse from Gemini.
        clip_manifest: Optional list of ClipManifestEntry for reference checking.
        raise_exception: If True, raises SchemaValidationError instead of returning rejection LlmResponse.

    Returns:
        None if valid, or rejection LlmResponse if schema verification fails.
    """
    resp = llm_response
    if resp is None and args:
        for arg in args:
            if isinstance(arg, LlmResponse):
                resp = arg
                break
    if resp is None and "response" in kwargs:
        resp = kwargs["response"]

    if resp is None or not resp.content or not resp.content.parts:
        return None

    # Extract text from response
    response_text = ""
    for part in resp.content.parts:
        if hasattr(part, "text") and part.text:
            response_text += part.text + "\n"

    # Check if text appears to contain an EDL (keywords or JSON array)
    is_edl_candidate = any(
        k in response_text for k in ["start_trim", "end_trim", "file_reference", "transition_intent"]
    )
    if not is_edl_candidate:
        # Conversational or non-EDL response passes through
        return None

    # Attempt JSON extraction
    parsed_json = extract_json_payload(response_text)
    if parsed_json is None:
        err_msg = "Could not parse valid JSON from response containing EDL keywords."
        logger.warning("[SCHEMA VALIDATION FAILURE] %s", err_msg)
        if raise_exception:
            raise SchemaValidationError(err_msg)
        return create_schema_rejection_response(err_msg)

    # Attempt to resolve clip manifest from callback context if not explicitly provided
    manifest = clip_manifest
    if manifest is None and callback_context is not None:
        session = getattr(callback_context, "session", None)
        if session and hasattr(session, "state"):
            manifest = session.state.get("clip_manifest")

    # Validate against schema
    try:
        validate_edl_data(parsed_json, clip_manifest=manifest)
        logger.debug("after_model_callback: EDL successfully validated against schema.")
        return None  # Validation passed, allow response to proceed
    except SchemaValidationError as exc:
        logger.warning("[SCHEMA VALIDATION REJECTED] %s", exc)
        if raise_exception:
            raise
        return create_schema_rejection_response(str(exc))
