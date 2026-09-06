"""Curation Tools for VideoGuru (SPEC-009).

Coordinates assembling an Edit Decision List (EDL) from the ingested Clip Manifest
by executing multimodal clip analysis (SPEC-008) for each clip against the creator's theme,
validating cut references, and storing the resulting EditDecisionList in session state under `session.state["edl"]`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.adk.tools import ToolContext

from config import settings
from schemas.edl import EDLEntry, EditDecisionList
from schemas.media import ClipManifestEntry
from tools.ingestion_tools import get_clip_manifest_from_state
from tools.intent_tools import get_theme_from_state
from tools.video_analysis import analyze_clip

logger = logging.getLogger(__name__)


def assemble_edl_from_manifest(
    manifest: list[ClipManifestEntry],
    theme: str,
    client: Optional[Any] = None,
    offline: bool = False,
    feedback: Optional[str] = None,
) -> EditDecisionList:
    """Iterate through the clip manifest and analyze each clip to construct an Edit Decision List.

    Args:
        manifest: Ordered list of ClipManifestEntry models discovered during Phase I.
        theme: The narrative anchor or highlight theme described by the creator.
        client: Optional google.genai.Client instance for dependency injection or mocking.
        offline: If True, executes deterministic offline analysis for each clip.
        feedback: Optional editorial review/critic feedback to steer segment curation.

    Returns:
        Assembled and validated EditDecisionList instance containing all selected cuts.

    Raises:
        ValueError: If manifest is empty or theme is empty.
    """
    if not manifest:
        raise ValueError("Cannot assemble EDL from an empty clip manifest.")

    cleaned_theme = theme.strip() if isinstance(theme, str) else ""
    if not cleaned_theme:
        raise ValueError("Theme must be a non-empty string for video curation.")

    effective_theme = cleaned_theme
    if feedback and str(feedback).strip():
        cleaned_feedback = str(feedback).strip()
        effective_theme = f"{cleaned_theme} [Editorial Feedback: {cleaned_feedback}]"
        logger.info(
            "Curation incorporating revision feedback: '%s'",
            cleaned_feedback,
        )

    logger.info(
        "Assembling EDL from manifest with %d clip(s) for theme: '%s' (mode: %s)",
        len(manifest),
        effective_theme,
        "offline" if offline else "live",
    )

    edl_entries: list[EDLEntry] = []
    for idx, clip in enumerate(manifest, 1):
        logger.debug(
            "Curating clip %d/%d: '%s' [%s] (%.2fs)",
            idx,
            len(manifest),
            clip.file_name,
            clip.clip_id,
            clip.duration_seconds,
        )
        try:
            entry = analyze_clip(
                clip_path=clip.absolute_path,
                theme=effective_theme,
                clip_id=clip.clip_id,
                client=client,
                offline=offline,
            )
            edl_entries.append(entry)
        except Exception as exc:
            logger.error(
                "Failed to analyze clip '%s' [%s] during curation: %s",
                clip.file_name,
                clip.clip_id,
                exc,
            )
            raise

    edl = EditDecisionList(entries=edl_entries)

    # Validate that every cut references a known clip from the manifest
    edl.validate_clip_references(manifest, raise_on_error=True)

    logger.info(
        "Successfully assembled EDL with %d cut(s), total duration: %.2fs (~%.1f min)",
        len(edl),
        edl.total_duration,
        edl.total_duration / 60.0,
    )
    return edl


def curate_edit_decision_list(
    tool_context: Optional[ToolContext] = None,
    theme: Optional[str] = None,
    offline: bool = False,
) -> str:
    """Analyze all clips in the session clip manifest and assemble the Edit Decision List (EDL).

    This function serves as a custom Google ADK tool callable by the Curation Agent.
    Retrieves the Clip Manifest and theme from session state, calls `analyze_clip` per clip,
    constructs the EditDecisionList, and saves it in `tool_context.state["edl"]`.

    Args:
        tool_context: Optional ADK ToolContext providing access to session state.
        theme: Optional explicit theme. If omitted, retrieved from session state.
        offline: If True, forces offline mock curation.

    Returns:
        Structured confirmation report summarizing the curated cuts and duration.

    Raises:
        ValueError: If clip manifest is missing or empty in session state.
    """
    state = tool_context.state if tool_context is not None else {}

    # 1. Retrieve manifest from session state
    manifest = get_clip_manifest_from_state(state)
    if not manifest:
        raise ValueError(
            "No clip manifest found in session state. Please run Phase I Media Ingestion "
            "(Ingestion Agent) before curating the Edit Decision List."
        )

    # 2. Retrieve theme
    resolved_theme = (
        theme.strip()
        if (theme and str(theme).strip())
        else (get_theme_from_state(state) or "General highlights and engaging moments")
    )

    # 3. Retrieve any critic or revision feedback from previous loop iterations
    feedback = state.get("critic_feedback") or state.get("revision_feedback")

    # 4. Assemble EDL
    has_api_key = bool(settings.GEMINI_API_KEY)
    is_offline = offline or not has_api_key

    edl = assemble_edl_from_manifest(
        manifest=manifest,
        theme=resolved_theme,
        offline=is_offline,
        feedback=str(feedback) if feedback else None,
    )

    # 5. Persist to session state
    serialized_edl = edl.to_dict_list()
    if tool_context is not None:
        tool_context.state["edl"] = serialized_edl
        tool_context.state["edl_total_duration"] = edl.total_duration
        logger.info(
            "Stored %d EDL cut(s) in session state under 'edl'.",
            len(edl),
        )

    # 6. Format summary report
    minutes = int(edl.total_duration // 60)
    seconds = int(edl.total_duration % 60)
    formatted_duration = f"{minutes}m {seconds}s ({edl.total_duration:.2f}s)"

    scores = [e.engagement_score for e in edl if e.engagement_score is not None]
    avg_score_str = f"{sum(scores) / len(scores):.1f}/10.0" if scores else "N/A"

    cut_lines: list[str] = []
    for idx, e in enumerate(edl, 1):
        score_info = f", score={e.engagement_score:.1f}" if e.engagement_score is not None else ""
        cut_lines.append(
            f"  {idx}. [{e.file_reference}] {e.start_trim:.2f}s -> {e.end_trim:.2f}s "
            f"({e.duration:.2f}s, transition={e.transition_intent.value}{score_info})\n"
            f"      Rationale: {e.scene_rationale}"
        )

    report = (
        f"[CurationAgent] Successfully curated Edit Decision List (EDL) with {len(edl)} cut(s).\n"
        f"Theme: {resolved_theme}\n"
        f"Total Runtime: {formatted_duration} | Average Engagement: {avg_score_str}\n"
        f"Cut List:\n" + "\n".join(cut_lines) + "\n"
        f"Session state updated: 'edl' locked in with {len(edl)} cut entries."
    )
    return report


def get_edl_from_state(state: dict[str, Any]) -> Optional[EditDecisionList]:
    """Retrieve and deserialize the EditDecisionList from session state.

    Args:
        state: ADK session state dictionary.

    Returns:
        EditDecisionList instance if found and valid, otherwise None.
    """
    raw_edl = state.get("edl")
    if not raw_edl:
        return None

    try:
        if isinstance(raw_edl, EditDecisionList):
            return raw_edl
        if isinstance(raw_edl, list):
            return EditDecisionList.from_list(raw_edl)
        if isinstance(raw_edl, dict):
            return EditDecisionList.model_validate(raw_edl)
    except Exception as exc:
        logger.warning("Failed to parse EditDecisionList from state: %s", exc)
        return None

    return None
