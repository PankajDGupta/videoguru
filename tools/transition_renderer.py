"""Transition Rendering Tool for VideoGuru (SPEC-017).

Orchestrates pre-trimming, normalization (1920x1080 @ 30fps), and xfade transition assembly
using FFmpeg command builder modules. Supports resolution from explicit arguments or ADK ToolContext
session state, handles single-cut edge cases without xfade filters, and writes final paths to session state.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import uuid

from config import settings
from rendering.ffmpeg_builder import (
    build_trim_command,
    build_xfade_chain,
    calculate_xfade_offsets,
    execute_ffmpeg_command,
    map_transition_intent,
)
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from tools.curation_tools import get_edl_from_state
from tools.ingestion_tools import get_clip_manifest_from_state

logger = logging.getLogger(__name__)


def _resolve_edl(
    edl: Optional[Union[EditDecisionList, Sequence[dict[str, Any]], Sequence[EDLEntry], str, Path]],
    state: dict[str, Any],
) -> EditDecisionList:
    """Resolve and validate an EditDecisionList instance from argument or state.

    Args:
        edl: Explicit EDL instance, sequence of entries/dicts, or filesystem path/JSON string.
        state: ADK session state dictionary.

    Returns:
        Validated EditDecisionList instance containing cut entries.

    Raises:
        ValueError: If EDL cannot be found or is empty.
    """
    resolved: Optional[EditDecisionList] = None

    if edl is not None:
        if isinstance(edl, EditDecisionList):
            resolved = edl
        elif isinstance(edl, (str, Path)):
            p = Path(edl)
            if p.is_file():
                resolved = EditDecisionList.from_file(p)
            else:
                try:
                    resolved = EditDecisionList.from_json(str(edl))
                except Exception as exc:
                    raise ValueError(f"Failed to parse EDL from string/path: {exc}") from exc
        elif isinstance(edl, dict):
            resolved = EditDecisionList.model_validate(edl)
        elif isinstance(edl, Sequence):
            resolved = EditDecisionList.from_list(list(edl))
    else:
        resolved = get_edl_from_state(state)

    if resolved is None:
        raise ValueError("No Edit Decision List provided or found in session state.")

    if len(resolved.entries) == 0:
        raise ValueError("Edit Decision List contains no cut entries.")

    return resolved


def _resolve_clip_manifest(
    clip_manifest: Optional[Union[list[ClipManifestEntry], Sequence[dict[str, Any]], str, Path]],
    state: dict[str, Any],
) -> list[ClipManifestEntry]:
    """Resolve and validate a list of ClipManifestEntry objects from argument or state.

    Args:
        clip_manifest: List of manifest entries, sequence of dicts, or path to manifest JSON.
        state: ADK session state dictionary.

    Returns:
        List of validated ClipManifestEntry instances.

    Raises:
        ValueError: If manifest cannot be found or is empty.
        FileNotFoundError: If manifest file path does not exist.
    """
    resolved: Optional[list[ClipManifestEntry]] = None

    if clip_manifest is not None:
        if isinstance(clip_manifest, (str, Path)):
            p = Path(clip_manifest).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Clip manifest file not found: {p}")
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                resolved = [ClipManifestEntry.model_validate(x) for x in data]
            elif isinstance(data, dict):
                raw_items = data.get("clip_manifest") or data.get("manifest") or data.get("entries") or []
                resolved = [ClipManifestEntry.model_validate(x) for x in raw_items]
        elif isinstance(clip_manifest, list) and all(isinstance(x, ClipManifestEntry) for x in clip_manifest):
            resolved = clip_manifest
        elif isinstance(clip_manifest, Sequence):
            resolved = [
                ClipManifestEntry.model_validate(x) if isinstance(x, dict) else x
                for x in clip_manifest
            ]
    else:
        if "clip_manifest" not in state:
            raise ValueError("No clip manifest provided or found in session state.")
        resolved = get_clip_manifest_from_state(state)

    if resolved is None:
        raise ValueError("No clip manifest provided or found in session state.")

    if len(resolved) == 0:
        raise ValueError("Clip manifest is empty.")

    return resolved


def _extract_transitions(edl: EditDecisionList) -> list[str]:
    """Extract and map canonical transition names for each adjacent cut pair.

    For each boundary between cut i and cut i+1:
    - Inspects incoming cut i+1's transition_intent (preferred if not CUT).
    - Falls back to outgoing cut i's transition_intent if not CUT.
    - Otherwise defaults to CUT.
    - Maps canonical FFmpeg xfade transition names.

    Args:
        edl: EditDecisionList with at least 2 cuts.

    Returns:
        List of canonical transition filter names (length = len(edl) - 1).
    """
    transitions: list[str] = []
    n = len(edl.entries)
    for i in range(n - 1):
        curr_entry = edl.entries[i]
        next_entry = edl.entries[i + 1]

        if next_entry.transition_intent != TransitionIntent.CUT:
            intent = next_entry.transition_intent
        elif curr_entry.transition_intent != TransitionIntent.CUT:
            intent = curr_entry.transition_intent
        else:
            intent = next_entry.transition_intent

        canonical = map_transition_intent(intent)
        transitions.append(canonical)

    return transitions


def render_with_transitions(
    edl: Optional[Union[EditDecisionList, Sequence[dict[str, Any]], Sequence[EDLEntry], str, Path]] = None,
    clip_manifest: Optional[Union[list[ClipManifestEntry], Sequence[dict[str, Any]], str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    transition_duration: float = 1.0,
    tool_context: Optional[Any] = None,
    staging_dir: Optional[Union[str, Path]] = None,
) -> str:
    """ADK Tool: Render a compiled video from an EDL with xfade transitions and audio crossfades (SPEC-017).

    Resolves EDL cuts and Clip Manifest metadata from arguments or tool_context.state.
    Pre-trims and normalizes each segment into a temporary staging clip (1920x1080 @ 30fps)
    using FFmpeg scale, pad, and fps filters. For multi-cut EDLs, chains video xfade and audio
    acrossfade filters dynamically. For single-cut EDLs, directly trims and normalizes to the output.

    Args:
        edl: Explicit EditDecisionList instance, sequence of cut dicts, or file path.
        clip_manifest: List of ClipManifestEntry objects, sequence of dicts, or manifest file path.
        output_path: Destination path for the rendered output video file. If None, uses settings.OUTPUT_DIR.
        transition_duration: Crossfade overlap duration in seconds (default: 1.0).
        tool_context: Optional ADK ToolContext providing access to session.state.
        staging_dir: Optional custom directory for intermediate trimmed segment clips.

    Returns:
        Absolute filesystem path string to the rendered video.

    Raises:
        ValueError: If EDL is empty, manifest is empty, clips are missing, or durations are invalid.
        RuntimeError: If FFmpeg execution fails.
    """
    state = tool_context.state if tool_context is not None and hasattr(tool_context, "state") else {}

    # 1. Resolve and validate EDL & Manifest
    resolved_edl = _resolve_edl(edl, state)
    resolved_manifest = _resolve_clip_manifest(clip_manifest, state)

    manifest_map: dict[str, ClipManifestEntry] = {
        clip.clip_id: clip for clip in resolved_manifest
    }

    # 2. Verify all EDL cuts map to an ingested source clip
    for idx, cut in enumerate(resolved_edl.entries):
        if cut.file_reference not in manifest_map:
            raise ValueError(
                f"Clip '{cut.file_reference}' referenced in EDL cut {idx} not found in manifest."
            )

    # 3. Resolve destination output path
    if output_path is not None:
        target_output = Path(output_path).resolve()
    else:
        out_dir = settings.OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        target_output = (out_dir / "rendered_transitions.mp4").resolve()

    target_output.parent.mkdir(parents=True, exist_ok=True)

    # 4. Handle Edge Case: Single Cut (direct trim to output, no xfade chain)
    if len(resolved_edl.entries) == 1:
        cut = resolved_edl.entries[0]
        clip = manifest_map[cut.file_reference]
        logger.info(
            "Rendering single cut EDL directly: clip=%s, start=%.3f, end=%.3f -> %s",
            cut.file_reference,
            cut.start_trim,
            cut.end_trim,
            target_output,
        )
        trim_cmd = build_trim_command(
            clip_path=clip.absolute_path,
            start=cut.start_trim,
            end=cut.end_trim,
            output_path=target_output,
            target_resolution="1920x1080",
            target_fps=30.0,
        )
        execute_ffmpeg_command(trim_cmd)

        if tool_context is not None and hasattr(tool_context, "state"):
            tool_context.state["transition_rendered_path"] = str(target_output)
            logger.info("Updated tool_context.state['transition_rendered_path'] = %s", target_output)

        return str(target_output)

    # 5. Multi-cut assembly: Validate transition duration
    if transition_duration <= 0.0:
        raise ValueError(f"transition_duration must be positive, got {transition_duration}")

    for idx, cut in enumerate(resolved_edl.entries):
        if cut.duration <= transition_duration:
            raise ValueError(
                f"Cut {idx} duration ({cut.duration:.3f}s) must be strictly greater than "
                f"transition_duration ({transition_duration:.3f}s)."
            )

    # 6. Prepare staging directory for pre-trimmed clips
    if staging_dir is not None:
        stage_path = Path(staging_dir).resolve()
    else:
        stage_path = settings.STAGING_DIR / f"render_staging_{uuid.uuid4().hex[:8]}"

    stage_path.mkdir(parents=True, exist_ok=True)

    # 7. Pre-trim each segment to normalized resolution and frame rate
    trimmed_paths: list[str] = []
    durations: list[float] = []

    for idx, cut in enumerate(resolved_edl.entries):
        clip = manifest_map[cut.file_reference]
        staged_clip = stage_path / f"cut_{idx}_{cut.file_reference}.mp4"

        trim_cmd = build_trim_command(
            clip_path=clip.absolute_path,
            start=cut.start_trim,
            end=cut.end_trim,
            output_path=staged_clip,
            target_resolution="1920x1080",
            target_fps=30.0,
        )
        logger.info(
            "Pre-trimming cut %d/%d (clip=%s, range=[%.2f, %.2f]) -> %s",
            idx + 1,
            len(resolved_edl),
            cut.file_reference,
            cut.start_trim,
            cut.end_trim,
            staged_clip,
        )
        execute_ffmpeg_command(trim_cmd)

        trimmed_paths.append(str(staged_clip))
        durations.append(cut.duration)

    # 8. Build xfade filter complex command
    transitions = _extract_transitions(resolved_edl)
    logger.info(
        "Building xfade chain for %d clips with transitions: %s",
        len(trimmed_paths),
        transitions,
    )

    xfade_cmd = build_xfade_chain(
        clips=trimmed_paths,
        transitions=transitions,
        durations=durations,
        output_path=target_output,
        transition_duration=transition_duration,
    )

    # 9. Execute full FFmpeg render command
    logger.info("Executing xfade render command -> %s", target_output)
    execute_ffmpeg_command(xfade_cmd)

    # 10. Update session state
    if tool_context is not None and hasattr(tool_context, "state"):
        tool_context.state["transition_rendered_path"] = str(target_output)
        logger.info("Updated tool_context.state['transition_rendered_path'] = %s", target_output)

    return str(target_output)
