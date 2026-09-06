"""OpenTimelineIO Converter Tool for VideoGuru (SPEC-013).

Provides functions to convert Edit Decision Lists (EDLs) into OpenTimelineIO (.otio) format
and load them back.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import opentimelineio as otio

from config import settings
from schemas.edl import EditDecisionList
from schemas.media import ClipManifestEntry
from tools.curation_tools import get_edl_from_state
from tools.ingestion_tools import get_clip_manifest_from_state

logger = logging.getLogger(__name__)


def _create_otio_timeline(
    edl: EditDecisionList,
    clip_manifest: list[ClipManifestEntry],
    timeline_name: str = 'VideoGuru Timeline',
) -> otio.schema.Timeline:
    """Create an OpenTimelineIO Timeline from an EditDecisionList and ClipManifest.

    Args:
        edl: EditDecisionList instance containing cut entries.
        clip_manifest: List of ClipManifestEntry objects.
        timeline_name: Name for the OTIO timeline.

    Returns:
        otio.schema.Timeline object.
    """
    timeline = otio.schema.Timeline(name=timeline_name)
    video_track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    audio_track = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
    
    timeline.tracks.append(video_track)
    timeline.tracks.append(audio_track)

    manifest_map = {clip.clip_id: clip for clip in clip_manifest}

    for entry in edl:
        clip = manifest_map.get(entry.file_reference)
        if not clip:
            raise ValueError(f"Clip reference '{entry.file_reference}' not found in manifest.")

        # Create time range
        frame_rate = clip.frame_rate
        start_time = otio.opentime.from_seconds(entry.start_trim, frame_rate)
        duration = otio.opentime.from_seconds(entry.duration, frame_rate)
        time_range = otio.opentime.TimeRange(start_time, duration)

        # Create media reference
        media_reference = otio.schema.ExternalReference(
            target_url=f"file://{clip.absolute_path}" if not clip.absolute_path.startswith("file://") else clip.absolute_path
        )

        # Create video clip
        video_clip = otio.schema.Clip(
            name=clip.file_name,
            media_reference=media_reference,
            source_range=time_range,
        )
        video_track.append(video_clip)

        # Create audio clip if clip has audio
        if clip.has_audio:
            audio_clip = otio.schema.Clip(
                name=clip.file_name,
                media_reference=media_reference,
                source_range=time_range,
            )
            audio_track.append(audio_clip)
        else:
            # If no audio, we must add a gap to the audio track to keep sync
            gap = otio.schema.Gap(source_range=time_range)
            audio_track.append(gap)

    return timeline


def edl_to_otio(
    edl: Optional[Union[EditDecisionList, Sequence[dict[str, Any]], str]] = None,
    clip_manifest: Optional[Union[list[ClipManifestEntry], Sequence[dict[str, Any]], str]] = None,
    output_dir: Optional[str] = None,
    tool_context: Optional[Any] = None,
) -> str:
    """ADK Tool: Convert an Edit Decision List to OpenTimelineIO (.otio) format.

    Args:
        edl: Explicit EditDecisionList instance, list of dicts, or file path. If None, reads from session state.
        clip_manifest: List of ClipManifestEntry objects, list of dicts, or file path. If None, reads from session state.
        output_dir: Custom output directory. If None, uses STAGING_DIR.
        tool_context: Optional ADK ToolContext.

    Returns:
        String absolute path to the generated .otio file.
    """
    state = tool_context.state if tool_context is not None else {}

    # 1. Resolve EDL
    resolved_edl: Optional[EditDecisionList] = None
    if edl is not None:
        if isinstance(edl, EditDecisionList):
            resolved_edl = edl
        elif isinstance(edl, str):
            resolved_edl = EditDecisionList.from_file(edl)
        elif isinstance(edl, Sequence):
            resolved_edl = EditDecisionList.from_dict_list(list(edl))
    else:
        resolved_edl = get_edl_from_state(state)

    if not resolved_edl:
        raise ValueError("No Edit Decision List provided or found in session state.")

    # 2. Resolve Clip Manifest
    resolved_manifest: Optional[list[ClipManifestEntry]] = None
    if clip_manifest is not None:
        if isinstance(clip_manifest, list) and all(isinstance(x, ClipManifestEntry) for x in clip_manifest):
            resolved_manifest = clip_manifest
        elif isinstance(clip_manifest, Sequence):
            resolved_manifest = [ClipManifestEntry.model_validate(x) for x in clip_manifest]
    else:
        resolved_manifest = get_clip_manifest_from_state(state)

    if not resolved_manifest:
        raise ValueError("No clip manifest provided or found in session state.")

    # 3. Create Timeline
    timeline = _create_otio_timeline(resolved_edl, resolved_manifest)

    # 4. Save to output directory
    out_dir = Path(output_dir) if output_dir else settings.STAGING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = out_dir / f"{timeline.name.replace(' ', '_').lower()}.otio"
    
    otio.adapters.write_to_file(timeline, str(file_path))

    # 5. Persist to session state
    if tool_context is not None:
        tool_context.state["otio_file_path"] = str(file_path)
        logger.info("Saved OTIO file to %s and updated session state.", file_path)

    return str(file_path)


def load_otio_timeline(file_path: str) -> otio.schema.Timeline:
    """Load an OpenTimelineIO timeline from a file.

    Args:
        file_path: Path to the .otio file.

    Returns:
        otio.schema.Timeline object.
    """
    timeline = otio.adapters.read_from_file(file_path)
    if not isinstance(timeline, otio.schema.Timeline):
        raise ValueError(f"File {file_path} does not contain a valid OTIO Timeline.")
    return timeline
