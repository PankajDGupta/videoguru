"""Gemini Video Analysis Tool for VideoGuru (SPEC-008).

Uses Gemini 2.0 Flash multimodal API via Google GenAI SDK to analyze raw video clips,
evaluating visual clarity, motion, narrative relevance, and pacing against the creator's theme.
Enforces deterministic structured output conforming to the EDLEntry Pydantic schema with
qualitative/quantitative engagement scoring and guaranteed Files API resource cleanup.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Optional

from google.adk.tools import ToolContext
from google import genai
from google.genai import types

from config import settings
from schemas.edl import EDLEntry, TransitionIntent
from tools.clip_metadata import extract_clip_metadata, probe_video_file
from tools.directory_scanner import SUPPORTED_VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)

# In-memory analysis cache: (clip_id, theme) -> EDLEntry
_ANALYSIS_CACHE: dict[tuple[str, str], EDLEntry] = {}


def clear_analysis_cache() -> None:
    """Clear the in-memory video analysis cache."""
    _ANALYSIS_CACHE.clear()
    logger.debug("Cleared video analysis cache.")


def get_video_rotation(video_path: Path | str) -> int:
    """Extract rotation angle in degrees from video metadata (e.g. 0, 90, 180, -180, 270)."""
    try:
        data = probe_video_file(video_path)
    except Exception as exc:
        logger.warning("Failed to probe video rotation for '%s': %s", video_path, exc)
        return 0

    streams = data.get("streams", [])
    for stream in streams:
        if stream.get("codec_type") == "video":
            # 1. Check side_data_list for Display Matrix rotation
            for sd in stream.get("side_data_list", []):
                if "rotation" in sd:
                    try:
                        return int(float(sd["rotation"]))
                    except (ValueError, TypeError):
                        pass
            # 2. Check tags for rotate
            tags = stream.get("tags", {})
            if "rotate" in tags:
                try:
                    return int(float(tags["rotate"]))
                except (ValueError, TypeError):
                    pass
    return 0


def create_upright_proxy(video_path: Path, output_dir: Path) -> Path:
    """Generate an upright 720p proxy video using FFmpeg for Gemini multimodal analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = output_dir / f"proxy_{video_path.stem}.mp4"
    if proxy_path.exists() and proxy_path.stat().st_size > 0:
        logger.debug("Reusing existing upright proxy '%s'", proxy_path.name)
        return proxy_path

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_path),
        "-vf", "scale=-2:720",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "128k",
        str(proxy_path),
    ]
    logger.info("Generating upright proxy for rotated clip '%s' -> '%s'...", video_path.name, proxy_path.name)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not proxy_path.exists():
        logger.warning(
            "Failed to create upright proxy for '%s': %s. Falling back to original clip.",
            video_path.name,
            res.stderr,
        )
        return video_path
    return proxy_path

ANALYSIS_SYSTEM_INSTRUCTION = (
    "You are a senior digital media editor and viral video strategist specializing in YouTube content curation. "
    "Your responsibility is to analyze the provided raw video clip and extract the single most engaging, "
    "visually compelling, and narratively relevant continuous segment that directly supports the creator's theme.\n"
    "Evaluation criteria:\n"
    "1. Visual Quality & Motion: Look for crisp focus, dynamic subject movement, framing, and camera stability.\n"
    "2. Narrative Resonance: Ensure the chosen segment prominently highlights and aligns with the creator's theme.\n"
    "3. Pacing & Hooks: Select moments with strong visual or audio energy that immediately hook the viewer.\n"
    "4. Precision Trims: start_trim and end_trim MUST be absolute decimal SECONDS (e.g., 7.0 for 7 seconds, NEVER use minute-second notation like 0.07 for 7s). The segment duration (end_trim - start_trim) must be at least 2.0 seconds and strictly greater than 1.0s.\n"
    "5. Transition Intent: Suggest the appropriate visual transition family (cut, fade, wipe, slide, dissolve).\n"
    "6. Engagement Score: Assign a score from 0.0 to 10.0 reflecting overall visual quality and thematic alignment.\n"
    "Output must strictly conform to the provided JSON schema."
)


def build_analysis_prompt(
    theme: str,
    clip_id: str,
    duration: float,
    resolution: str = "",
    frame_rate: float = 0.0,
) -> str:
    """Construct the contextual user prompt sent to Gemini for video clip analysis."""
    lines = [
        f"Clip Identifier: {clip_id}",
        f"Clip Duration: {duration:.2f} seconds",
    ]
    if resolution:
        lines.append(f"Resolution: {resolution}")
    if frame_rate > 0:
        lines.append(f"Frame Rate: {frame_rate:.2f} fps")
    lines.extend(
        [
            f"Creator's Theme: {theme}",
            "",
            f"Analyze this video footage and select the single best continuous cut segment conforming to the schema.",
            f"Requirements:",
            f"- Set file_reference to '{clip_id}'.",
            f"- start_trim must be >= 0.0 in absolute SECONDS (e.g. 7.0 for 7 seconds, NEVER use 0.07 for 7s).",
            f"- end_trim must be > start_trim and <= {duration:.2f} in absolute SECONDS.",
            f"- The cut duration (end_trim - start_trim) MUST be between 3.0s and 7.0s (minimum 2.0s).",
            f"- scene_rationale must explain why this segment was selected and how it highlights '{theme}'.",
            f"- transition_intent must be one of: cut, fade, wipe, slide, dissolve.",
            f"- engagement_score must be a float between 0.0 and 10.0.",
            f"- playback_speed must be a float (default 1.0). Set to 2.0 or 4.0 if the segment contains slow action, walking, travel, setup, or montage that benefits from fast-forwarding for viewer retention.",
            f"- Pacing: Select an energetic, tight continuous segment between 3.0s and 7.0s to maintain strong viewer retention (avoid static cuts > 8.0s).",
        ]
    )
    return "\n".join(lines)


def poll_file_active(
    client: Any,
    file_ref: Any,
    timeout_seconds: float = 180.0,
    poll_interval: float = 2.0,
) -> Any:
    """Poll Gemini Files API until the uploaded video file enters the ACTIVE state.

    Args:
        client: google.genai.Client instance.
        file_ref: Uploaded file reference object.
        timeout_seconds: Maximum seconds to poll before timing out.
        poll_interval: Seconds to wait between polling attempts.

    Returns:
        The active file object.

    Raises:
        RuntimeError: If file processing fails or timeout is reached.
    """
    file_name = getattr(file_ref, "name", None) or str(file_ref)
    start_time = time.time()

    current_file = file_ref
    while True:
        state = getattr(current_file, "state", None)
        state_str = getattr(state, "name", str(state)).upper() if state is not None else "ACTIVE"

        if state_str == "ACTIVE":
            logger.debug("Gemini file '%s' is ACTIVE and ready for analysis.", file_name)
            return current_file

        if state_str == "FAILED":
            error_msg = getattr(current_file, "error", "Unknown processing error")
            raise RuntimeError(f"Gemini Files API failed to process video '{file_name}': {error_msg}")

        elapsed = time.time() - start_time
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"Timed out waiting for Gemini file '{file_name}' to become ACTIVE after {timeout_seconds:.1f}s (state: {state_str})."
            )

        logger.debug(
            "Waiting for Gemini file '%s' to become ACTIVE (current state: %s, elapsed: %.1fs)...",
            file_name,
            state_str,
            elapsed,
        )
        time.sleep(poll_interval)

        # Re-fetch file status from API
        if hasattr(client, "files") and hasattr(client.files, "get"):
            current_file = client.files.get(name=file_name)
        else:
            break

    return current_file


def mock_clip_analysis(
    clip_id: str,
    duration: float,
    theme: str,
    playback_speed: float = 1.0,
) -> EDLEntry:
    """Generate a deterministic mock EDLEntry for offline execution and testing."""
    if duration <= 1.0:
        start_trim = 0.0
        end_trim = max(0.1, duration)
    elif duration <= 6.0:
        start_trim = 0.0
        end_trim = round(duration, 2)
    else:
        # Select best 5 to 10 second segment
        start_trim = round(min(duration * 0.15, 2.0), 2)
        segment_len = min(duration - start_trim, 8.0)
        end_trim = round(start_trim + max(2.0, segment_len), 2)

    return EDLEntry(
        file_reference=clip_id,
        start_trim=start_trim,
        end_trim=min(duration, end_trim),
        scene_rationale=(
            f"Selected dynamic, high-engagement segment ({start_trim:.2f}s - {end_trim:.2f}s) "
            f"prominently capturing the theme: '{theme}'."
        ),
        transition_intent=TransitionIntent.CUT,
        engagement_score=8.5,
        playback_speed=playback_speed,
    )


def analyze_clip(
    clip_path: str | Path,
    theme: str,
    clip_id: Optional[str] = None,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    offline: bool = False,
    tool_context: Optional[ToolContext] = None,
    force_refresh: bool = False,
) -> EDLEntry:
    """Analyze a video clip using Gemini 2.0 Flash multimodal API and return a structured EDLEntry.

    Evaluates visual clarity, subject movement, narrative relevance, pacing, and audio
    quality against the specified theme. Enforces deterministic Pydantic structured output
    (`EDLEntry`) with quantitative engagement scoring (0.0 - 10.0).

    Args:
        clip_path: Local filesystem path to the raw video file.
        theme: Creator's thematic intent or highlight focus for narrative alignment.
        clip_id: Optional unique identifier for the clip. If omitted, probed from metadata.
        client: Optional google.genai.Client instance (supports dependency injection/mocking).
        model: Optional model identifier (defaults to settings.GEMINI_MODEL: 'gemini-2.0-flash').
        offline: If True, performs deterministic offline analysis without invoking Gemini API.
        tool_context: Optional ADK ToolContext providing access to session state.
        force_refresh: If True, bypasses cache and forces re-analysis.

    Returns:
        Structured EDLEntry populated with file_reference, start_trim, end_trim,
        scene_rationale, transition_intent, and engagement_score.

    Raises:
        ValueError: If clip_path is empty, not a supported video file, or theme is empty.
        FileNotFoundError: If the clip file does not exist on disk.
        RuntimeError: If Gemini API call or video file processing fails.
    """
    # 1. Validate clip path
    if isinstance(clip_path, str):
        cleaned_path = clip_path.strip()
        if not cleaned_path:
            raise ValueError("clip_path must be a non-empty string.")
        target_path = Path(cleaned_path).resolve()
    else:
        target_path = clip_path.resolve()

    if not target_path.exists():
        raise FileNotFoundError(f"Video clip does not exist: {target_path}")

    if not target_path.is_file():
        raise ValueError(f"Target clip path is not a regular file: {target_path}")

    if target_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        exts = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported video format '{target_path.suffix}'. Supported formats: {exts}"
        )

    # 2. Validate / resolve theme
    cleaned_theme = theme.strip() if isinstance(theme, str) else ""
    if not cleaned_theme and tool_context is not None:
        state_theme = tool_context.state.get("theme")
        if state_theme and isinstance(state_theme, str):
            cleaned_theme = state_theme.strip()

    if not cleaned_theme:
        raise ValueError(
            "Theme must be a non-empty string. Please provide a theme or ensure 'theme' is in session state."
        )

    # 3. Probe clip metadata for duration, dimensions, and clip ID
    metadata_entry = extract_clip_metadata(str(target_path), clip_id=clip_id)
    final_clip_id = metadata_entry.clip_id
    duration = metadata_entry.duration_seconds
    resolution = metadata_entry.resolution
    frame_rate = metadata_entry.frame_rate

    # Check in-memory cache first to avoid redundant Gemini uploads/calls across loop iterations
    cache_key = (final_clip_id, cleaned_theme)
    if not force_refresh and cache_key in _ANALYSIS_CACHE:
        cached_entry = _ANALYSIS_CACHE[cache_key]
        logger.info(
            "Reusing cached analysis for clip [%s] with theme '%s': %.2fs -> %.2fs",
            final_clip_id,
            cleaned_theme,
            cached_entry.start_trim,
            cached_entry.end_trim,
        )
        _persist_to_state(tool_context, final_clip_id, cached_entry)
        return cached_entry

    logger.info(
        "Initiating video analysis for clip '%s' [%s] (%.2fs, %s) with theme '%s'",
        target_path.name,
        final_clip_id,
        duration,
        resolution,
        cleaned_theme,
    )

    # 4. Check for offline / mock execution mode
    resolved_api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    has_api_key = bool(resolved_api_key)
    is_offline = offline or (client is None and not has_api_key and offline)

    if is_offline:
        logger.info("Executing offline analysis for clip '%s' [%s]", target_path.name, final_clip_id)
        edl_entry = mock_clip_analysis(
            clip_id=final_clip_id,
            duration=duration,
            theme=cleaned_theme,
        )
        _ANALYSIS_CACHE[cache_key] = edl_entry
        _persist_to_state(tool_context, final_clip_id, edl_entry)
        return edl_entry

    if client is None and not has_api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. Please set GEMINI_API_KEY in your .env file, "
            "provide an initialized genai.Client instance, or set offline=True."
        )

    # 5. Initialize client if needed
    active_client = client if client is not None else genai.Client(api_key=resolved_api_key)
    active_model = model or settings.GEMINI_MODEL or "gemini-2.0-flash"

    # 6. Check video rotation and normalize to upright proxy if necessary
    rotation = get_video_rotation(target_path)
    if rotation % 360 != 0:
        logger.info(
            "Clip '%s' has non-zero rotation (%d°). Generating normalized upright proxy for Gemini...",
            target_path.name,
            rotation,
        )
        upload_path = create_upright_proxy(target_path, settings.STAGING_DIR)
    else:
        upload_path = target_path

    # Upload file to Gemini Files API and ensure guaranteed cleanup
    uploaded_file = None
    try:
        logger.info("Uploading clip '%s' to Gemini Files API...", upload_path.name)
        uploaded_file = active_client.files.upload(file=str(upload_path))

        # Poll until active
        active_file = poll_file_active(active_client, uploaded_file)

        # Build prompt & configuration
        prompt = build_analysis_prompt(
            theme=cleaned_theme,
            clip_id=final_clip_id,
            duration=duration,
            resolution=resolution,
            frame_rate=frame_rate,
        )

        config = types.GenerateContentConfig(
            system_instruction=ANALYSIS_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=EDLEntry,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        logger.info(
            "Calling Gemini model '%s' with structured output schema for clip [%s]...",
            active_model,
            final_clip_id,
        )
        response = active_client.models.generate_content(
            model=active_model,
            contents=[active_file, prompt],
            config=config,
        )

        # 7. Parse response
        edl_entry = _parse_gemini_response(response, final_clip_id, duration)

    except Exception as exc:
        logger.error("Gemini video analysis failed for clip '%s': %s", target_path.name, exc)
        raise RuntimeError(f"Gemini video analysis failed for clip '{target_path.name}': {exc}") from exc
    finally:
        # Guaranteed cleanup of uploaded temporary file in Gemini storage
        if uploaded_file is not None and hasattr(active_client, "files") and hasattr(active_client.files, "delete"):
            try:
                active_client.files.delete(name=uploaded_file.name)
                logger.debug("Cleaned up Gemini temporary file '%s'", getattr(uploaded_file, "name", "unknown"))
            except Exception as cleanup_err:
                logger.warning(
                    "Failed to delete Gemini temporary file '%s': %s",
                    getattr(uploaded_file, "name", "unknown"),
                    cleanup_err,
                )

    # 8. Cache result and persist in session state if tool_context available
    _ANALYSIS_CACHE[cache_key] = edl_entry
    _persist_to_state(tool_context, final_clip_id, edl_entry)

    logger.info(
        "Successfully analyzed clip [%s]: %.2fs -> %.2fs (duration: %.2fs, score: %s, transition: %s)",
        edl_entry.file_reference,
        edl_entry.start_trim,
        edl_entry.end_trim,
        edl_entry.duration,
        f"{edl_entry.engagement_score:.1f}" if edl_entry.engagement_score is not None else "N/A",
        edl_entry.transition_intent.value,
    )

    return edl_entry


def _parse_gemini_response(
    response: Any,
    expected_clip_id: str,
    clip_duration: float,
) -> EDLEntry:
    """Parse and post-validate Gemini API structured response into an EDLEntry."""
    raw_text = getattr(response, "text", None)
    parsed_obj = getattr(response, "parsed", None)

    entry: Optional[EDLEntry] = None

    if isinstance(parsed_obj, EDLEntry):
        entry = parsed_obj
    elif isinstance(parsed_obj, dict):
        entry = EDLEntry.model_validate(parsed_obj)
    elif raw_text:
        try:
            data = json.loads(raw_text)
            entry = EDLEntry.model_validate(data)
        except Exception as err:
            logger.warning("Failed to parse response.text as JSON (%s), raw text: %s", err, raw_text)

    if entry is None:
        raise ValueError(f"Gemini response did not return valid EDLEntry structured data: {response}")

    # Enforce file_reference matches the clip being analyzed
    file_ref = expected_clip_id

    # Post-validate and clamp trims to physical clip duration
    raw_start = float(entry.start_trim)
    raw_end = float(entry.end_trim)

    # Detect MM:SS notation decimal hallucination (e.g. 0.07 -> 7.0s, 0.12 -> 12.0s)
    # when raw values are < 1.0, difference < 0.60, and clip duration is large enough
    if (
        0.0 < raw_start < 1.0
        and 0.0 < raw_end < 1.0
        and (raw_end - raw_start) < 0.60
        and clip_duration >= 5.0
    ):
        scaled_start = round(raw_start * 100.0, 3)
        scaled_end = round(raw_end * 100.0, 3)
        if scaled_start < clip_duration and scaled_end <= clip_duration and scaled_end > scaled_start:
            logger.info(
                "Detected MM:SS decimal notation in Gemini response (%.3f -> %.3f). Normalized to seconds: %.2fs -> %.2fs",
                raw_start,
                raw_end,
                scaled_start,
                scaled_end,
            )
            raw_start = scaled_start
            raw_end = scaled_end

    start_trim = max(0.0, round(raw_start, 3))
    raw_end_trim = round(raw_end, 3)

    if start_trim >= clip_duration:
        start_trim = max(0.0, clip_duration - 1.0)
        end_trim = clip_duration
    else:
        end_trim = min(clip_duration, raw_end_trim)

    # Minimum cut duration enforcement (must be strictly greater than 1.0s transition duration)
    min_cut_duration = min(clip_duration, 1.2)
    if (end_trim - start_trim) < min_cut_duration:
        if start_trim + min_cut_duration <= clip_duration:
            end_trim = start_trim + min_cut_duration
        else:
            end_trim = clip_duration
            start_trim = max(0.0, end_trim - min_cut_duration)

    # Reconstruct sanitized EDLEntry
    return EDLEntry(
        file_reference=file_ref,
        start_trim=start_trim,
        end_trim=end_trim,
        scene_rationale=entry.scene_rationale.strip() or f"Selected segment for clip {file_ref}.",
        transition_intent=entry.transition_intent,
        engagement_score=entry.engagement_score,
        playback_speed=getattr(entry, "playback_speed", 1.0),
    )


def _persist_to_state(
    tool_context: Optional[ToolContext],
    clip_id: str,
    entry: EDLEntry,
) -> None:
    """Helper to store analyzed clip result in ADK session state."""
    if tool_context is None or not hasattr(tool_context, "state"):
        return

    clip_analyses = tool_context.state.setdefault("clip_analyses", {})
    clip_analyses[clip_id] = entry.model_dump()
    logger.debug("Stored analysis for clip [%s] in session state under 'clip_analyses'.", clip_id)
