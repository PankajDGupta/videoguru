"""Motion Graphics Tools for VideoGuru (SPEC-035).

Analyzes the rendered video with Gemini's multimodal understanding, designs a small set of
theme-related animated motion graphics (kinetic titles, lower thirds, stat callouts, badges and
a progress bar) tied to the moments they describe, and composites them with FFmpeg.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pydantic import BaseModel, Field

from config import settings
from rendering.ffmpeg_builder import (
    build_motion_graphics_command,
    execute_ffmpeg_command,
    find_ffmpeg_executable,
)
from schemas.edl import EditDecisionList
from schemas.motion_graphics import (
    DEFAULT_PRIMARY_COLOR,
    DEFAULT_SECONDARY_COLOR,
    SHARED_ZONE_GROUPS,
    MotionGraphicElement,
    MotionGraphicsPlan,
    MotionGraphicType,
)
from tools.curation_tools import get_edl_from_state
from tools.overlay_text_tools import _compute_cut_timeline_offsets

logger = logging.getLogger(__name__)

# Shortest / longest time a text graphic stays on screen (seconds)
MIN_GRAPHIC_DURATION = 1.6
MAX_GRAPHIC_DURATION = 4.5

# Per-type copy limits enforced after Gemini responds (renderer fits the font, but keeps copy punchy)
MAX_TEXT_CHARS: dict[MotionGraphicType, int] = {
    MotionGraphicType.KINETIC_TITLE: 24,
    MotionGraphicType.LOWER_THIRD: 28,
    MotionGraphicType.STAT_CALLOUT: 12,
    MotionGraphicType.CORNER_BADGE: 14,
}
MAX_SUBTEXT_CHARS = 40

ANALYSIS_SYSTEM_INSTRUCTION = (
    "You are a motion graphics designer for short-form video. You watch a finished edit and design "
    "a few tasteful animated graphics that reinforce its theme. You never invent facts, and you never "
    "transcribe or translate speech."
)


class _GeminiGraphic(BaseModel):
    """One graphic as returned by Gemini structured output."""

    graphic_type: MotionGraphicType
    start_time: float = Field(description="Seconds on the video timeline when the graphic starts.")
    end_time: float = Field(description="Seconds on the video timeline when the graphic ends.")
    text: Optional[str] = None
    subtext: Optional[str] = None
    cut_index: Optional[int] = None
    rationale: str = ""


class _GeminiMotionPlan(BaseModel):
    """Whole-plan structured output schema for Gemini."""

    content_summary: str = ""
    primary_color: str = DEFAULT_PRIMARY_COLOR
    secondary_color: str = DEFAULT_SECONDARY_COLOR
    graphics: list[_GeminiGraphic] = Field(default_factory=list)


def _max_elements(video_duration: float) -> int:
    """Roughly one graphic per five seconds, between 2 and 10 in total."""
    return max(2, min(10, int(video_duration // 5)))


def _truncate_copy(text: str, limit: int) -> str:
    """Shorten copy to `limit` characters, cutting at a word boundary when one exists."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if text[limit] != " " and " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip()


def get_video_duration(video_path: Union[str, Path]) -> float:
    """Return the duration in seconds of a video file via ffprobe."""
    from tools.clip_metadata import probe_video_file

    info = probe_video_file(video_path)
    duration = float(info.get("format", {}).get("duration") or 0.0)
    if duration <= 0.0:
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video" and stream.get("duration"):
                duration = float(stream["duration"])
                break
    if duration <= 0.0:
        raise RuntimeError(f"Could not determine duration of '{video_path}'.")
    return duration


def _create_analysis_proxy(video_path: Path, output_dir: Path) -> Path:
    """Create a small silent low-fps proxy of the video so the Gemini upload is quick and cheap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_path = output_dir / f"{video_path.stem}_mg_analysis.mp4"
    cmd = [
        find_ffmpeg_executable(),
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        "scale=480:480:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-r",
        "10",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-pix_fmt",
        "yuv420p",
        str(proxy_path),
    ]
    execute_ffmpeg_command(cmd, timeout=300.0)
    return proxy_path


def build_motion_graphics_prompt(
    theme: str,
    video_duration: float,
    is_shorts: bool,
    resolution: str,
    edl: Optional[EditDecisionList] = None,
    transition_duration: float = 1.0,
    overlay_entries: Optional[Sequence[dict[str, Any]]] = None,
) -> str:
    """Build the Gemini prompt describing the video and the graphics the model may design."""
    orientation = "vertical 9:16 YouTube Short" if is_shorts else "landscape 16:9 video"
    max_elements = _max_elements(video_duration)

    cut_lines: list[str] = []
    if edl and edl.entries:
        offsets = _compute_cut_timeline_offsets(edl, transition_duration)
        for idx, (entry, (cut_start, cut_end)) in enumerate(zip(edl.entries, offsets)):
            cut_lines.append(f"Cut {idx}: {cut_start:.1f}s-{cut_end:.1f}s - {entry.scene_rationale}")

    overlay_lines = [
        f'"{e.get("text", "")}" at {float(e.get("start_time", 0.0)):.1f}s-{float(e.get("end_time", 0.0)):.1f}s'
        for e in (overlay_entries or [])
    ]

    sections = [
        f"Watch the attached video ({video_duration:.1f}s, {orientation}, {resolution}) and design a small set "
        f"of ANIMATED motion graphics that are related to its theme and to what is actually on screen.",
        f"THEME / STORYLINE:\n{theme or 'Not specified - infer it from the footage.'}",
    ]
    if cut_lines:
        sections.append("CUT TIMELINE FROM THE EDIT:\n" + "\n".join(cut_lines))
    if overlay_lines:
        sections.append(
            "TEXT OVERLAYS ALREADY IN THE VIDEO (do not repeat their wording):\n" + "\n".join(overlay_lines)
        )
    sections.append(
        "GRAPHIC TYPES YOU MAY USE:\n"
        f"- kinetic_title: a headline of at most 3 words (max {MAX_TEXT_CHARS[MotionGraphicType.KINETIC_TITLE]} "
        "characters) shown large in the centre for a key moment or chapter. Lasts 2-3.5s.\n"
        f"- lower_third: text = the place / subject / activity name (max "
        f"{MAX_TEXT_CHARS[MotionGraphicType.LOWER_THIRD]} characters), optional subtext = one short context "
        f"line (max {MAX_SUBTEXT_CHARS} characters). Lasts 3-4.5s.\n"
        f"- stat_callout: text = a short number or keyword (max "
        f"{MAX_TEXT_CHARS[MotionGraphicType.STAT_CALLOUT]} characters, e.g. '5 KM', 'DAY 3'), subtext = its label. "
        "Use a number ONLY if it is visible on screen or stated in the theme - otherwise use a keyword. "
        "Lasts 2.5-3.5s.\n"
        f"- corner_badge: a tag of at most {MAX_TEXT_CHARS[MotionGraphicType.CORNER_BADGE]} characters such as "
        "'DAY 1', 'LIVE', 'BEHIND THE SCENES'. Lasts 2.5-4s.\n"
        "- progress_bar: no text; spans the whole video (start 0, end = video duration). Optional, at most one."
    )
    sections.append(
        "RULES:\n"
        "1. Tie every graphic to the moment it describes: start it when that action or subject appears.\n"
        f"2. Use at most {max_elements} graphics in total and keep the video breathing - do not cover it "
        "with graphics. Never place two graphics of the same type at the same time, and never overlap a "
        "kinetic_title with a stat_callout.\n"
        "3. Text must be plain letters, digits and basic punctuation. NO emojis.\n"
        "4. Do NOT transcribe or translate speech. Do NOT invent names, places or statistics.\n"
        "5. Pick primary_color and secondary_color as vivid '#RRGGBB' values that match the theme's mood and "
        "contrast with the footage (e.g. warm orange/gold for fitness, teal/cyan for travel and water, "
        "green for nature).\n"
        f"6. All times are seconds on the video timeline, between 0 and {video_duration:.1f}.\n"
        "7. content_summary: one sentence describing what the video shows."
    )
    return "\n\n".join(sections)


def sanitize_motion_graphics_plan(
    plan: MotionGraphicsPlan,
    video_duration: float,
) -> MotionGraphicsPlan:
    """Clamp, trim and de-conflict a motion graphics plan so it always renders cleanly.

    - Clamps every element to the video timeline and drops graphics that are too short to read.
    - Caps the lifetime of text graphics and the length of their copy.
    - Removes overlaps between graphics of the same type and between graphics sharing a screen zone.
    - Limits the total number of graphics to a density suited to the video length.
    """
    limit = _max_elements(video_duration)
    kept: list[MotionGraphicElement] = []

    for element in sorted(plan.elements, key=lambda e: e.start_time):
        start = max(0.0, element.start_time)
        end = min(video_duration, element.end_time)
        is_bar = element.graphic_type == MotionGraphicType.PROGRESS_BAR

        if not is_bar and end - start > MAX_GRAPHIC_DURATION:
            end = start + MAX_GRAPHIC_DURATION
        if end - start < (MIN_GRAPHIC_DURATION if not is_bar else 1.0):
            continue

        updates: dict[str, Any] = {"start_time": round(start, 3), "end_time": round(end, 3)}
        max_chars = MAX_TEXT_CHARS.get(element.graphic_type)
        if max_chars and element.text and len(element.text) > max_chars:
            updates["text"] = _truncate_copy(element.text, max_chars)
        if element.subtext and len(element.subtext) > MAX_SUBTEXT_CHARS:
            updates["subtext"] = _truncate_copy(element.subtext, MAX_SUBTEXT_CHARS)
        if is_bar:
            updates["text"] = None
            updates["subtext"] = None
        elif element.graphic_type == MotionGraphicType.CORNER_BADGE:
            updates["subtext"] = None
        candidate = element.model_copy(update=updates)

        conflict = False
        for other in kept:
            overlaps = candidate.start_time < other.end_time and other.start_time < candidate.end_time
            if not overlaps:
                continue
            same_type = other.graphic_type == candidate.graphic_type
            shared_zone = any(
                candidate.graphic_type in group and other.graphic_type in group
                for group in SHARED_ZONE_GROUPS
            )
            if same_type or shared_zone:
                conflict = True
                break
        if conflict:
            continue
        kept.append(candidate)

    # The (unobtrusive) progress bar does not count towards the density cap; drop the latest other graphics
    bars = [e for e in kept if e.graphic_type == MotionGraphicType.PROGRESS_BAR]
    others = [e for e in kept if e.graphic_type != MotionGraphicType.PROGRESS_BAR]
    if len(others) > limit:
        kept = sorted(bars + others[:limit], key=lambda e: e.start_time)

    return plan.model_copy(update={"elements": kept})


def _generate_mock_motion_graphics(
    edl: Optional[EditDecisionList],
    theme: str,
    video_duration: float,
    transition_duration: float = 1.0,
) -> MotionGraphicsPlan:
    """Deterministic, fact-safe motion graphics plan for offline mode and API failures.

    Uses only information known without watching the video: the theme text and the cut count.
    """
    theme_clean = " ".join((theme or "").split()) or "Highlights"
    short_theme = _truncate_copy(theme_clean, MAX_TEXT_CHARS[MotionGraphicType.LOWER_THIRD])
    badge_label = _truncate_copy(" ".join(theme_clean.split()[:2]), MAX_TEXT_CHARS[MotionGraphicType.CORNER_BADGE])

    cut_count = len(edl.entries) if edl and edl.entries else 0
    offsets = _compute_cut_timeline_offsets(edl, transition_duration) if cut_count else []

    elements: list[MotionGraphicElement] = [
        MotionGraphicElement(
            graphic_type=MotionGraphicType.PROGRESS_BAR,
            start_time=0.0,
            end_time=video_duration,
            rationale="Shows playback progress to encourage watching to the end.",
        )
    ]

    badge_start = offsets[0][0] + 0.5 if offsets else 0.5
    if badge_label:
        elements.append(MotionGraphicElement(
            graphic_type=MotionGraphicType.CORNER_BADGE,
            start_time=badge_start,
            end_time=badge_start + 3.0,
            text=badge_label,
            cut_index=0 if offsets else None,
            rationale="Theme tag introduced at the start.",
        ))

    if cut_count >= 2:
        last_start, last_end = offsets[-1]
        lt_start = max(last_start + 0.4, video_duration - 4.5)
        elements.append(MotionGraphicElement(
            graphic_type=MotionGraphicType.LOWER_THIRD,
            start_time=lt_start,
            end_time=min(video_duration, lt_start + 3.5),
            text=short_theme,
            subtext=f"{cut_count} moments",
            cut_index=cut_count - 1,
            rationale="Closing title card summarising the theme.",
        ))

    plan = MotionGraphicsPlan(
        elements=elements,
        theme=theme,
        content_summary="",
    )
    return sanitize_motion_graphics_plan(plan, video_duration)


def _resolve_api_key() -> Optional[str]:
    return settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")


def _parse_gemini_motion_response(response: Any, theme: str) -> MotionGraphicsPlan:
    """Convert a Gemini structured response into a MotionGraphicsPlan, skipping malformed graphics."""
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, _GeminiMotionPlan):
        payload = parsed
    else:
        text = (getattr(response, "text", "") or "").strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        payload = _GeminiMotionPlan.model_validate(json.loads(text))

    elements: list[MotionGraphicElement] = []
    for graphic in payload.graphics:
        try:
            elements.append(MotionGraphicElement(**graphic.model_dump()))
        except ValueError as exc:
            logger.warning("Skipping malformed motion graphic from Gemini (%s): %s", graphic.graphic_type, exc)

    return MotionGraphicsPlan(
        elements=elements,
        theme=theme,
        content_summary=payload.content_summary,
        primary_color=payload.primary_color,
        secondary_color=payload.secondary_color,
    )


def analyze_video_for_motion_graphics(
    video_path: Union[str, Path],
    theme: str,
    edl: Optional[EditDecisionList] = None,
    transition_duration: float = 1.0,
    target_resolution: Optional[str] = None,
    is_shorts: bool = False,
    overlay_entries: Optional[Sequence[dict[str, Any]]] = None,
    offline: bool = False,
) -> MotionGraphicsPlan:
    """Analyze a rendered video with Gemini and design theme-related animated motion graphics.

    Falls back to a deterministic plan built from the theme and EDL when running offline, when no API
    key is configured, or when the Gemini call fails, so rendering can always continue.

    Args:
        video_path: The rendered video (with final timeline timing) to analyze.
        theme: The creator's video theme / intent.
        edl: Optional finalized EDL; supplies cut timings and scene rationales.
        transition_duration: xfade overlap used when rendering, for cut timeline offsets.
        target_resolution: Video resolution string, e.g. '1080x1920'.
        is_shorts: Whether the video is a vertical YouTube Short.
        overlay_entries: Existing overlay-text entries, so Gemini avoids repeating them.
        offline: Force the deterministic plan.

    Returns:
        A sanitized MotionGraphicsPlan.
    """
    src = Path(str(video_path)).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source video not found: {src}")

    duration = get_video_duration(src)
    is_shorts_mode = is_shorts or (
        target_resolution is not None and str(target_resolution).strip().lower() == "1080x1920"
    )
    resolution = target_resolution or ("1080x1920" if is_shorts_mode else "1920x1080")

    api_key = _resolve_api_key()
    if offline or not api_key:
        logger.info("Motion graphics: using deterministic plan (offline=%s, api_key=%s).", offline, bool(api_key))
        return _generate_mock_motion_graphics(edl, theme, duration, transition_duration)

    try:
        from google import genai
        from google.genai import types
        from tools.video_analysis import poll_file_active
    except ImportError:
        logger.warning("google-genai not available. Using deterministic motion graphics plan.")
        return _generate_mock_motion_graphics(edl, theme, duration, transition_duration)

    client = genai.Client(api_key=api_key)
    uploaded_file = None
    try:
        proxy_path = _create_analysis_proxy(src, settings.STAGING_DIR)
        logger.info("Uploading '%s' to Gemini for motion graphics analysis...", proxy_path.name)
        uploaded_file = client.files.upload(file=str(proxy_path))
        active_file = poll_file_active(client, uploaded_file)

        prompt = build_motion_graphics_prompt(
            theme=theme,
            video_duration=duration,
            is_shorts=is_shorts_mode,
            resolution=resolution,
            edl=edl,
            transition_duration=transition_duration,
            overlay_entries=overlay_entries,
        )
        config = types.GenerateContentConfig(
            system_instruction=ANALYSIS_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_GeminiMotionPlan,
            temperature=0.4,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[active_file, prompt],
            config=config,
        )
        plan = sanitize_motion_graphics_plan(_parse_gemini_motion_response(response, theme), duration)
        if not plan.elements:
            logger.warning("Gemini returned no usable motion graphics. Using deterministic plan.")
            return _generate_mock_motion_graphics(edl, theme, duration, transition_duration)
        logger.info("Designed %d motion graphics via Gemini for theme: %s", len(plan), (theme or "")[:50])
        return plan
    except Exception as exc:
        logger.warning("Gemini motion graphics analysis failed (%s). Using deterministic plan.", exc)
        return _generate_mock_motion_graphics(edl, theme, duration, transition_duration)
    finally:
        if uploaded_file is not None:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception as cleanup_err:
                logger.warning("Failed to delete Gemini temporary file: %s", cleanup_err)


def burn_motion_graphics(
    video_path: Union[str, Path],
    motion_plan: Union[MotionGraphicsPlan, list[dict[str, Any]]],
    output_path: Union[str, Path],
    target_resolution: Optional[str] = None,
    is_shorts: bool = False,
) -> str:
    """Composite the animated motion graphics onto a video with FFmpeg.

    Args:
        video_path: Source video file path.
        motion_plan: MotionGraphicsPlan or the list of dicts produced by `to_dict_list()`.
        output_path: Destination path for the output video.
        target_resolution: Video resolution string, e.g. '1080x1920'.
        is_shorts: Whether the target video is a YouTube Short.

    Returns:
        Absolute string path to the generated video.

    Raises:
        ValueError: If paths are empty or there are no graphics to draw.
        FileNotFoundError: If the source video does not exist.
        RuntimeError: If FFmpeg fails.
    """
    str_video = str(video_path).strip()
    str_out = str(output_path).strip()
    if not str_video:
        raise ValueError("video_path cannot be empty.")
    if not str_out:
        raise ValueError("output_path cannot be empty.")

    vid_file = Path(str_video).resolve()
    out_file = Path(str_out).resolve()
    if not vid_file.exists():
        raise FileNotFoundError(f"Source video not found: {vid_file}")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(motion_plan, MotionGraphicsPlan):
        elements = motion_plan.to_dict_list()
    elif isinstance(motion_plan, list):
        elements = motion_plan
    else:
        raise TypeError(f"Expected MotionGraphicsPlan or list, got {type(motion_plan).__name__}")
    if not elements:
        raise ValueError("No motion graphics to burn.")

    cmd = build_motion_graphics_command(
        video_path=vid_file,
        graphic_elements=elements,
        output_path=out_file,
        target_resolution=target_resolution,
        is_shorts=is_shorts,
    )
    logger.info("Compositing %d motion graphics into '%s' -> '%s'", len(elements), vid_file.name, out_file.name)
    execute_ffmpeg_command(cmd, timeout=600.0)
    return str(out_file)


def generate_motion_graphics_plan(
    tool_context: Optional[Any] = None,
    video_path: Optional[str] = None,
    theme: Optional[str] = None,
    offline: bool = False,
    transition_duration: float = 1.0,
) -> str:
    """ADK Tool: Analyze the rendered video and design theme-related animated motion graphics.

    Reads the theme, EDL and rendered video path from session state (unless given), asks Gemini to
    design graphics tied to what is on screen, and stores the plan in session.state['motion_graphics'].

    Args:
        tool_context: ADK ToolContext for session state access.
        video_path: Optional video to analyze (defaults to the state's rendered transition video).
        theme: Optional theme override.
        offline: If True, uses the deterministic plan.
        transition_duration: xfade overlap used when the video was rendered.

    Returns:
        Summary string of the designed graphics.
    """
    state = tool_context.state if tool_context and hasattr(tool_context, "state") else {}

    resolved_video = video_path or state.get("transition_rendered_path") or state.get("final_video_path")
    if not resolved_video:
        return "No rendered video available to analyze. Render the transitions first."

    resolved_theme = theme or state.get("theme", "") or "Engaging video highlights"
    resolution = state.get("target_resolution")
    is_shorts = str(state.get("video_type", "")).strip().lower() in ("shorts", "vertical") or (
        resolution is not None and str(resolution).strip().lower() == "1080x1920"
    )
    edl = get_edl_from_state(state)

    plan = analyze_video_for_motion_graphics(
        resolved_video,
        resolved_theme,
        edl=edl,
        transition_duration=transition_duration,
        target_resolution=resolution,
        is_shorts=is_shorts,
        overlay_entries=state.get("overlay_texts"),
        offline=offline,
    )

    if tool_context and hasattr(tool_context, "state") and isinstance(state, dict):
        state["motion_graphics"] = plan.to_dict_list()
        state["motion_graphics_palette"] = [plan.primary_color, plan.secondary_color]

    lines = [f"Designed {len(plan)} motion graphics for theme: '{resolved_theme[:50]}'"]
    if plan.content_summary:
        lines.append(f"  Video shows: {plan.content_summary}")
    for element in plan.elements:
        copy = f' "{element.text}"' if element.text else ""
        lines.append(
            f"  {element.graphic_type.value}{copy} ({element.start_time:.1f}s-{element.end_time:.1f}s)"
        )
    return "\n".join(lines)
