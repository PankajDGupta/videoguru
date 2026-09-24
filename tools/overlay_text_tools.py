"""Overlay Text Tools for VideoGuru (SPEC-031).

Provides Gemini-powered overlay text generation and FFmpeg drawtext burn-in
for per-cut engaging YouTube Shorts-style text overlays.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from config import settings
from rendering.ffmpeg_builder import (
    build_drawtext_overlay_command,
    execute_ffmpeg_command,
)
from schemas.edl import EditDecisionList
from schemas.overlay_text import (
    YOUTUBE_SHORTS_COLORS,
    OverlayTextEntry,
    OverlayTextPlan,
    OverlayTextStyle,
    TextPosition,
)
from tools.curation_tools import get_edl_from_state

logger = logging.getLogger(__name__)

# Default overlay text display duration per cut (seconds)
DEFAULT_OVERLAY_DURATION = 3.5

# Hook text gets a bigger font
HOOK_FONT_SIZE = 56
SCENE_FONT_SIZE = 42


def _compute_cut_timeline_offsets(
    edl: EditDecisionList,
    transition_duration: float = 1.0,
) -> list[tuple[float, float]]:
    """Calculate the absolute start/end time of each cut on the rendered timeline.

    Accounts for xfade transition overlaps between adjacent cuts.

    Args:
        edl: The finalized Edit Decision List.
        transition_duration: Duration of xfade overlap between cuts (seconds).

    Returns:
        List of (absolute_start, absolute_end) tuples for each cut.
    """
    offsets: list[tuple[float, float]] = []
    running_time = 0.0

    for idx, entry in enumerate(edl.entries):
        cut_duration = entry.duration
        cut_start = running_time
        cut_end = running_time + cut_duration
        offsets.append((cut_start, cut_end))

        # After this cut, advance timeline minus transition overlap
        if idx < len(edl.entries) - 1:
            overlap = min(transition_duration, cut_duration * 0.4)
            running_time += cut_duration - overlap
        else:
            running_time += cut_duration

    return offsets


def _generate_mock_overlay_texts(
    edl: EditDecisionList,
    theme: str,
    transition_duration: float = 1.0,
) -> OverlayTextPlan:
    """Generate deterministic mock overlay texts for testing or offline mode."""
    offsets = _compute_cut_timeline_offsets(edl, transition_duration)

    # Extract theme keywords for mock text
    theme_short = theme[:60] if theme else "Highlights"

    mock_texts = [
        f"🎬 {theme_short}",
        "Back to the grind 💪",
        "The journey continues 🚀",
        "Making it happen ⚡",
        "Level up every day 🔥",
        "No days off 💯",
        "Stay focused 🎯",
        "Building the dream ✨",
        "One step at a time 🏃",
        "Never stop grinding 💎",
    ]

    entries: list[OverlayTextEntry] = []
    for idx, (cut_start, cut_end) in enumerate(offsets):
        cut_duration = cut_end - cut_start
        overlay_duration = min(DEFAULT_OVERLAY_DURATION, cut_duration * 0.8)
        text_start = cut_start + 0.3  # Small delay after cut begins
        text_end = text_start + overlay_duration

        is_hook = idx == 0
        color = YOUTUBE_SHORTS_COLORS[idx % len(YOUTUBE_SHORTS_COLORS)]
        font_size = HOOK_FONT_SIZE if is_hook else SCENE_FONT_SIZE

        style = OverlayTextStyle(
            font_family="Impact",
            font_size=font_size,
            font_color=color,
            border_color="#000000",
            border_width=3,
            shadow_x=2,
            shadow_y=2,
            position=TextPosition.UPPER_THIRD,
            box_enabled=True,
            box_color="black",
            box_opacity=0.5,
            box_border_width=15,
        )

        entries.append(OverlayTextEntry(
            cut_index=idx,
            text=mock_texts[idx % len(mock_texts)],
            start_time=round(text_start, 3),
            end_time=round(text_end, 3),
            style=style,
            is_hook=is_hook,
        ))

    return OverlayTextPlan(entries=entries, theme=theme)


def generate_overlay_texts_with_gemini(
    edl: EditDecisionList,
    theme: str,
    transition_duration: float = 1.0,
) -> OverlayTextPlan:
    """Use Gemini to generate engaging overlay text for each cut based on the theme.

    Args:
        edl: The finalized Edit Decision List.
        theme: Video theme/intent description.
        transition_duration: Transition duration for timeline offset computation.

    Returns:
        OverlayTextPlan with one entry per cut.
    """
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai not available. Falling back to mock overlay texts.")
        return _generate_mock_overlay_texts(edl, theme, transition_duration)

    api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No Gemini API key available. Using mock overlay texts.")
        return _generate_mock_overlay_texts(edl, theme, transition_duration)

    offsets = _compute_cut_timeline_offsets(edl, transition_duration)

    # Build the prompt describing each cut
    cuts_description = []
    for idx, entry in enumerate(edl.entries):
        cut_start, cut_end = offsets[idx]
        cuts_description.append(
            f"Cut {idx + 1}: Duration {entry.duration:.1f}s, "
            f"Timeline {cut_start:.1f}s-{cut_end:.1f}s, "
            f"Scene: {entry.scene_rationale}"
        )

    prompt = (
        f"You are a YouTube Shorts viral content strategist. Generate short, punchy overlay text "
        f"for each scene cut in a video.\n\n"
        f"VIDEO THEME: {theme}\n\n"
        f"CUTS:\n" + "\n".join(cuts_description) + "\n\n"
        f"RULES:\n"
        f"- Cut 1 MUST be a strong HOOK that makes viewers stop scrolling (use emojis)\n"
        f"- Each text should be 3-8 words MAX\n"
        f"- Use relevant emojis for engagement\n"
        f"- Match the energy and theme of each scene\n"
        f"- Text should feel authentic, not clickbaity\n"
        f"- Use a mix of motivation, humor, and emotion\n\n"
        f"Return ONLY a JSON array where each element has: "
        f'{{"cut_index": 0, "text": "your overlay text here", "is_hook": true/false}}\n'
        f"Return {len(edl.entries)} entries, one per cut."
    )

    try:
        client = genai.Client(api_key=api_key)

        # Use a model that's available
        model_name = settings.GEMINI_MODEL
        # Fallback to known available models if needed
        available_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]
        if model_name not in available_models:
            model_name = "gemini-2.5-flash"

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )

        # Parse the response
        response_text = response.text.strip()
        # Extract JSON from potential markdown fences
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        generated_texts = json.loads(response_text)

        entries: list[OverlayTextEntry] = []
        for item in generated_texts:
            idx = item.get("cut_index", 0)
            if idx >= len(offsets):
                continue

            cut_start, cut_end = offsets[idx]
            cut_duration = cut_end - cut_start
            overlay_duration = min(DEFAULT_OVERLAY_DURATION, cut_duration * 0.8)
            text_start = cut_start + 0.3
            text_end = text_start + overlay_duration

            is_hook = item.get("is_hook", idx == 0)
            color = YOUTUBE_SHORTS_COLORS[idx % len(YOUTUBE_SHORTS_COLORS)]
            font_size = HOOK_FONT_SIZE if is_hook else SCENE_FONT_SIZE

            style = OverlayTextStyle(
                font_family="Impact",
                font_size=font_size,
                font_color=color,
                border_color="#000000",
                border_width=3,
                shadow_x=2,
                shadow_y=2,
                position=TextPosition.UPPER_THIRD,
                box_enabled=True,
                box_color="black",
                box_opacity=0.5,
                box_border_width=15,
            )

            entries.append(OverlayTextEntry(
                cut_index=idx,
                text=item.get("text", f"Scene {idx + 1}"),
                start_time=round(text_start, 3),
                end_time=round(text_end, 3),
                style=style,
                is_hook=is_hook,
            ))

        if not entries:
            logger.warning("Gemini returned no valid overlay texts. Using mock texts.")
            return _generate_mock_overlay_texts(edl, theme, transition_duration)

        logger.info("Generated %d overlay texts via Gemini for theme: %s", len(entries), theme[:50])
        return OverlayTextPlan(entries=entries, theme=theme)

    except Exception as exc:
        logger.warning(
            "Gemini overlay text generation failed (%s). Falling back to mock texts.",
            exc,
        )
        return _generate_mock_overlay_texts(edl, theme, transition_duration)


def generate_overlay_text_plan(
    tool_context: Optional[Any] = None,
    theme: Optional[str] = None,
    offline: bool = False,
    transition_duration: float = 1.0,
) -> str:
    """ADK Tool: Generate engaging overlay text for each cut based on the video theme.

    Reads EDL and theme from session state, generates per-cut overlay text via Gemini,
    and stores the plan in session.state['overlay_texts'].

    Args:
        tool_context: ADK ToolContext for session state access.
        theme: Optional theme override.
        offline: If True, uses mock text generation.
        transition_duration: Duration of xfade transitions.

    Returns:
        Summary string of generated overlay texts.
    """
    state = tool_context.state if tool_context and hasattr(tool_context, "state") else {}

    # Resolve EDL
    edl = get_edl_from_state(state)
    if not edl or not edl.entries:
        return "No EDL found in session state. Cannot generate overlay texts."

    # Resolve theme
    resolved_theme = theme or state.get("theme", "")
    if not resolved_theme:
        resolved_theme = "Engaging YouTube Shorts highlights"

    # Generate overlay texts
    if offline:
        plan = _generate_mock_overlay_texts(edl, resolved_theme, transition_duration)
    else:
        plan = generate_overlay_texts_with_gemini(edl, resolved_theme, transition_duration)

    # Store in session state
    if tool_context and hasattr(tool_context, "state") and isinstance(state, dict):
        state["overlay_texts"] = plan.to_dict_list()
        logger.info("Stored %d overlay text entries in session state.", len(plan))

    # Build summary
    lines = [f"Generated {len(plan)} overlay texts for theme: '{resolved_theme[:50]}'"]
    for entry in plan.entries:
        hook_marker = " [HOOK]" if entry.is_hook else ""
        lines.append(
            f"  Cut {entry.cut_index + 1}: \"{entry.text}\" "
            f"({entry.start_time:.1f}s-{entry.end_time:.1f}s, "
            f"color: {entry.style.font_color}, size: {entry.style.font_size}px){hook_marker}"
        )

    return "\n".join(lines)


def burn_overlay_text(
    video_path: Union[str, Path],
    overlay_plan: Union[OverlayTextPlan, list[dict[str, Any]]],
    output_path: Union[str, Path],
) -> str:
    """Burn overlay text entries into a video using FFmpeg drawtext filters.

    Args:
        video_path: Source video file path.
        overlay_plan: OverlayTextPlan or list of overlay entry dicts.
        output_path: Destination path for the output video.

    Returns:
        Absolute string path to the generated video with overlay text.

    Raises:
        ValueError: If paths are empty or no overlay entries.
        FileNotFoundError: If source video does not exist.
        RuntimeError: If FFmpeg execution fails.
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

    # Convert OverlayTextPlan to list of dicts
    if isinstance(overlay_plan, OverlayTextPlan):
        entries_dicts = overlay_plan.to_dict_list()
    elif isinstance(overlay_plan, list):
        entries_dicts = overlay_plan
    else:
        raise TypeError(f"Expected OverlayTextPlan or list, got {type(overlay_plan).__name__}")

    if not entries_dicts:
        raise ValueError("No overlay text entries to burn.")

    cmd = build_drawtext_overlay_command(
        video_path=vid_file,
        overlay_entries=entries_dicts,
        output_path=out_file,
    )

    logger.info(
        "Burning %d overlay texts into '%s' -> '%s'",
        len(entries_dicts),
        vid_file.name,
        out_file.name,
    )
    execute_ffmpeg_command(cmd)

    return str(out_file)
