"""Audio Ducking Tool for VideoGuru (SPEC-018).

Provides an ADK tool function to automatically compress (duck) background music volume
beneath dialogue / speech using FFmpeg's sidechaincompress audio filter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union
import uuid

from config import settings
from rendering.ffmpeg_builder import build_ducking_command, execute_ffmpeg_command

logger = logging.getLogger(__name__)


def apply_audio_ducking(
    video_path: Optional[Union[str, Path]] = None,
    music_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.05,
    ratio: float = 4.0,
    attack: float = 20.0,
    release: float = 250.0,
    music_volume: float = 0.3,
    tool_context: Optional[Any] = None,
) -> str:
    """ADK Tool: Apply sidechain compression audio ducking of background music under video speech.

    The primary video stream audio acts as the sidechain trigger. When speech occurs,
    background music volume is dynamically attenuated, maintaining clear dialogue intelligibility
    and smooth music transitions.

    Args:
        video_path: Primary video file path. If None, retrieved from tool_context state
            ('transition_rendered_path' or 'video_path').
        music_path: Background music file path. If None, retrieved from tool_context state
            ('music_path') or auto-discovered at settings.MEDIA_INPUT_DIR / 'music.mp3'.
        output_path: Destination path for the ducked video. If None, generated in STAGING_DIR.
        threshold: Sidechain compressor threshold level in range (0.0, 1.0] (default: 0.05).
        ratio: Compression ratio >= 1.0 (default: 4.0).
        attack: Attack time in milliseconds > 0.0 (default: 20.0).
        release: Release time in milliseconds > 0.0 (default: 250.0).
        music_volume: Pre-compression gain multiplier >= 0.0 for background music (default: 0.3).
        tool_context: Optional ADK ToolContext providing access to session state.

    Returns:
        Absolute string path to the rendered ducked video (or original video if no music is used).

    Raises:
        ValueError: If video_path cannot be resolved, paths are empty, or parameters are invalid.
        FileNotFoundError: If resolved video_path or explicit music_path does not exist on disk.
        RuntimeError: If FFmpeg execution fails.
    """
    # 1. Resolve tool_context state helper
    state: Optional[Any] = None
    if tool_context is not None:
        if hasattr(tool_context, "state") and tool_context.state is not None:
            state = tool_context.state
        elif isinstance(tool_context, dict):
            state = tool_context

    # 2. Resolve video_path
    resolved_video_str: Optional[str] = None
    if video_path is not None:
        cleaned_video = str(video_path).strip()
        if not cleaned_video:
            raise ValueError("video_path cannot be empty.")
        resolved_video_str = cleaned_video
    elif state is not None:
        video_from_state = state.get("transition_rendered_path") or state.get("video_path")
        if video_from_state:
            cleaned_video = str(video_from_state).strip()
            if cleaned_video:
                resolved_video_str = cleaned_video

    if not resolved_video_str:
        raise ValueError(
            "video_path must be provided or available in tool_context state "
            "('transition_rendered_path' or 'video_path')."
        )

    resolved_video_path = Path(resolved_video_str).resolve()
    if not resolved_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {resolved_video_path}")

    # 3. Resolve music_path
    resolved_music_path: Optional[Path] = None
    is_explicit_music = False

    if music_path is not None:
        cleaned_music = str(music_path).strip()
        if not cleaned_music:
            raise ValueError("music_path cannot be empty.")
        resolved_music_path = Path(cleaned_music).resolve()
        is_explicit_music = True
    elif state is not None and state.get("music_path"):
        cleaned_music = str(state.get("music_path")).strip()
        if cleaned_music:
            resolved_music_path = Path(cleaned_music).resolve()
            is_explicit_music = True
    else:
        # Check auto-discovery in state media_dir or MEDIA_INPUT_DIR
        search_dirs = []
        if state is not None and state.get("media_dir"):
            search_dirs.append(Path(state["media_dir"]))
        search_dirs.append(settings.MEDIA_INPUT_DIR)

        for sdir in search_dirs:
            candidate_music = (sdir / "music.mp3").resolve()
            if candidate_music.exists():
                resolved_music_path = candidate_music
                is_explicit_music = False
                logger.info("Auto-discovered background music at %s", resolved_music_path)
                break

    # 4. Handle missing music
    if resolved_music_path is None:
        logger.warning(
            "No background music file provided or found in %s. "
            "Returning original video path without ducking.",
            settings.MEDIA_INPUT_DIR,
        )
        if state is not None:
            state["ducked_video_path"] = str(resolved_video_path)
        return str(resolved_video_path)

    if is_explicit_music and not resolved_music_path.exists():
        raise FileNotFoundError(f"Specified music file not found: {resolved_music_path}")

    # 5. Resolve output_path
    if output_path is not None:
        cleaned_output = str(output_path).strip()
        if not cleaned_output:
            raise ValueError("output_path cannot be empty.")
        resolved_output_path = Path(cleaned_output).resolve()
    else:
        settings.STAGING_DIR.mkdir(parents=True, exist_ok=True)
        unique_name = f"ducked_video_{uuid.uuid4().hex[:8]}.mp4"
        resolved_output_path = (settings.STAGING_DIR / unique_name).resolve()

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 6. Build and execute FFmpeg ducking command
    cmd = build_ducking_command(
        video_path=resolved_video_path,
        music_path=resolved_music_path,
        output_path=resolved_output_path,
        threshold=threshold,
        ratio=ratio,
        attack=attack,
        release=release,
        music_volume=music_volume,
    )

    logger.info("Applying audio ducking: video=%s, music=%s -> output=%s",
                resolved_video_path, resolved_music_path, resolved_output_path)
    execute_ffmpeg_command(cmd)

    # 7. Persist to session state if available
    if state is not None:
        state["ducked_video_path"] = str(resolved_output_path)
        logger.info("Updated tool_context state['ducked_video_path'] = %s", resolved_output_path)

    return str(resolved_output_path)
