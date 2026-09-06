"""VideoGuru Rendering Package (SPEC-016)."""

from rendering.ffmpeg_builder import (
    TRANSITION_MAP,
    VALID_XFADE_TRANSITIONS,
    build_audio_crossfade,
    build_caption_burn_command,
    build_ducking_command,
    build_trim_command,
    build_xfade_chain,
    calculate_xfade_offsets,
    escape_subtitles_path,
    execute_ffmpeg_command,
    find_ffmpeg_executable,
    map_transition_intent,
)

__all__ = [
    "TRANSITION_MAP",
    "VALID_XFADE_TRANSITIONS",
    "build_audio_crossfade",
    "build_caption_burn_command",
    "build_ducking_command",
    "build_trim_command",
    "build_xfade_chain",
    "calculate_xfade_offsets",
    "escape_subtitles_path",
    "execute_ffmpeg_command",
    "find_ffmpeg_executable",
    "map_transition_intent",
]
