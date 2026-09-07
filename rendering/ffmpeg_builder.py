"""FFmpeg Command Builder for VideoGuru (SPEC-016).

Provides deterministic construction of FFmpeg commands for trimming, transition crossfading,
audio crossfading, sidechain compression audio ducking, and subtitle burning, along with
sandboxed command execution and path normalization.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Optional, Sequence, Union

from schemas.edl import TransitionIntent

logger = logging.getLogger(__name__)

# Mapping from TransitionIntent or shorthand strings to valid FFmpeg xfade transition names
TRANSITION_MAP: dict[str, str] = {
    "fade": "fade",
    "wipe": "wipeleft",
    "wipeleft": "wipeleft",
    "wiperight": "wiperight",
    "wipeup": "wipeup",
    "wipedown": "wipedown",
    "slide": "slideleft",
    "slideleft": "slideleft",
    "slideright": "slideright",
    "slideup": "slideup",
    "slidedown": "slidedown",
    "dissolve": "dissolve",
    "cut": "fade",  # Fallback for cut transition in xfade chains
    "fadeblack": "fadeblack",
    "fadewhite": "fadewhite",
    "pixelize": "pixelize",
    "circlecrop": "circlecrop",
    "rectcrop": "rectcrop",
    "circleopen": "circleopen",
    "circleclose": "circleclose",
    "vertopen": "vertopen",
    "vertclose": "vertclose",
    "horzopen": "horzopen",
    "horzclose": "horzclose",
    "smoothleft": "smoothleft",
    "smoothright": "smoothright",
    "smoothup": "smoothup",
    "smoothdown": "smoothdown",
}

VALID_XFADE_TRANSITIONS: frozenset[str] = frozenset([
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown", "slideleft",
    "slideright", "slideup", "slidedown", "dissolve", "pixelize",
    "circlecrop", "rectcrop", "circleopen", "circleclose", "vertopen",
    "vertclose", "horzopen", "horzclose", "fadeblack", "fadewhite",
    "smoothleft", "smoothright", "smoothup", "smoothdown", "radial",
    "hblur", "wipetl", "wipetr", "wipebl", "wipebr", "squeezeh", "squeezev",
])


def find_ffmpeg_executable() -> str:
    """Locate the ffmpeg executable on PATH, with Windows registry fallback.

    Returns:
        Absolute or resolved path string to the ffmpeg executable.

    Raises:
        RuntimeError: If ffmpeg cannot be located on the system PATH.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    if os.name == "nt":
        paths = os.environ.get("PATH", "").split(os.pathsep)
        try:
            import winreg

            for root in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
                subkey = (
                    r"Environment"
                    if root == winreg.HKEY_CURRENT_USER
                    else r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                )
                try:
                    with winreg.OpenKey(root, subkey) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        for p in val.split(";"):
                            if p and p not in paths:
                                paths.append(p)
                except Exception:
                    pass
        except ImportError:
            pass

        refreshed_path = os.pathsep.join(paths)
        ffmpeg_path = shutil.which("ffmpeg", path=refreshed_path)
        if ffmpeg_path:
            return ffmpeg_path

    raise RuntimeError(
        "FFmpeg executable 'ffmpeg' not found in system PATH. "
        "Please ensure FFmpeg is installed and accessible."
    )


def map_transition_intent(transition: Union[str, TransitionIntent]) -> str:
    """Map a TransitionIntent or string to a canonical FFmpeg xfade transition name.

    Args:
        transition: TransitionIntent enum or string identifier.

    Returns:
        Canonical FFmpeg xfade transition name.

    Raises:
        ValueError: If the transition string is empty.
    """
    if isinstance(transition, TransitionIntent):
        raw_val = transition.value
    elif isinstance(transition, str):
        raw_val = transition.strip()
        if not raw_val:
            raise ValueError("Transition string cannot be empty.")
    else:
        raise TypeError(f"Expected str or TransitionIntent, got {type(transition).__name__}.")

    key = raw_val.lower()
    if key in TRANSITION_MAP:
        return TRANSITION_MAP[key]
    if key in VALID_XFADE_TRANSITIONS:
        return key

    logger.warning("Unknown transition '%s', defaulting to 'fade'", raw_val)
    return "fade"


def escape_subtitles_path(path: Union[str, Path]) -> str:
    """Escape a filesystem path for FFmpeg's subtitles filter argument on Windows and POSIX.

    Windows paths typically contain backslashes and drive letter colons (e.g. C:\\dir\\file.srt).
    FFmpeg's filtergraph parser treats ':' as an option separator and '\\' as an escape character.
    This function converts backslashes to forward slashes, escapes colons (C\\:), and escapes single quotes.

    Args:
        path: Path to the subtitle (.srt) file.

    Returns:
        Properly escaped path string safe for insertion into `subtitles='...'`.
    """
    cleaned = str(path).strip()
    if not cleaned:
        raise ValueError("Subtitle path cannot be empty.")

    p = cleaned.replace("\\", "/")
    p = p.replace(":", r"\:")
    p = p.replace("'", r"\'")
    return p


def build_atempo_filter(speed: float) -> str:
    """Build a chained atempo audio filter for speeds between 0.25 and 16.0.

    FFmpeg's `atempo` filter accepts values strictly between 0.5 and 2.0.
    For speeds outside this range, multiple atempo filters must be chained in series
    (e.g., 4.0x speed requires 'atempo=2.0000,atempo=2.0000').

    Args:
        speed: Playback speed multiplier (> 0.0).

    Returns:
        Comma-separated string of atempo filters, or empty string if speed is 1.0.

    Raises:
        ValueError: If speed <= 0.0.
    """
    if speed <= 0.0:
        raise ValueError(f"Speed multiplier must be positive, got {speed}")

    if abs(speed - 1.0) < 1e-4:
        return ""

    filters: list[str] = []
    curr = float(speed)

    # For speed > 2.0, chain atempo=2.0 factors
    while curr > 2.0:
        filters.append("atempo=2.0000")
        curr /= 2.0

    # For speed < 0.5, chain atempo=0.5 factors
    while curr < 0.5:
        filters.append("atempo=0.5000")
        curr /= 0.5

    filters.append(f"atempo={curr:.4f}")
    return ",".join(filters)


def build_trim_command(
    clip_path: Union[str, Path],
    start: float,
    end: float,
    output_path: Union[str, Path],
    target_resolution: str = "1920x1080",
    target_fps: float = 30.0,
    playback_speed: float = 1.0,
    has_audio: bool = True,
) -> list[str]:
    """Build an FFmpeg CLI command to trim, normalize, and speed-adjust a single video clip.

    Normalizes video to target resolution (with aspect-ratio preservation and padding),
    target frame rate, and applies presentation timestamp (setpts) and audio tempo (atempo)
    speed scaling for seamless downstream concatenation.

    Args:
        clip_path: Path to the source video clip.
        start: Segment start time in seconds (>= 0.0).
        end: Segment end time in seconds (> start).
        output_path: Destination path for the trimmed video clip.
        target_resolution: Normalized resolution in 'WIDTHxHEIGHT' format (default: '1920x1080').
        target_fps: Normalized frame rate (default: 30.0).
        playback_speed: Playback speed multiplier (default: 1.0; e.g. 2.0 for 2x fast-forward).
        has_audio: Whether the source clip has an audio track to preserve and speed-scale.

    Returns:
        List of FFmpeg command arguments.

    Raises:
        ValueError: If start < 0, end <= start, target_fps <= 0, playback_speed <= 0,
                    or invalid resolution format.
    """
    str_clip = str(clip_path).strip()
    str_out = str(output_path).strip()
    if not str_clip:
        raise ValueError("clip_path cannot be empty.")
    if not str_out:
        raise ValueError("output_path cannot be empty.")

    if start < 0.0:
        raise ValueError(f"start trim must be non-negative, got {start}")
    if end <= start:
        raise ValueError(f"end trim must be strictly greater than start trim ({end} <= {start})")
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if playback_speed <= 0.0:
        raise ValueError(f"playback_speed must be positive, got {playback_speed}")

    res_match = re.match(r"^(\d+)x(\d+)$", target_resolution.strip().lower())
    if not res_match:
        raise ValueError(
            f"Invalid target_resolution format: '{target_resolution}'. "
            "Expected 'WIDTHxHEIGHT', e.g. '1920x1080'."
        )
    width, height = int(res_match.group(1)), int(res_match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution dimensions must be positive, got {width}x{height}")

    ffmpeg_bin = find_ffmpeg_executable()

    filter_parts = []
    if abs(playback_speed - 1.0) > 1e-4:
        pts_factor = 1.0 / playback_speed
        filter_parts.append(f"setpts={pts_factor:.6f}*PTS")

    filter_parts.extend([
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
        f"fps={target_fps}",
    ])
    vf_filter = ",".join(filter_parts)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str_clip,
        "-vf",
        vf_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
    ]

    if has_audio:
        if abs(playback_speed - 1.0) > 1e-4:
            af_filter = build_atempo_filter(playback_speed)
            if af_filter:
                cmd.extend(["-af", af_filter])
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")

    cmd.append(str_out)
    return cmd


def calculate_xfade_offsets(
    durations: Sequence[float],
    transition_duration: float = 1.0,
) -> list[float]:
    """Calculate the cumulative transition offset timestamps for an FFmpeg xfade chain.

    For durations [D0, D1, D2] and transition duration T:
      offset_1 = D0 - T
      blended_duration_1 = D0 + D1 - T
      offset_2 = blended_duration_1 - T = (D0 + D1) - 2*T
      ...

    Args:
        durations: Durations of each sequential clip in seconds.
        transition_duration: Duration of the crossfade transition in seconds (default: 1.0).

    Returns:
        List of offset timestamps in seconds corresponding to each transition.

    Raises:
        ValueError: If durations is empty, transition_duration <= 0, or any duration <= transition_duration.
    """
    if not durations:
        raise ValueError("durations list cannot be empty.")
    if transition_duration <= 0.0:
        raise ValueError(f"transition_duration must be positive, got {transition_duration}")

    for idx, d in enumerate(durations):
        if d <= 0.0:
            raise ValueError(f"Clip duration at index {idx} must be positive, got {d}")
        if d <= transition_duration:
            raise ValueError(
                f"Clip duration at index {idx} ({d:.3f}s) must be strictly greater than "
                f"transition duration ({transition_duration:.3f}s)"
            )

    if len(durations) < 2:
        return []

    offsets: list[float] = []
    running_duration = float(durations[0])
    for i in range(1, len(durations)):
        offset = running_duration - transition_duration
        offsets.append(round(offset, 4))
        running_duration += float(durations[i]) - transition_duration

    return offsets


def build_audio_crossfade(
    clips: Sequence[Union[str, Path]],
    durations: Sequence[float],
    transition_duration: float = 1.0,
    curve1: str = "tri",
    curve2: str = "tri",
) -> str:
    """Build an FFmpeg audio crossfade filter chain using the acrossfade filter.

    Chains multiple audio streams sequentially with crossfading:
      [0:a][1:a]acrossfade=d=1.0:c1=tri:c2=tri[a1];[a1][2:a]acrossfade=d=1.0:c1=tri:c2=tri[a2]...

    Args:
        clips: Sequence of clip paths.
        durations: Durations of each clip in seconds.
        transition_duration: Crossfade overlap duration in seconds (default: 1.0).
        curve1: Fade-out curve for first stream (default: 'tri').
        curve2: Fade-in curve for second stream (default: 'tri').

    Returns:
        FFmpeg filter_complex substring for audio crossfading.

    Raises:
        ValueError: If clips is empty, len(clips) < 2, len(clips) != len(durations),
                    or any duration <= transition_duration.
    """
    if not clips:
        raise ValueError("clips sequence cannot be empty.")
    if len(clips) < 2:
        raise ValueError("At least 2 clips are required for an audio crossfade chain.")
    if len(clips) != len(durations):
        raise ValueError(
            f"Number of clips ({len(clips)}) does not match number of durations ({len(durations)})."
        )
    if transition_duration <= 0.0:
        raise ValueError(f"transition_duration must be positive, got {transition_duration}")

    for idx, d in enumerate(durations):
        if d <= 0.0:
            raise ValueError(f"Clip duration at index {idx} must be positive, got {d}")
        if d <= transition_duration:
            raise ValueError(
                f"Clip duration at index {idx} ({d:.3f}s) must be strictly greater than "
                f"transition duration ({transition_duration:.3f}s)"
            )

    filter_steps: list[str] = []
    n = len(clips)
    for i in range(1, n):
        prev_label = "[0:a]" if i == 1 else f"[a{i-1}]"
        next_label = f"[{i}:a]"
        out_label = f"[a{i}]"
        filter_steps.append(
            f"{prev_label}{next_label}acrossfade=d={transition_duration}:c1={curve1}:c2={curve2}{out_label}"
        )

    return ";".join(filter_steps)


def build_xfade_chain(
    clips: Sequence[Union[str, Path]],
    transitions: Union[Sequence[Union[str, TransitionIntent]], str, TransitionIntent],
    durations: Sequence[float],
    output_path: Union[str, Path],
    transition_duration: float = 1.0,
) -> list[str]:
    """Build a complete FFmpeg command chaining video xfade and audio acrossfade filters.

    Args:
        clips: Ordered list of input clip paths.
        transitions: List of transition intents (length = len(clips) - 1), or a single intent broadcast to all.
        durations: List of durations corresponding to each clip.
        output_path: Path for rendered output video.
        transition_duration: Overlap duration in seconds for each transition (default: 1.0).

    Returns:
        List of FFmpeg command arguments.

    Raises:
        ValueError: If fewer than 2 clips provided, length mismatch, or invalid durations.
    """
    if not clips:
        raise ValueError("clips sequence cannot be empty.")
    if len(clips) < 2:
        raise ValueError("At least 2 clips are required to build an xfade chain.")
    if len(clips) != len(durations):
        raise ValueError(
            f"Number of clips ({len(clips)}) does not match number of durations ({len(durations)})."
        )

    str_out = str(output_path).strip()
    if not str_out:
        raise ValueError("output_path cannot be empty.")

    n = len(clips)
    needed_transitions = n - 1

    if isinstance(transitions, (str, TransitionIntent)):
        trans_list = [transitions] * needed_transitions
    else:
        trans_list = list(transitions)
        if len(trans_list) != needed_transitions:
            raise ValueError(
                f"Expected {needed_transitions} transitions for {n} clips, got {len(trans_list)}."
            )

    offsets = calculate_xfade_offsets(durations, transition_duration)

    video_filters: list[str] = []
    for i in range(1, n):
        trans_name = map_transition_intent(trans_list[i - 1])
        prev_v = "[0:v]" if i == 1 else f"[v{i-1}]"
        next_v = f"[{i}:v]"
        out_v = f"[v{i}]"
        offset = offsets[i - 1]
        video_filters.append(
            f"{prev_v}{next_v}xfade=transition={trans_name}:duration={transition_duration}:offset={offset}{out_v}"
        )

    audio_filter_str = build_audio_crossfade(clips, durations, transition_duration)
    combined_filter_complex = ";".join(video_filters) + ";" + audio_filter_str

    ffmpeg_bin = find_ffmpeg_executable()
    cmd = [ffmpeg_bin, "-y"]
    for clip in clips:
        cmd.extend(["-i", str(clip)])

    final_v_label = f"[v{n-1}]"
    final_a_label = f"[a{n-1}]"

    cmd.extend([
        "-filter_complex",
        combined_filter_complex,
        "-map",
        final_v_label,
        "-map",
        final_a_label,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str_out,
    ])
    return cmd


def build_ducking_command(
    video_path: Union[str, Path],
    music_path: Union[str, Path],
    output_path: Union[str, Path],
    threshold: float = 0.05,
    ratio: float = 4.0,
    attack: float = 20.0,
    release: float = 250.0,
    music_volume: float = 0.3,
) -> list[str]:
    """Build an FFmpeg command to duck background music under video speech using sidechaincompress.

    The video's audio track serves as the sidechain detector. When speech occurs,
    the background music is automatically compressed. The ducked music and original speech
    are mixed and synchronized with the video stream.

    Args:
        video_path: Path to the primary video file (speech audio on [0:a]).
        music_path: Path to the background music file ([1:a]).
        output_path: Path for the ducked output video.
        threshold: Sidechain compressor threshold level (default: 0.05, range (0.0, 1.0]).
        ratio: Compression ratio (default: 4.0, >= 1.0).
        attack: Compressor attack time in milliseconds (default: 20.0, > 0.0).
        release: Compressor release time in milliseconds (default: 250.0, > 0.0).
        music_volume: Pre-compression volume scale for background music (default: 0.3, >= 0.0).

    Returns:
        List of FFmpeg command arguments.

    Raises:
        ValueError: If paths are empty or parameters are out of valid range.
    """
    str_video = str(video_path).strip()
    str_music = str(music_path).strip()
    str_out = str(output_path).strip()
    if not str_video:
        raise ValueError("video_path cannot be empty.")
    if not str_music:
        raise ValueError("music_path cannot be empty.")
    if not str_out:
        raise ValueError("output_path cannot be empty.")

    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be in range (0.0, 1.0], got {threshold}")
    if ratio < 1.0:
        raise ValueError(f"ratio must be >= 1.0, got {ratio}")
    if attack <= 0.0:
        raise ValueError(f"attack must be positive, got {attack}")
    if release <= 0.0:
        raise ValueError(f"release must be positive, got {release}")
    if music_volume < 0.0:
        raise ValueError(f"music_volume must be non-negative, got {music_volume}")

    ffmpeg_bin = find_ffmpeg_executable()
    filter_complex = (
        f"[1:a]volume={music_volume}[music];"
        f"[music][0:a]sidechaincompress=threshold={threshold}:ratio={ratio}:"
        f"attack={attack}:release={release}[ducked];"
        f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )

    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str_video,
        "-i",
        str_music,
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str_out,
    ]


def build_caption_burn_command(
    video_path: Union[str, Path],
    srt_path: Union[str, Path],
    output_path: Union[str, Path],
    font_size: int = 16,
) -> list[str]:
    """Build an FFmpeg command to burn-in captions from an SRT file into video frames.

    Handles cross-platform path escaping including Windows drive letter colons and backslashes.

    Args:
        video_path: Source video file path.
        srt_path: SubRip Subtitle (.srt) file path.
        output_path: Destination path for the captioned video.
        font_size: Subtitle font size in points (default: 16).

    Returns:
        List of FFmpeg command arguments.

    Raises:
        ValueError: If paths are empty or font_size <= 0.
    """
    str_video = str(video_path).strip()
    str_out = str(output_path).strip()
    if not str_video:
        raise ValueError("video_path cannot be empty.")
    if not str_out:
        raise ValueError("output_path cannot be empty.")
    if font_size <= 0:
        raise ValueError(f"font_size must be a positive integer, got {font_size}")

    escaped_srt = escape_subtitles_path(srt_path)
    vf_arg = f"subtitles='{escaped_srt}':force_style='FontSize={font_size}'"

    ffmpeg_bin = find_ffmpeg_executable()
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str_video,
        "-vf",
        vf_arg,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "copy",
        str_out,
    ]


def execute_ffmpeg_command(
    cmd: Sequence[str],
    timeout: float = 300.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Execute an FFmpeg command via subprocess with argument validation and error handling.

    Args:
        cmd: Sequence of command-line arguments.
        timeout: Execution timeout in seconds (default: 300.0).
        check: If True, raises RuntimeError when FFmpeg exits with a non-zero returncode.

    Returns:
        subprocess.CompletedProcess containing stdout and stderr.

    Raises:
        ValueError: If cmd is empty, non-string arguments present, or timeout <= 0.
        TimeoutError: If execution exceeds the specified timeout.
        RuntimeError: If subprocess fails or returns non-zero when check=True.
    """
    if not cmd:
        raise ValueError("Command cannot be empty.")
    if not all(isinstance(arg, str) for arg in cmd):
        raise TypeError("All command arguments must be strings.")
    if timeout <= 0.0:
        raise ValueError(f"timeout must be positive, got {timeout}")

    bin_name = Path(cmd[0]).name.lower()
    if not (bin_name.startswith("ffmpeg") or bin_name.startswith("ffprobe")):
        raise ValueError(
            f"Permitted binary must be ffmpeg or ffprobe, got '{cmd[0]}'"
        )

    logger.debug("Executing FFmpeg command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("FFmpeg command timed out after %s seconds: %s", timeout, cmd)
        raise TimeoutError(f"FFmpeg command timed out after {timeout} seconds") from exc
    except OSError as exc:
        logger.error("FFmpeg command execution failed with OS error: %s", exc)
        raise RuntimeError(f"FFmpeg execution failed: {exc}") from exc

    if check and result.returncode != 0:
        error_msg = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"Process exited with return code {result.returncode}"
        )
        logger.error("FFmpeg command failed with code %d: %s", result.returncode, error_msg)
        raise RuntimeError(
            f"FFmpeg command execution failed (code {result.returncode}): {error_msg}"
        )

    return result
