"""Clip metadata extraction tool for VideoGuru (SPEC-005).

Uses FFmpeg (ffprobe) to inspect video files and extract core container and stream properties:
duration, frame rate, resolution, video/audio codecs, and generates a UUID-based clip identifier
(e.g., 'vid_8f3a9b21'), returning a structured ClipManifestEntry Pydantic model.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Optional

from schemas.media import ClipManifestEntry

logger = logging.getLogger(__name__)


def find_ffprobe_executable() -> str:
    """Locate the ffprobe executable on PATH, with refreshed Windows registry fallback.

    Returns:
        Absolute or resolved path string to the ffprobe executable.

    Raises:
        RuntimeError: If ffprobe cannot be located on the system.
    """
    # 1. Standard PATH lookup
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        return ffprobe_path

    # 2. Windows registry PATH refresh fallback
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
        ffprobe_path = shutil.which("ffprobe", path=refreshed_path)
        if ffprobe_path:
            return ffprobe_path

    raise RuntimeError(
        "ffprobe executable not found on system PATH. "
        "Please ensure FFmpeg and ffprobe are installed and accessible."
    )


def probe_video_file(file_path: Path | str) -> dict[str, Any]:
    """Execute ffprobe on a video file and return parsed JSON metadata.

    Args:
        file_path: Path to the target video file.

    Returns:
        Parsed JSON dictionary containing 'streams' and 'format' metadata.

    Raises:
        ValueError: If file_path is empty or not a regular file.
        FileNotFoundError: If the file does not exist on disk.
        RuntimeError: If ffprobe cannot be found, execution fails, or output cannot be parsed.
    """
    if isinstance(file_path, str):
        cleaned_path = file_path.strip()
        if not cleaned_path:
            raise ValueError("file_path must be a non-empty string.")
        target_path = Path(cleaned_path).resolve()
    else:
        target_path = file_path.resolve()

    if not target_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {target_path}")

    if not target_path.is_file():
        raise ValueError(f"Path is not a regular file: {target_path}")

    ffprobe_bin = find_ffprobe_executable()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(target_path),
    ]

    logger.debug("Executing ffprobe command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as err:
        raise RuntimeError(f"Failed to execute ffprobe: {err}") from err

    if result.returncode != 0:
        err_msg = result.stderr.strip() or f"ffprobe exited with code {result.returncode}"
        raise RuntimeError(f"ffprobe failed on file '{target_path}': {err_msg}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as err:
        raise RuntimeError(
            f"Failed to parse ffprobe JSON output for file '{target_path}': {err}"
        ) from err

    return data


def _parse_rational_fps(val: Optional[str]) -> Optional[float]:
    """Parse a rational frame rate string (e.g., '30/1', '30000/1001', '29.97') into a float."""
    if not val or val.strip() in ("", "0/0", "0"):
        return None
    cleaned = val.strip()
    if "/" in cleaned:
        num_str, den_str = cleaned.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
            if den != 0:
                fps = round(num / den, 2)
                return fps if fps > 0 else None
        except (ValueError, ZeroDivisionError):
            return None
    else:
        try:
            fps = round(float(cleaned), 2)
            return fps if fps > 0 else None
        except ValueError:
            return None
    return None


def _parse_frame_rate(
    r_frame_rate: Optional[str],
    avg_frame_rate: Optional[str],
) -> float:
    """Extract and validate the video frame rate from r_frame_rate or avg_frame_rate."""
    fps = _parse_rational_fps(r_frame_rate)
    if fps is not None and fps > 0:
        return fps

    fps = _parse_rational_fps(avg_frame_rate)
    if fps is not None and fps > 0:
        return fps

    raise ValueError(
        f"Could not determine valid frame rate from stream (r_frame_rate='{r_frame_rate}', avg_frame_rate='{avg_frame_rate}')."
    )


def _parse_duration(
    video_stream: dict[str, Any],
    format_info: dict[str, Any],
) -> float:
    """Extract and validate video duration from stream or container format."""
    # 1. Try video stream duration
    stream_duration = video_stream.get("duration")
    if stream_duration and str(stream_duration).strip() not in ("", "N/A"):
        try:
            dur = float(stream_duration)
            if dur >= 0:
                return round(dur, 3)
        except (ValueError, TypeError):
            pass

    # 2. Try container format duration
    format_duration = format_info.get("duration")
    if format_duration and str(format_duration).strip() not in ("", "N/A"):
        try:
            dur = float(format_duration)
            if dur >= 0:
                return round(dur, 3)
        except (ValueError, TypeError):
            pass

    # 3. Check stream tags
    tags = video_stream.get("tags") or {}
    for tag_key in ("DURATION", "duration"):
        if tag_key in tags:
            tag_val = tags[tag_key]
            # Handle HH:MM:SS.mmmmmm or raw seconds
            if ":" in str(tag_val):
                try:
                    parts = str(tag_val).split(":")
                    if len(parts) == 3:
                        dur = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                        if dur >= 0:
                            return round(dur, 3)
                except (ValueError, TypeError):
                    pass
            else:
                try:
                    dur = float(tag_val)
                    if dur >= 0:
                        return round(dur, 3)
                except (ValueError, TypeError):
                    pass

    raise ValueError("Could not determine duration for media file.")


def extract_clip_metadata(
    file_path: str,
    clip_id: Optional[str] = None,
) -> ClipManifestEntry:
    """Extract deep container and stream metadata from a video file using FFmpeg (ffprobe).

    Extracts duration, frame rate, resolution, video/audio codecs, and associates
    the clip with a unique UUID-based identifier (e.g. 'vid_8f3a9b21').
    Designed as a custom Google ADK tool callable by agents during Phase I Media Ingestion.

    Args:
        file_path: Local filesystem path to the video file to inspect.
        clip_id: Optional custom clip identifier. If None, a UUID-based ID ('vid_<hex>') is generated.

    Returns:
        Structured ClipManifestEntry Pydantic model populated with clip properties.

    Raises:
        ValueError: If file_path is invalid, not a regular file, or missing a video stream.
        FileNotFoundError: If the target file does not exist on disk.
        RuntimeError: If ffprobe fails during execution.
    """
    probe_data = probe_video_file(file_path)

    p = Path(file_path).resolve()
    file_name = p.name
    file_size_bytes = p.stat().st_size

    streams = probe_data.get("streams", [])
    format_info = probe_data.get("format", {})

    # 1. Locate primary video stream
    video_stream: Optional[dict[str, Any]] = None
    audio_stream: Optional[dict[str, Any]] = None

    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise ValueError(f"No video stream found in media file: '{p}'")

    # 2. Extract dimensions & resolution
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video dimensions extracted from stream: width={width}, height={height}"
        )
    resolution = f"{width}x{height}"

    # 3. Extract codecs
    video_codec = str(video_stream.get("codec_name") or "unknown").lower()
    has_audio = audio_stream is not None
    audio_codec = (
        str(audio_stream.get("codec_name")).lower()
        if audio_stream and audio_stream.get("codec_name")
        else None
    )

    # 4. Extract frame rate & duration
    frame_rate = _parse_frame_rate(
        video_stream.get("r_frame_rate"),
        video_stream.get("avg_frame_rate"),
    )
    duration_seconds = _parse_duration(video_stream, format_info)

    # 5. Resolve clip ID
    if clip_id is not None and str(clip_id).strip():
        final_clip_id = str(clip_id).strip()
    else:
        final_clip_id = ClipManifestEntry.generate_clip_id(prefix="vid_")

    entry = ClipManifestEntry(
        clip_id=final_clip_id,
        absolute_path=str(p),
        file_name=file_name,
        duration_seconds=duration_seconds,
        frame_rate=frame_rate,
        resolution=resolution,
        width=width,
        height=height,
        video_codec=video_codec,
        audio_codec=audio_codec,
        has_audio=has_audio,
        file_size_bytes=file_size_bytes,
    )

    logger.info(
        "Extracted metadata for clip '%s' [%s]: %s, %.2ffps, %.2fs, video=%s, audio=%s",
        file_name,
        final_clip_id,
        resolution,
        frame_rate,
        duration_seconds,
        video_codec,
        audio_codec or "none",
    )

    return entry
