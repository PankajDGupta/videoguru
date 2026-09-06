"""Whisper Captioning Tool for VideoGuru (SPEC-019).

Provides transcription using OpenAI Whisper, SubRip (.srt) subtitle formatting and generation,
FFmpeg subtitle burn-in rendering, and the ADK tool `generate_captions` for the Enhancement &
Rendering Agent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from config import settings
from rendering.ffmpeg_builder import (
    build_caption_burn_command,
    execute_ffmpeg_command,
)

logger = logging.getLogger(__name__)


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SubRip (SRT) timecode format: HH:MM:SS,mmm.

    Example:
        format_timestamp_srt(83.456) -> "00:01:23,456"
        format_timestamp_srt(0.0) -> "00:00:00,000"
        format_timestamp_srt(3661.123) -> "01:01:01,123"

    Args:
        seconds: Time offset in seconds (non-negative float). Negative values are clamped to 0.0.

    Returns:
        Formatted SRT timecode string formatted as HH:MM:SS,mmm.
    """
    if seconds < 0.0:
        seconds = 0.0

    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3_600_000
    rem = total_milliseconds % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1_000
    millis = rem % 1_000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt_file(
    segments: Sequence[dict[str, Any]],
    output_path: Union[str, Path],
) -> Path:
    """Format subtitle segments into a standard SubRip (.srt) file.

    Each segment is expected to have 'start', 'end', and 'text' keys. Empty or whitespace-only
    segments are skipped.

    Args:
        segments: Sequence of segment dictionaries containing 'start', 'end', and 'text'.
        output_path: Filesystem destination path for the .srt file.

    Returns:
        Resolved Path object pointing to the written .srt file.

    Raises:
        ValueError: If output_path is empty.
    """
    cleaned_path = str(output_path).strip()
    if not cleaned_path:
        raise ValueError("output_path cannot be empty.")

    out_path = Path(cleaned_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    block_index = 1

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        try:
            start_sec = float(segment.get("start", 0.0))
        except (TypeError, ValueError):
            start_sec = 0.0

        try:
            end_sec = float(segment.get("end", start_sec))
        except (TypeError, ValueError):
            end_sec = start_sec

        if end_sec < start_sec:
            end_sec = start_sec

        start_tc = format_timestamp_srt(start_sec)
        end_tc = format_timestamp_srt(end_sec)

        block = f"{block_index}\n{start_tc} --> {end_tc}\n{text}\n"
        blocks.append(block)
        block_index += 1

    content = "\n".join(blocks)
    if blocks and not content.endswith("\n"):
        content += "\n"

    out_path.write_text(content, encoding="utf-8")
    logger.info("Wrote %d subtitle blocks to %s", len(blocks), out_path)
    return out_path


def _get_mock_transcription(media_path: Union[str, Path]) -> dict[str, Any]:
    """Generate a deterministic mock transcription result for testing or offline mode."""
    stem = Path(media_path).stem
    return {
        "text": f"Welcome to VideoGuru automated video production pipeline for {stem}.",
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": 3.0,
                "text": f"Welcome to VideoGuru automated video production pipeline for {stem}.",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.2,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.01,
            },
            {
                "id": 1,
                "seek": 0,
                "start": 3.5,
                "end": 6.5,
                "text": "Creating broadcast quality YouTube videos with AI captions.",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": -0.2,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.01,
            },
        ],
        "language": "en",
    }


def transcribe_audio_whisper(
    media_path: Union[str, Path],
    model_name: Optional[str] = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Load OpenAI Whisper model and transcribe audio from a media file.

    Features automatic language detection and supports offline/mock mode for testing
    or when GPU/Whisper execution is unavailable.

    Args:
        media_path: Path to the target video or audio file.
        model_name: Optional Whisper model name (defaults to settings.WHISPER_MODEL or 'medium').
        offline: If True, bypasses Whisper model execution and returns mock transcription data.

    Returns:
        Dictionary containing transcribed 'text', 'segments' list, and detected 'language'.

    Raises:
        ValueError: If media_path is empty.
        FileNotFoundError: If media_path does not exist on disk and not in offline mode.
        RuntimeError: If Whisper transcription fails during execution.
    """
    cleaned_path = str(media_path).strip()
    if not cleaned_path:
        raise ValueError("media_path cannot be empty.")

    target_media = Path(cleaned_path).resolve()

    resolved_model = model_name or settings.WHISPER_MODEL or "medium"
    is_offline = (
        offline
        or os.getenv("VIDEOGURU_OFFLINE", "").lower() in ("1", "true", "yes")
        or os.getenv("MOCK_WHISPER", "").lower() in ("1", "true", "yes")
    )

    if is_offline:
        logger.info(
            "Transcribing '%s' in offline/mock mode (model=%s)",
            target_media.name,
            resolved_model,
        )
        return _get_mock_transcription(target_media)

    if not target_media.exists():
        raise FileNotFoundError(f"Media file does not exist: {target_media}")

    try:
        import whisper

        logger.info("Loading Whisper model '%s' for transcription...", resolved_model)
        model = whisper.load_model(resolved_model)
        logger.info("Transcribing media file: %s", target_media)
        result = model.transcribe(str(target_media))
        return result
    except ImportError as exc:
        logger.warning("openai-whisper package not found (%s). Falling back to mock transcription.", exc)
        return _get_mock_transcription(target_media)
    except Exception as exc:
        logger.error("Whisper transcription failed on '%s': %s", target_media, exc)
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc


def burn_subtitles_to_video(
    video_path: Union[str, Path],
    srt_path: Union[str, Path],
    output_path: Union[str, Path],
    font_size: int = 16,
) -> str:
    """Burn subtitles from an SRT file into video frames using FFmpeg's subtitles filter.

    Args:
        video_path: Path to the source video file.
        srt_path: Path to the SubRip Subtitle (.srt) file.
        output_path: Destination path for the captioned output video.
        font_size: Subtitle font size in points (default: 16).

    Returns:
        Absolute string path to the generated captioned video file.

    Raises:
        ValueError: If paths are empty or font_size <= 0.
        FileNotFoundError: If source video or SRT file does not exist.
        RuntimeError: If FFmpeg execution fails.
    """
    str_video = str(video_path).strip()
    str_srt = str(srt_path).strip()
    str_out = str(output_path).strip()

    if not str_video:
        raise ValueError("video_path cannot be empty.")
    if not str_srt:
        raise ValueError("srt_path cannot be empty.")
    if not str_out:
        raise ValueError("output_path cannot be empty.")

    vid_file = Path(str_video).resolve()
    srt_file = Path(str_srt).resolve()
    out_file = Path(str_out).resolve()

    if not vid_file.exists():
        raise FileNotFoundError(f"Source video not found: {vid_file}")
    if not srt_file.exists():
        raise FileNotFoundError(f"Subtitle file not found: {srt_file}")

    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_caption_burn_command(
        video_path=vid_file,
        srt_path=srt_file,
        output_path=out_file,
        font_size=font_size,
    )

    logger.info("Burning subtitles '%s' into '%s' -> '%s'", srt_file.name, vid_file.name, out_file.name)
    execute_ffmpeg_command(cmd)

    return str(out_file)


def generate_captions(
    video_path: Union[str, Path],
    output_srt_path: Optional[Union[str, Path]] = None,
    output_video_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    tool_context: Optional[Any] = None,
    offline: bool = False,
    font_size: int = 16,
) -> dict[str, str]:
    """ADK Tool: Generate Whisper captions for a video and burn them in with FFmpeg.

    Orchestrates transcription, .srt creation, subtitle burn-in rendering, and updates
    session state (`tool_context.state["srt_path"]` and
    `tool_context.state["captioned_video_path"]`).

    Args:
        video_path: Source video file path.
        output_srt_path: Optional destination path for the .srt file (defaults to STAGING_DIR/<stem>.srt).
        output_video_path: Optional destination path for captioned video (defaults to OUTPUT_DIR/<stem>_captioned.mp4).
        model_name: Optional Whisper model size (defaults to settings.WHISPER_MODEL or 'medium').
        tool_context: Optional ADK ToolContext for session state persistence.
        offline: If True, runs in offline/mock transcription mode.
        font_size: Subtitle font size in points (default: 16).

    Returns:
        Dictionary containing 'srt_path' and 'captioned_video_path'.

    Raises:
        ValueError: If video_path is empty.
        FileNotFoundError: If video_path does not exist.
        RuntimeError: If transcription or FFmpeg rendering fails.
    """
    str_video = str(video_path).strip()
    if not str_video:
        raise ValueError("video_path cannot be empty.")

    vid_path = Path(str_video).resolve()
    if not offline and not vid_path.exists():
        raise FileNotFoundError(f"Video file not found: {vid_path}")

    # 1. Resolve output SRT path
    if output_srt_path is not None and str(output_srt_path).strip():
        resolved_srt_path = Path(output_srt_path).resolve()
    else:
        resolved_srt_path = settings.STAGING_DIR / f"{vid_path.stem}.srt"

    # 2. Resolve output Video path
    if output_video_path is not None and str(output_video_path).strip():
        resolved_video_path = Path(output_video_path).resolve()
    else:
        resolved_video_path = settings.OUTPUT_DIR / f"{vid_path.stem}_captioned.mp4"

    resolved_model = model_name or settings.WHISPER_MODEL or "medium"

    try:
        # 3. Transcribe audio
        transcription = transcribe_audio_whisper(
            media_path=vid_path,
            model_name=resolved_model,
            offline=offline,
        )

        # 4. Format and write .srt file
        segments = transcription.get("segments", [])
        write_srt_file(segments=segments, output_path=resolved_srt_path)

        # 5. Burn subtitles into video
        burn_subtitles_to_video(
            video_path=vid_path,
            srt_path=resolved_srt_path,
            output_path=resolved_video_path,
            font_size=font_size,
        )
    except Exception as exc:
        logger.warning(
            "generate_captions failed (%s). Degrading gracefully: returning uncaptioned video.",
            exc,
        )
        if tool_context is not None and hasattr(tool_context, "state"):
            from services.resilience import record_session_warning
            record_session_warning(
                state=tool_context.state,
                warning_message=f"Whisper subtitle generation failed ({exc}); continuing without captions.",
                category="captioning",
            )
            if isinstance(tool_context.state, dict):
                tool_context.state["captioned_video_path"] = str(vid_path)
                tool_context.state["srt_path"] = None
        return {
            "srt_path": None,
            "captioned_video_path": str(vid_path),
        }

    # 6. Update session state if ToolContext is provided
    if tool_context is not None and hasattr(tool_context, "state") and isinstance(tool_context.state, dict):
        tool_context.state["srt_path"] = str(resolved_srt_path)
        tool_context.state["captioned_video_path"] = str(resolved_video_path)
        logger.info(
            "Updated session state: srt_path='%s', captioned_video_path='%s'",
            resolved_srt_path,
            resolved_video_path,
        )

    return {
        "srt_path": str(resolved_srt_path),
        "captioned_video_path": str(resolved_video_path),
    }
