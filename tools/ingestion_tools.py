"""Ingestion tools for VideoGuru (SPEC-006).

Coordinates local directory scanning and clip metadata extraction to assemble
the comprehensive Clip Manifest (list of ClipManifestEntry) and store it in
session state under `session.state["clip_manifest"]`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from google.adk.tools import ToolContext

from config import settings
from schemas.media import ClipManifestEntry
from tools.clip_metadata import extract_clip_metadata
from tools.directory_scanner import scan_local_directory

logger = logging.getLogger(__name__)


def build_clip_manifest(
    directory_path: str | Path,
) -> list[ClipManifestEntry]:
    """Scan a local directory and extract deep metadata for every discovered video clip.

    Orchestrates `scan_local_directory` -> `extract_clip_metadata` (per file)
    to assemble a complete list of `ClipManifestEntry` models.

    Args:
        directory_path: Path to the directory containing raw video footage.

    Returns:
        List of ClipManifestEntry models sorted deterministically by absolute_path.

    Raises:
        ValueError: If directory_path is empty or whitespace.
        FileNotFoundError: If directory_path does not exist on disk.
        NotADirectoryError: If directory_path points to a file rather than a directory.
    """
    if isinstance(directory_path, str):
        cleaned = directory_path.strip()
        if not cleaned:
            raise ValueError("directory_path must be a non-empty string.")
        target_dir = Path(cleaned).resolve()
    else:
        target_dir = directory_path.resolve()

    if not target_dir.exists():
        raise FileNotFoundError(f"Media directory does not exist: {target_dir}")

    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    logger.info("Building clip manifest for directory: '%s'", target_dir)

    # 1. Discover all valid video file paths
    file_paths = scan_local_directory(str(target_dir))
    if not file_paths:
        logger.warning("No video files discovered in directory: '%s'", target_dir)
        return []

    # 2. Extract metadata per file and assemble manifest
    manifest: list[ClipManifestEntry] = []
    for file_path in file_paths:
        try:
            entry = extract_clip_metadata(file_path)
            manifest.append(entry)
            logger.debug(
                "Ingested clip '%s' [%s]: %s, %.2ffps, %.2fs",
                entry.file_name,
                entry.clip_id,
                entry.resolution,
                entry.frame_rate,
                entry.duration_seconds,
            )
        except Exception as exc:
            logger.warning("Failed to extract metadata for '%s', skipping: %s", file_path, exc)

    # Sort deterministically by absolute path
    manifest.sort(key=lambda x: x.absolute_path)
    total_duration = sum(e.duration_seconds for e in manifest)
    logger.info(
        "Successfully assembled Clip Manifest: %d clip(s), total duration: %.2fs (~%.1f min)",
        len(manifest),
        total_duration,
        total_duration / 60.0,
    )
    return manifest


def ingest_media_directory(
    directory_path: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> str:
    """Scan local directory, extract clip metadata, build Clip Manifest, and store in session state.

    This function serves as a custom Google ADK tool callable by agents in Phase I.
    Saves the list of serialized clip entries into `tool_context.state["clip_manifest"]`.

    Args:
        directory_path: Optional path to the video directory. If omitted, falls back to
            `tool_context.state.get("media_dir")` or the configured `settings.MEDIA_INPUT_DIR`.
        tool_context: Optional ADK ToolContext providing access to session state.

    Returns:
        Structured confirmation message summarizing the ingested clips and total duration.

    Raises:
        ValueError: If directory path cannot be resolved or is invalid.
        FileNotFoundError: If the resolved directory does not exist.
        NotADirectoryError: If the target path is not a directory.
    """
    # 1. Resolve directory path
    resolved_path_str: Optional[str] = None
    if directory_path and str(directory_path).strip():
        resolved_path_str = str(directory_path).strip()
    elif tool_context is not None and tool_context.state.get("media_dir"):
        resolved_path_str = str(tool_context.state["media_dir"]).strip()
    elif settings.MEDIA_INPUT_DIR:
        resolved_path_str = str(settings.MEDIA_INPUT_DIR).strip()

    if not resolved_path_str:
        raise ValueError(
            "No media directory specified. Please provide a valid directory_path."
        )

    target_dir = Path(resolved_path_str).resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Media directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    # 2. Build Clip Manifest
    manifest = build_clip_manifest(target_dir)

    # 3. Persist to session state if tool_context available
    serialized_manifest = [entry.model_dump() for entry in manifest]
    if tool_context is not None:
        tool_context.state["clip_manifest"] = serialized_manifest
        tool_context.state["media_dir"] = str(target_dir)
        logger.info(
            "Stored %d clip(s) in session state under 'clip_manifest'.",
            len(manifest),
        )

    # 4. Format summary report
    if not manifest:
        return (
            f"[IngestionAgent] Directory '{target_dir}' scanned. No supported video clips found."
        )

    total_duration = sum(e.duration_seconds for e in manifest)
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    formatted_duration = f"{minutes}m {seconds}s ({total_duration:.2f}s)"

    clip_lines: list[str] = []
    for idx, e in enumerate(manifest, 1):
        audio_info = f"audio={e.audio_codec}" if e.has_audio else "silent"
        clip_lines.append(
            f"  {idx}. [{e.clip_id}] {e.file_name} — {e.resolution}, {e.frame_rate:.2f}fps, "
            f"{e.duration_seconds:.2f}s, video={e.video_codec}, {audio_info}"
        )

    report = (
        f"[IngestionAgent] Successfully ingested {len(manifest)} clip(s) from '{target_dir}'.\n"
        f"Total Duration: {formatted_duration}\n"
        f"Clip Manifest:\n" + "\n".join(clip_lines) + "\n"
        f"Session state updated: 'clip_manifest' locked in with {len(manifest)} entries."
    )
    return report


def get_clip_manifest_from_state(state: dict[str, Any]) -> list[ClipManifestEntry]:
    """Retrieve and reconstruct ClipManifestEntry instances from session state.

    Args:
        state: ADK session state dictionary.

    Returns:
        List of ClipManifestEntry models, or empty list if 'clip_manifest' is absent.
    """
    raw_manifest = state.get("clip_manifest")
    if not raw_manifest or not isinstance(raw_manifest, list):
        return []

    entries: list[ClipManifestEntry] = []
    for item in raw_manifest:
        if isinstance(item, ClipManifestEntry):
            entries.append(item)
        elif isinstance(item, dict):
            try:
                entries.append(ClipManifestEntry.model_validate(item))
            except Exception as exc:
                logger.warning("Failed to validate ClipManifestEntry from state dict: %s", exc)
    return entries
