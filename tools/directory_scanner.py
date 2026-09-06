"""Local directory scanning tool for VideoGuru (SPEC-004).

Recursively discovers video files (.mp4, .mov, .avi, .mkv), extracts basic filesystem
metadata (file name, file size, last modified timestamp), and returns a list of resolved
file path strings for downstream processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from schemas.media import ScannedVideoFile

logger = logging.getLogger(__name__)

# File extensions recognized as valid video footage in Phase I media ingestion
SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv"})


def extract_file_basic_metadata(file_path: Path | str) -> ScannedVideoFile:
    """Extract basic filesystem metadata for a given file.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        ScannedVideoFile Pydantic model populated with path, name, size, and timestamp.

    Raises:
        FileNotFoundError: If the file does not exist on disk.
        ValueError: If the path is not a regular file.
    """
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Video file does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a regular file: {p}")

    return ScannedVideoFile.from_path(p)


def scan_local_directory_with_metadata(
    directory_path: str | Path,
    supported_extensions: Optional[set[str] | frozenset[str]] = None,
) -> list[ScannedVideoFile]:
    """Recursively scan a local directory for video files and extract their basic metadata.

    Args:
        directory_path: Directory path to scan recursively.
        supported_extensions: Optional set of lowercase extensions with leading dot.
            Defaults to SUPPORTED_VIDEO_EXTENSIONS.

    Returns:
        List of ScannedVideoFile models sorted deterministically by path.

    Raises:
        ValueError: If directory_path is empty or whitespace.
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the target path exists but is not a directory.
    """
    if isinstance(directory_path, str):
        cleaned_path = directory_path.strip()
        if not cleaned_path:
            raise ValueError("directory_path must be a non-empty string.")
        target_dir = Path(cleaned_path).resolve()
    else:
        target_dir = directory_path.resolve()

    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    if not target_dir.is_dir():
        raise NotADirectoryError(f"Target path is not a directory: {target_dir}")

    exts = supported_extensions or SUPPORTED_VIDEO_EXTENSIONS
    normalized_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    scanned_files: list[ScannedVideoFile] = []
    logger.info("Scanning directory '%s' for video files with extensions: %s", target_dir, sorted(normalized_exts))

    for item in target_dir.rglob("*"):
        # Skip hidden files or files located within hidden directories (e.g. .git, .cache)
        if any(part.startswith(".") for part in item.relative_to(target_dir).parts):
            continue

        if item.is_file() and item.suffix.lower() in normalized_exts:
            try:
                metadata = extract_file_basic_metadata(item)
                scanned_files.append(metadata)
                logger.debug(
                    "Discovered video file: %s (size: %d bytes, modified: %s)",
                    metadata.file_name,
                    metadata.file_size_bytes,
                    metadata.last_modified_iso,
                )
            except (OSError, PermissionError) as err:
                logger.warning("Failed to read metadata for file '%s': %s", item, err)

    # Sort deterministically by resolved path
    scanned_files.sort(key=lambda x: x.path)
    logger.info("Found %d video file(s) in directory '%s'", len(scanned_files), target_dir)
    return scanned_files


def scan_local_directory(directory_path: str) -> list[str]:
    """Recursively scan a local directory for video clips (.mp4, .mov, .avi, .mkv).

    Extracts basic filesystem metadata (file name, size, timestamp) and returns a list
    of resolved absolute file path strings for downstream processing.

    This function is designed as a Google ADK tool callable by agents.

    Args:
        directory_path: Local directory path containing raw video clips downloaded from Google Photos.

    Returns:
        List of absolute file path strings of discovered video files.

    Raises:
        ValueError: If directory_path is empty or whitespace.
        FileNotFoundError: If the specified directory does not exist.
        NotADirectoryError: If the specified path is a file rather than a directory.
    """
    scanned_models = scan_local_directory_with_metadata(directory_path)
    return [model.path for model in scanned_models]
