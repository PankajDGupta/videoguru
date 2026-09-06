"""Media and filesystem scanner schemas for VideoGuru."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class ScannedVideoFile(BaseModel):
    """Structured representation of a scanned video file on the local filesystem."""

    path: str = Field(
        ...,
        description="Absolute resolved filesystem path to the video file.",
        min_length=1,
    )
    file_name: str = Field(
        ...,
        description="Filename including extension (e.g. 'vlog_01.mp4').",
        min_length=1,
    )
    file_size_bytes: int = Field(
        ...,
        description="File size on disk in bytes.",
        ge=0,
    )
    last_modified: float = Field(
        ...,
        description="Last modified timestamp in seconds since epoch.",
    )
    last_modified_iso: str = Field(
        ...,
        description="Last modified timestamp formatted as ISO 8601 string in UTC.",
    )
    extension: str = Field(
        ...,
        description="Normalized lowercase file extension with leading dot (e.g. '.mp4').",
    )

    @field_validator("extension")
    @classmethod
    def validate_extension(cls, v: str) -> str:
        trimmed = v.strip().lower()
        if not trimmed.startswith("."):
            trimmed = f".{trimmed}"
        return trimmed

    @classmethod
    def from_path(cls, file_path: Path | str) -> ScannedVideoFile:
        """Construct a ScannedVideoFile instance by inspecting a filesystem path."""
        p = Path(file_path).resolve()
        stat = p.stat()
        mtime = stat.st_mtime
        mtime_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        return cls(
            path=str(p),
            file_name=p.name,
            file_size_bytes=stat.st_size,
            last_modified=mtime,
            last_modified_iso=mtime_iso,
            extension=p.suffix.lower(),
        )
