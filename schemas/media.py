"""Media and filesystem scanner schemas for VideoGuru."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


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


class ClipManifestEntry(BaseModel):
    """Structured representation of extracted media clip metadata for the Clip Manifest (SPEC-005)."""

    clip_id: str = Field(
        ...,
        description="Unique clip identifier (UUID-based e.g. 'vid_8f3a9b21').",
        min_length=1,
    )
    absolute_path: str = Field(
        ...,
        description="Absolute resolved filesystem path to the video file.",
        min_length=1,
    )
    file_name: str = Field(
        ...,
        description="Filename including extension (e.g. 'clip_01.mp4').",
        min_length=1,
    )
    duration_seconds: float = Field(
        ...,
        description="Duration of the video clip in seconds.",
        ge=0.0,
    )
    frame_rate: float = Field(
        ...,
        description="Normalized video frame rate in frames per second (fps).",
        gt=0.0,
    )
    resolution: str = Field(
        ...,
        description="Video resolution formatted as WxH (e.g. '1920x1080').",
        min_length=3,
    )
    width: int = Field(
        ...,
        description="Video frame width in pixels.",
        gt=0,
    )
    height: int = Field(
        ...,
        description="Video frame height in pixels.",
        gt=0,
    )
    video_codec: str = Field(
        ...,
        description="Video stream codec name (e.g. 'h264', 'hevc', 'vp9').",
        min_length=1,
    )
    audio_codec: Optional[str] = Field(
        default=None,
        description="Audio stream codec name (e.g. 'aac', 'mp3'), or None if video has no audio.",
    )
    has_audio: bool = Field(
        default=False,
        description="Whether the clip contains at least one audio stream.",
    )
    file_size_bytes: int = Field(
        default=0,
        description="File size on disk in bytes.",
        ge=0,
    )

    @model_validator(mode="before")
    @classmethod
    def handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "file_path" in data and "absolute_path" not in data:
                data["absolute_path"] = data["file_path"]
            if "codec" in data and "video_codec" not in data:
                data["video_codec"] = data["codec"]
        return data

    @field_validator("resolution")
    @classmethod
    def validate_resolution_format(cls, v: str) -> str:
        trimmed = v.strip()
        if "x" not in trimmed:
            raise ValueError(f"Resolution must be formatted as '<width>x<height>', got: '{v}'")
        parts = trimmed.split("x")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError(f"Resolution dimensions must be positive integers, got: '{v}'")
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"Resolution dimensions must be strictly positive, got: '{v}'")
        return trimmed

    @property
    def file_path(self) -> str:
        """Alias for absolute_path."""
        return self.absolute_path

    @property
    def codec(self) -> str:
        """Alias for video_codec."""
        return self.video_codec

    @classmethod
    def generate_clip_id(cls, prefix: str = "vid_") -> str:
        """Generate a standard UUID-based clip identifier (e.g. 'vid_8f3a9b21')."""
        return f"{prefix}{uuid.uuid4().hex[:8]}"

