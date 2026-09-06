"""VideoGuru Domain Exceptions Hierarchy (SPEC-029).

Defines structured domain exceptions for media ingestion, Gemini multimodal analysis,
FFmpeg rendering, audio ducking, Whisper transcription, and schema validation.
"""

from __future__ import annotations

from typing import Any, Optional


class VideoGuruError(Exception):
    """Base domain exception for all VideoGuru errors."""

    def __init__(
        self,
        message: str,
        category: str = "general",
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        """Initialize VideoGuruError.

        Args:
            message: Technical error message for logs/developers.
            category: High-level error classification category.
            details: Optional dictionary containing debugging metadata.
            user_message: User-friendly explanation safe to present in UI/CLI.
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
        self.user_message = user_message or message

    def to_dict(self) -> dict[str, Any]:
        """Convert error to a serializable dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "category": self.category,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
        }


class IngestionError(VideoGuruError):
    """Raised when local media scanning, file probing, or manifest generation fails."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        merged_details = details or {}
        if file_path:
            merged_details["file_path"] = file_path
        super().__init__(
            message=message,
            category="ingestion",
            details=merged_details,
            user_message=user_message or f"Failed to ingest media files: {message}",
        )


class GeminiApiError(VideoGuruError):
    """Raised when Gemini Multimodal API calls fail (rate limits, timeouts, quota)."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        merged_details = details or {}
        if status_code is not None:
            merged_details["status_code"] = status_code
        super().__init__(
            message=message,
            category="gemini_api",
            details=merged_details,
            user_message=user_message or "AI video analysis is currently unavailable. Falling back to heuristic curation.",
        )


class SchemaValidationError(VideoGuruError):
    """Raised when model-generated EDL JSON violates Pydantic schema contracts."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[list[str]] = None,
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        merged_details = details or {}
        if validation_errors:
            merged_details["validation_errors"] = validation_errors
        super().__init__(
            message=message,
            category="schema_validation",
            details=merged_details,
            user_message=user_message or f"Edit Decision List format validation error: {message}",
        )


class RenderError(VideoGuruError):
    """Raised when FFmpeg processing or timeline rendering fails."""

    def __init__(
        self,
        message: str,
        command: Optional[list[str]] = None,
        stderr: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        merged_details = details or {}
        if command:
            merged_details["command"] = command
        if stderr:
            merged_details["stderr_tail"] = stderr[-1000:]
        super().__init__(
            message=message,
            category="rendering",
            details=merged_details,
            user_message=user_message or f"Video rendering encountered an issue: {message}",
        )


class AudioDuckingError(RenderError):
    """Raised when FFmpeg sidechain compression or audio mixing fails."""

    def __init__(
        self,
        message: str,
        music_path: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        merged_details = details or {}
        if music_path:
            merged_details["music_path"] = music_path
        super().__init__(
            message=message,
            details=merged_details,
            user_message=user_message or "Audio ducking could not be applied. Continuing with original audio.",
        )
        self.category = "audio_ducking"


class WhisperError(RenderError):
    """Raised when Whisper speech-to-text or subtitle burn-in fails."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ) -> None:
        super().__init__(
            message=message,
            details=details,
            user_message=user_message or "Speech transcription and subtitle burn-in failed. Video rendered without captions.",
        )
        self.category = "captioning"
