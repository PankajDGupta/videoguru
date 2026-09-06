"""Theme and intent schemas for VideoGuru."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ThemeIntent(BaseModel):
    """Structured representation of the creator's theme and editorial intent."""

    theme: str = Field(
        ...,
        description="The primary theme, highlight, or story focus of the video.",
        min_length=1,
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional additional creative notes, keywords, or preferences.",
    )

    @field_validator("theme")
    @classmethod
    def validate_non_empty_theme(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Theme cannot be empty or solely whitespace.")
        return trimmed
