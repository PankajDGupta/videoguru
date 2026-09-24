"""Overlay Text schemas for VideoGuru (SPEC-031).

Defines the Pydantic models for per-cut overlay text configuration:
- OverlayTextStyle: Font, color, position, and animation settings
- OverlayTextEntry: Individual overlay text for a specific cut
- OverlayTextPlan: Container for all overlay text entries in a video
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator


class TextPosition(str, Enum):
    """Vertical position presets for overlay text."""
    TOP = "top"           # Upper area (y = h*0.08)
    UPPER_THIRD = "upper_third"  # Upper third (y = h*0.15)
    CENTER = "center"     # Center (y = h*0.45)
    LOWER_THIRD = "lower_third"  # Lower third (y = h*0.72)
    BOTTOM = "bottom"     # Bottom area (y = h*0.88)


# Vibrant color palette optimized for YouTube Shorts visibility
YOUTUBE_SHORTS_COLORS = [
    "#FFD700",  # Gold/Yellow
    "#00FFFF",  # Cyan
    "#FF1493",  # Deep Pink
    "#00FF7F",  # Spring Green
    "#FF6347",  # Tomato Red
    "#7B68EE",  # Medium Slate Blue
    "#FF8C00",  # Dark Orange
    "#00BFFF",  # Deep Sky Blue
]


class OverlayTextStyle(BaseModel):
    """Visual styling for overlay text."""
    font_family: str = Field(
        default="Impact",
        description="Font family name (must be available on the system).",
    )
    font_size: int = Field(
        default=48,
        ge=12,
        le=120,
        description="Font size in pixels.",
    )
    font_color: str = Field(
        default="#FFD700",
        description="Text color as hex string (e.g. '#FFD700').",
    )
    border_color: str = Field(
        default="#000000",
        description="Text outline/border color as hex string.",
    )
    border_width: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Text outline width in pixels.",
    )
    shadow_color: str = Field(
        default="#000000",
        description="Drop shadow color.",
    )
    shadow_x: int = Field(default=2, description="Shadow X offset.")
    shadow_y: int = Field(default=2, description="Shadow Y offset.")
    position: TextPosition = Field(
        default=TextPosition.UPPER_THIRD,
        description="Vertical position preset.",
    )
    box_enabled: bool = Field(
        default=True,
        description="Whether to draw a semi-transparent background box.",
    )
    box_color: str = Field(
        default="black",
        description="Background box color.",
    )
    box_opacity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Background box opacity (0.0 = transparent, 1.0 = opaque).",
    )
    box_border_width: int = Field(
        default=15,
        ge=0,
        le=40,
        description="Padding around text in the background box.",
    )

    @field_validator("position", mode="before")
    @classmethod
    def normalize_position(cls, v: Any) -> TextPosition:
        if isinstance(v, TextPosition):
            return v
        if isinstance(v, str):
            try:
                return TextPosition(v.strip().lower())
            except ValueError:
                return TextPosition.UPPER_THIRD
        return TextPosition.UPPER_THIRD


class OverlayTextEntry(BaseModel):
    """A single overlay text element for a specific cut in the video."""
    cut_index: int = Field(
        ...,
        ge=0,
        description="Index of the cut in the EDL this text applies to.",
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The overlay text content to display.",
    )
    start_time: float = Field(
        ...,
        ge=0.0,
        description="Start time in seconds (relative to the full rendered video timeline) when text appears.",
    )
    end_time: float = Field(
        ...,
        ge=0.0,
        description="End time in seconds when text disappears.",
    )
    style: OverlayTextStyle = Field(
        default_factory=OverlayTextStyle,
        description="Visual styling for this text element.",
    )
    is_hook: bool = Field(
        default=False,
        description="Whether this is the primary hook text (uses larger font).",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Overlay text cannot be empty.")
        return trimmed


class OverlayTextPlan(BaseModel):
    """Container for all overlay text entries for a rendered video."""
    entries: list[OverlayTextEntry] = Field(
        default_factory=list,
        description="Ordered list of overlay text entries.",
    )
    theme: str = Field(
        default="",
        description="The video theme used to generate overlay text.",
    )

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: Union[int, slice]) -> Union[OverlayTextEntry, list[OverlayTextEntry]]:
        return self.entries[index]

    def __bool__(self) -> bool:
        return bool(self.entries)

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [entry.model_dump() for entry in self.entries]

    @classmethod
    def from_list(cls, raw_list: list[dict[str, Any]], theme: str = "") -> OverlayTextPlan:
        entries = [OverlayTextEntry.model_validate(e) for e in raw_list]
        return cls(entries=entries, theme=theme)
