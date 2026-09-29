"""Motion Graphics schemas for VideoGuru (SPEC-035).

Defines the Pydantic models describing theme-related animated motion graphics that are
composited on top of the rendered video:
- MotionGraphicType: The supported animated graphic templates
- MotionGraphicElement: One animated graphic with its timing and copy
- MotionGraphicsPlan: The ordered set of graphics plus the theme colour palette
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class MotionGraphicType(str, Enum):
    """Animated graphic templates the renderer knows how to draw."""

    KINETIC_TITLE = "kinetic_title"  # Large centred phrase that rises in, with a whooshing underline
    LOWER_THIRD = "lower_third"      # Broadcast-style title + caption strips sliding in from the left
    STAT_CALLOUT = "stat_callout"    # Big number / keyword that pops up with an overshoot
    CORNER_BADGE = "corner_badge"    # Small pill label sliding in from the right edge
    PROGRESS_BAR = "progress_bar"    # Thin bar that fills across the video for the element's lifetime


# Graphics that need on-screen copy (a progress bar is purely graphical)
TEXT_GRAPHIC_TYPES: frozenset[MotionGraphicType] = frozenset({
    MotionGraphicType.KINETIC_TITLE,
    MotionGraphicType.LOWER_THIRD,
    MotionGraphicType.STAT_CALLOUT,
    MotionGraphicType.CORNER_BADGE,
})

# Graphics that occupy the same screen zone and therefore must never overlap in time
SHARED_ZONE_GROUPS: tuple[frozenset[MotionGraphicType], ...] = (
    frozenset({MotionGraphicType.KINETIC_TITLE, MotionGraphicType.STAT_CALLOUT}),
)

DEFAULT_PRIMARY_COLOR = "#FF6B00"
DEFAULT_SECONDARY_COLOR = "#FFD700"

_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

# Emoji / pictographs cannot be rendered by the bundled fonts and would show as tofu boxes
_UNRENDERABLE_RE = re.compile(
    r"[\U00010000-\U0010ffff☀-➿⌀-⏿️\\]+",
    flags=re.UNICODE,
)


def normalize_hex_color(value: Any, default: str) -> str:
    """Return a canonical '#RRGGBB' string, or `default` if `value` is not a valid hex colour."""
    if isinstance(value, str):
        match = _HEX_COLOR_RE.match(value.strip())
        if match:
            return f"#{match.group(1).upper()}"
    return default


def clean_graphic_text(value: Optional[str]) -> Optional[str]:
    """Strip characters the overlay fonts cannot draw and collapse whitespace."""
    if value is None:
        return None
    cleaned = _UNRENDERABLE_RE.sub("", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


class MotionGraphicElement(BaseModel):
    """A single animated motion graphic on the rendered video timeline."""

    graphic_type: MotionGraphicType = Field(
        ..., description="Which animated template to draw."
    )
    start_time: float = Field(
        ..., ge=0.0, description="Second on the final timeline when the graphic starts animating in."
    )
    end_time: float = Field(
        ..., ge=0.0, description="Second on the final timeline when the graphic has finished animating out."
    )
    text: Optional[str] = Field(
        default=None,
        max_length=60,
        description="Primary copy: title, number, keyword or badge label. Not used by progress bars.",
    )
    subtext: Optional[str] = Field(
        default=None,
        max_length=80,
        description="Secondary caption (lower thirds and stat callouts only).",
    )
    accent_color: Optional[str] = Field(
        default=None,
        description="Optional '#RRGGBB' override; the plan's primary colour is used when omitted.",
    )
    cut_index: Optional[int] = Field(
        default=None, ge=0, description="EDL cut the graphic relates to, if any."
    )
    rationale: str = Field(
        default="", description="Why this graphic fits the scene (for debugging / review)."
    )

    @field_validator("text", "subtext", mode="before")
    @classmethod
    def _clean_copy(cls, v: Any) -> Optional[str]:
        return clean_graphic_text(v) if isinstance(v, str) else v

    @field_validator("accent_color", mode="before")
    @classmethod
    def _clean_color(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        normalized = normalize_hex_color(v, "")
        return normalized or None

    @model_validator(mode="after")
    def _check_consistency(self) -> "MotionGraphicElement":
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})."
            )
        if self.graphic_type in TEXT_GRAPHIC_TYPES and not self.text:
            raise ValueError(f"'{self.graphic_type.value}' graphics require non-empty text.")
        return self

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class MotionGraphicsPlan(BaseModel):
    """Container for all motion graphics planned for a rendered video."""

    elements: list[MotionGraphicElement] = Field(default_factory=list)
    theme: str = Field(default="", description="Video theme the plan was designed for.")
    content_summary: str = Field(
        default="", description="One-sentence description of what the analysed video shows."
    )
    primary_color: str = Field(default=DEFAULT_PRIMARY_COLOR)
    secondary_color: str = Field(default=DEFAULT_SECONDARY_COLOR)

    @field_validator("primary_color", mode="before")
    @classmethod
    def _clean_primary(cls, v: Any) -> str:
        return normalize_hex_color(v, DEFAULT_PRIMARY_COLOR)

    @field_validator("secondary_color", mode="before")
    @classmethod
    def _clean_secondary(cls, v: Any) -> str:
        return normalize_hex_color(v, DEFAULT_SECONDARY_COLOR)

    def __iter__(self):
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, index: Union[int, slice]):
        return self.elements[index]

    def __bool__(self) -> bool:
        return bool(self.elements)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Serialise elements with the plan palette resolved into `accent_color`/`secondary_color`."""
        rows: list[dict[str, Any]] = []
        for element in self.elements:
            row = element.model_dump(mode="json")
            row["accent_color"] = element.accent_color or self.primary_color
            row["secondary_color"] = self.secondary_color
            rows.append(row)
        return rows

    @classmethod
    def from_list(
        cls,
        raw_list: list[dict[str, Any]],
        theme: str = "",
        primary_color: Optional[str] = None,
        secondary_color: Optional[str] = None,
    ) -> "MotionGraphicsPlan":
        """Rebuild a plan from `to_dict_list()` output (as stored in session state)."""
        elements = [MotionGraphicElement.model_validate(row) for row in raw_list]
        palette_primary = primary_color
        palette_secondary = secondary_color
        if raw_list:
            palette_secondary = palette_secondary or raw_list[0].get("secondary_color")
        return cls(
            elements=elements,
            theme=theme,
            primary_color=palette_primary or DEFAULT_PRIMARY_COLOR,
            secondary_color=palette_secondary or DEFAULT_SECONDARY_COLOR,
        )
