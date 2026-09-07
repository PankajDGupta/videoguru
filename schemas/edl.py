"""Edit Decision List (EDL) schemas for VideoGuru (SPEC-007).

Defines the Pydantic models for structured editorial decisions:
- TransitionIntent: Enum of supported visual transitions
- EDLEntry: Individual cut decision specifying source clip, trims, and intent
- EditDecisionList: Container model representing an ordered timeline of cuts
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TransitionIntent(str, Enum):
    """Supported visual transition families between video segments (SPEC-007)."""

    CUT = "cut"
    FADE = "fade"
    WIPE = "wipe"
    SLIDE = "slide"
    DISSOLVE = "dissolve"

    @classmethod
    def from_str(cls, value: Union[str, TransitionIntent]) -> TransitionIntent:
        """Parse a string or enum instance into a canonical TransitionIntent."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError(f"Expected str or TransitionIntent, got {type(value).__name__}: {value}")
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            allowed = ", ".join(f"'{m.value}'" for m in cls)
            raise ValueError(
                f"Invalid transition_intent: '{value}'. Allowed transitions are: {allowed}"
            )

    @classmethod
    def values(cls) -> list[str]:
        """Return list of supported transition string values."""
        return [m.value for m in cls]


class EDLEntry(BaseModel):
    """An individual Edit Decision List (EDL) entry representing a trimmed scene cut."""

    file_reference: str = Field(
        ...,
        description="The unique clip identifier matching the local ingestion map (e.g. 'vid_8f3a9b21').",
        min_length=1,
    )
    start_trim: float = Field(
        ...,
        description="The exact second within the source clip where the segment begins (>= 0.0).",
        ge=0.0,
    )
    end_trim: float = Field(
        ...,
        description="The exact second within the source clip where the segment ends (> start_trim).",
        ge=0.0,
    )
    scene_rationale: str = Field(
        ...,
        description="A brief justification explaining why this clip segment was selected.",
        min_length=1,
    )
    transition_intent: TransitionIntent = Field(
        default=TransitionIntent.CUT,
        description="Suggested visual transition to this or subsequent segment (cut, fade, wipe, slide, dissolve).",
    )
    engagement_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=10.0,
        description="Quality, narrative resonance, and thematic relevance score (0.0 to 10.0) evaluated by Gemini multimodal analysis.",
    )
    playback_speed: float = Field(
        default=1.0,
        ge=0.25,
        le=16.0,
        description="Playback speed multiplier for the clip segment (e.g. 1.0 for normal speed, 2.0 for 2x fast-forward, 4.0 for montage/timelapse).",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_speed_alias(cls, data: Any) -> Any:
        """Allow 'speed' as an alias for 'playback_speed'."""
        if isinstance(data, dict):
            if "speed" in data and "playback_speed" not in data:
                data["playback_speed"] = data["speed"]
        return data

    @field_validator("file_reference")
    @classmethod
    def validate_file_reference(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("file_reference cannot be empty or solely whitespace.")
        return trimmed

    @field_validator("scene_rationale")
    @classmethod
    def validate_scene_rationale(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("scene_rationale cannot be empty or solely whitespace.")
        return trimmed

    @field_validator("transition_intent", mode="before")
    @classmethod
    def normalize_transition_intent(cls, v: Any) -> TransitionIntent:
        if isinstance(v, TransitionIntent):
            return v
        if isinstance(v, str):
            return TransitionIntent.from_str(v)
        raise ValueError(f"Invalid type for transition_intent: {type(v).__name__}")

    @field_validator("playback_speed", mode="before")
    @classmethod
    def normalize_playback_speed(cls, v: Any) -> float:
        if v is None:
            return 1.0
        try:
            val = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid playback_speed: {v}. Expected numeric value.")
        if val <= 0.0:
            raise ValueError(f"playback_speed must be positive, got {val}")
        return val

    @model_validator(mode="after")
    def validate_trim_duration(self) -> EDLEntry:
        """Ensure end_trim is strictly greater than start_trim."""
        if self.end_trim <= self.start_trim:
            raise ValueError(
                f"end_trim ({self.end_trim}) must be strictly greater than start_trim ({self.start_trim})."
            )
        return self

    @property
    def source_duration(self) -> float:
        """Source segment duration in seconds prior to speed modification."""
        return self.end_trim - self.start_trim

    @property
    def duration(self) -> float:
        """Calculated timeline segment duration in seconds after speed adjustment."""
        speed = self.playback_speed if self.playback_speed > 0 else 1.0
        return (self.end_trim - self.start_trim) / speed

    @property
    def is_fast_forward(self) -> bool:
        """Return True if this segment is sped up beyond real-time playback (speed > 1.0)."""
        return self.playback_speed > 1.0

    @property
    def clip_id(self) -> str:
        """Alias for file_reference, matching ClipManifestEntry.clip_id."""
        return self.file_reference


class EditDecisionList(BaseModel):
    """Container model representing an ordered sequence of EDL cut entries."""

    entries: list[EDLEntry] = Field(
        default_factory=list,
        description="Ordered sequence of edit decision list entries composing the video timeline.",
    )

    def __init__(self, *args: Any, **data: Any) -> None:
        """Initialize EditDecisionList from either positional list/entries or keyword args."""
        if args:
            if len(args) == 1 and isinstance(args[0], (list, tuple)):
                super().__init__(entries=list(args[0]), **data)
            else:
                super().__init__(entries=list(args), **data)
        else:
            super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def handle_input(cls, data: Any) -> Any:
        """Allow initializing EditDecisionList directly with a list of entries or dict with 'entries'."""
        if isinstance(data, (list, tuple)):
            return {"entries": list(data)}
        if isinstance(data, dict):
            if not data:
                return data
            if "entries" not in data:
                raise ValueError("JSON object must contain an 'entries' key holding the list of EDL cuts.")
            return data
        return data

    def __iter__(self):
        """Iterate over the underlying EDL entries."""
        return iter(self.entries)

    def __len__(self) -> int:
        """Return the number of entries in the EDL."""
        return len(self.entries)

    def __getitem__(self, index: Union[int, slice]) -> Union[EDLEntry, list[EDLEntry]]:
        """Access entries by index or slice."""
        return self.entries[index]

    def __contains__(self, item: Any) -> bool:
        """Check if an EDLEntry or clip_id string is in the EDL."""
        if isinstance(item, EDLEntry):
            return item in self.entries
        if isinstance(item, str):
            return any(entry.file_reference == item for entry in self.entries)
        return False

    def __bool__(self) -> bool:
        """Return True if EDL contains at least one entry."""
        return bool(self.entries)

    def append(self, entry: EDLEntry) -> None:
        """Append an entry to the EDL."""
        if not isinstance(entry, EDLEntry):
            raise TypeError(f"Expected EDLEntry, got {type(entry).__name__}")
        self.entries.append(entry)

    def extend(self, entries: Iterable[EDLEntry]) -> None:
        """Extend the EDL with multiple entries."""
        for item in entries:
            self.append(item)

    @property
    def total_duration(self) -> float:
        """Total duration of all cut segments in seconds."""
        return sum(entry.duration for entry in self.entries)

    @property
    def clip_references(self) -> set[str]:
        """Set of unique source clip IDs referenced in this EDL."""
        return {entry.file_reference for entry in self.entries}

    def validate_clip_references(
        self,
        valid_clips: Union[set[str], Sequence[str], Sequence[Any]],
        raise_on_error: bool = True,
    ) -> list[str]:
        """Validate that all file_reference values in entries exist in valid_clips.

        Args:
            valid_clips: Set/list of clip_id strings or objects with a `clip_id` attribute (e.g. ClipManifestEntry).
            raise_on_error: If True, raises ValueError if invalid references are found.

        Returns:
            List of invalid file_reference strings (empty if all valid).
        """
        valid_set: set[str] = set()
        for item in valid_clips:
            if isinstance(item, str):
                valid_set.add(item.strip())
            elif hasattr(item, "clip_id"):
                valid_set.add(str(item.clip_id).strip())
            else:
                valid_set.add(str(item).strip())

        missing = [
            entry.file_reference
            for entry in self.entries
            if entry.file_reference not in valid_set
        ]

        if missing and raise_on_error:
            unique_missing = sorted(set(missing))
            raise ValueError(
                f"EDL references {len(unique_missing)} unknown clip ID(s) not in manifest: "
                f"{', '.join(unique_missing)}"
            )

        return missing

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Return raw list of dictionaries suitable for JSON serialization or Gemini context."""
        return [entry.model_dump() for entry in self.entries]

    def to_json(self, indent: int = 2) -> str:
        """Serialize the EDL to a formatted JSON string."""
        return json.dumps(self.to_dict_list(), indent=indent)

    def save_json(self, file_path: Union[Path, str], indent: int = 2) -> Path:
        """Save the EDL to a JSON file on disk."""
        target = Path(file_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")
        return target

    @classmethod
    def from_list(cls, raw_list: list[Any]) -> EditDecisionList:
        """Construct an EditDecisionList from a list of dicts or EDLEntry objects."""
        return cls.model_validate(raw_list)

    @classmethod
    def from_dict_list(cls, raw_list: list[Any]) -> EditDecisionList:
        """Alias for from_list."""
        return cls.model_validate(raw_list)


    @classmethod
    def from_json(cls, json_str: str) -> EditDecisionList:
        """Construct an EditDecisionList from a JSON string (either array or object with 'entries')."""
        data = json.loads(json_str)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> EditDecisionList:
        """Load and parse an EditDecisionList from a JSON file on disk."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"EDL file not found: {p}")
        content = p.read_text(encoding="utf-8")
        return cls.from_json(content)
