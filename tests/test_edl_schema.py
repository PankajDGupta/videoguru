"""Unit and integration tests for SPEC-007: Pydantic EDL Schema.

Covers:
- TransitionIntent enum validation and parsing
- EDLEntry model validation, trim bounds, and property calculations
- EditDecisionList container behavior, duration aggregation, and clip validation
- JSON serialization, disk file round-trips, and CLI --validate-edl integration
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest
from pydantic import ValidationError

from main import main
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry


class TestTransitionIntentEnum:
    """Test suite for TransitionIntent enum behaviors."""

    def test_canonical_enum_values(self):
        assert TransitionIntent.CUT.value == "cut"
        assert TransitionIntent.FADE.value == "fade"
        assert TransitionIntent.WIPE.value == "wipe"
        assert TransitionIntent.SLIDE.value == "slide"
        assert TransitionIntent.DISSOLVE.value == "dissolve"

    def test_values_helper(self):
        vals = TransitionIntent.values()
        assert vals == ["cut", "fade", "wipe", "slide", "dissolve"]

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("cut", TransitionIntent.CUT),
            ("CUT", TransitionIntent.CUT),
            ("  Fade  ", TransitionIntent.FADE),
            ("wipe", TransitionIntent.WIPE),
            ("SLIDE", TransitionIntent.SLIDE),
            ("dissolve", TransitionIntent.DISSOLVE),
            (TransitionIntent.WIPE, TransitionIntent.WIPE),
        ],
    )
    def test_from_str_valid(self, input_val, expected):
        assert TransitionIntent.from_str(input_val) == expected

    def test_from_str_invalid_string(self):
        with pytest.raises(ValueError, match="Invalid transition_intent: 'glitch'"):
            TransitionIntent.from_str("glitch")

    def test_from_str_invalid_type(self):
        with pytest.raises(ValueError, match="Expected str or TransitionIntent"):
            TransitionIntent.from_str(1234)  # type: ignore


class TestEDLEntrySchema:
    """Test suite for EDLEntry Pydantic model validation."""

    def test_valid_entry_full(self):
        entry = EDLEntry(
            file_reference="vid_8f3a9b21",
            start_trim=2.5,
            end_trim=7.8,
            scene_rationale="High-energy intro sequence hook",
            transition_intent=TransitionIntent.FADE,
        )
        assert entry.file_reference == "vid_8f3a9b21"
        assert entry.clip_id == "vid_8f3a9b21"
        assert entry.start_trim == 2.5
        assert entry.end_trim == 7.8
        assert entry.scene_rationale == "High-energy intro sequence hook"
        assert entry.transition_intent == TransitionIntent.FADE
        assert pytest.approx(entry.duration, 0.001) == 5.3

    def test_valid_entry_defaults(self):
        entry = EDLEntry(
            file_reference="vid_12345678",
            start_trim=0.0,
            end_trim=4.0,
            scene_rationale="Establishing landscape shot",
        )
        assert entry.transition_intent == TransitionIntent.CUT
        assert entry.duration == 4.0

    def test_string_transition_intent_normalized(self):
        entry = EDLEntry(
            file_reference="vid_12345678",
            start_trim=0.0,
            end_trim=3.5,
            scene_rationale="Montage cut",
            transition_intent="dissolve",  # type: ignore
        )
        assert entry.transition_intent == TransitionIntent.DISSOLVE

    def test_zero_duration_rejected(self):
        with pytest.raises(ValidationError, match="end_trim .* must be strictly greater than start_trim"):
            EDLEntry(
                file_reference="vid_001",
                start_trim=5.0,
                end_trim=5.0,
                scene_rationale="Zero length scene",
            )

    def test_inverted_trim_rejected(self):
        with pytest.raises(ValidationError, match="end_trim .* must be strictly greater than start_trim"):
            EDLEntry(
                file_reference="vid_001",
                start_trim=10.0,
                end_trim=4.0,
                scene_rationale="Inverted trim",
            )

    def test_negative_start_trim_rejected(self):
        with pytest.raises(ValidationError):
            EDLEntry(
                file_reference="vid_001",
                start_trim=-1.0,
                end_trim=5.0,
                scene_rationale="Negative start",
            )

    def test_negative_end_trim_rejected(self):
        with pytest.raises(ValidationError):
            EDLEntry(
                file_reference="vid_001",
                start_trim=0.0,
                end_trim=-0.5,
                scene_rationale="Negative end",
            )

    def test_empty_file_reference_rejected(self):
        with pytest.raises(ValidationError):
            EDLEntry(
                file_reference="",
                start_trim=0.0,
                end_trim=5.0,
                scene_rationale="Test",
            )

    def test_whitespace_file_reference_rejected(self):
        with pytest.raises(ValidationError, match="file_reference cannot be empty or solely whitespace"):
            EDLEntry(
                file_reference="   ",
                start_trim=0.0,
                end_trim=5.0,
                scene_rationale="Test",
            )

    def test_empty_scene_rationale_rejected(self):
        with pytest.raises(ValidationError):
            EDLEntry(
                file_reference="vid_001",
                start_trim=0.0,
                end_trim=5.0,
                scene_rationale="",
            )

    def test_whitespace_scene_rationale_rejected(self):
        with pytest.raises(ValidationError, match="scene_rationale cannot be empty or solely whitespace"):
            EDLEntry(
                file_reference="vid_001",
                start_trim=0.0,
                end_trim=5.0,
                scene_rationale=" \t \n ",
            )

    def test_invalid_transition_intent_type(self):
        with pytest.raises(ValidationError):
            EDLEntry(
                file_reference="vid_001",
                start_trim=0.0,
                end_trim=5.0,
                scene_rationale="Test",
                transition_intent=999,  # type: ignore
            )

    def test_json_schema_generation_for_gemini(self):
        """Verify the JSON schema is valid and contains all required Gemini structured output properties."""
        schema = EDLEntry.model_json_schema()
        assert "properties" in schema
        props = schema["properties"]
        assert "file_reference" in props
        assert "start_trim" in props
        assert "end_trim" in props
        assert "scene_rationale" in props
        assert "transition_intent" in props
        assert set(schema.get("required", [])) == {
            "file_reference",
            "start_trim",
            "end_trim",
            "scene_rationale",
        }


class TestEditDecisionListModel:
    """Test suite for EditDecisionList container and operations."""

    @pytest.fixture
    def sample_entries(self) -> list[EDLEntry]:
        return [
            EDLEntry(
                file_reference="vid_001",
                start_trim=0.0,
                end_trim=5.5,
                scene_rationale="Opening hook",
                transition_intent=TransitionIntent.FADE,
            ),
            EDLEntry(
                file_reference="vid_002",
                start_trim=1.0,
                end_trim=4.0,
                scene_rationale="Action cut",
                transition_intent=TransitionIntent.CUT,
            ),
            EDLEntry(
                file_reference="vid_001",
                start_trim=8.0,
                end_trim=12.0,
                scene_rationale="Conclusion shot",
                transition_intent=TransitionIntent.DISSOLVE,
            ),
        ]

    def test_empty_edl_initialization(self):
        edl = EditDecisionList()
        assert len(edl) == 0
        assert edl.total_duration == 0.0
        assert edl.clip_references == set()
        assert bool(edl) is False
        assert list(edl) == []

    def test_initialization_with_entries_keyword(self, sample_entries):
        edl = EditDecisionList(entries=sample_entries)
        assert len(edl) == 3
        assert bool(edl) is True
        assert pytest.approx(edl.total_duration, 0.001) == (5.5 + 3.0 + 4.0)

    def test_initialization_with_bare_list(self, sample_entries):
        edl = EditDecisionList(sample_entries)  # type: ignore
        assert len(edl) == 3
        assert edl[0].file_reference == "vid_001"
        assert edl[1].file_reference == "vid_002"

    def test_initialization_from_list_of_dicts(self):
        data = [
            {
                "file_reference": "vid_abc",
                "start_trim": 0.0,
                "end_trim": 3.0,
                "scene_rationale": "Sample cut",
                "transition_intent": "wipe",
            }
        ]
        edl = EditDecisionList.from_list(data)
        assert len(edl) == 1
        assert edl[0].file_reference == "vid_abc"
        assert edl[0].transition_intent == TransitionIntent.WIPE
        assert edl.total_duration == 3.0

    def test_container_indexing_and_slicing(self, sample_entries):
        edl = EditDecisionList(entries=sample_entries)
        assert edl[1].file_reference == "vid_002"
        sliced = edl[0:2]
        assert isinstance(sliced, list)
        assert len(sliced) == 2
        assert sliced[0].file_reference == "vid_001"
        assert sliced[1].file_reference == "vid_002"

    def test_container_contains_check(self, sample_entries):
        edl = EditDecisionList(entries=sample_entries)
        assert sample_entries[0] in edl
        assert "vid_001" in edl
        assert "vid_002" in edl
        assert "vid_999" not in edl
        assert 12345 not in edl

    def test_append_and_extend(self, sample_entries):
        edl = EditDecisionList()
        edl.append(sample_entries[0])
        assert len(edl) == 1
        assert edl.total_duration == 5.5

        edl.extend(sample_entries[1:])
        assert len(edl) == 3
        assert pytest.approx(edl.total_duration, 0.001) == 12.5

    def test_append_invalid_type_raises(self):
        edl = EditDecisionList()
        with pytest.raises(TypeError, match="Expected EDLEntry"):
            edl.append({"file_reference": "not_an_entry"})  # type: ignore

    def test_clip_references_unique_set(self, sample_entries):
        edl = EditDecisionList(entries=sample_entries)
        # vid_001 is used twice, vid_002 once
        assert edl.clip_references == {"vid_001", "vid_002"}

    def test_validate_clip_references_success_with_strings(self, sample_entries):
        edl = EditDecisionList(entries=sample_entries)
        missing = edl.validate_clip_references(["vid_001", "vid_002", "vid_extra"], raise_on_error=True)
        assert missing == []

    def test_validate_clip_references_success_with_manifest_entries(self, sample_entries):
        manifest = [
            ClipManifestEntry(
                clip_id="vid_001",
                absolute_path="C:/media/01.mp4",
                file_name="01.mp4",
                duration_seconds=15.0,
                frame_rate=30.0,
                resolution="1920x1080",
                width=1920,
                height=1080,
                video_codec="h264",
            ),
            ClipManifestEntry(
                clip_id="vid_002",
                absolute_path="C:/media/02.mp4",
                file_name="02.mp4",
                duration_seconds=10.0,
                frame_rate=30.0,
                resolution="1920x1080",
                width=1920,
                height=1080,
                video_codec="h264",
            ),
        ]
        edl = EditDecisionList(entries=sample_entries)
        missing = edl.validate_clip_references(manifest, raise_on_error=True)
        assert missing == []

    def test_validate_clip_references_failure_raises(self, sample_entries):
        edl = EditDecisionList(entries=sample_entries)
        # Only vid_001 is in the valid set, vid_002 is missing
        with pytest.raises(ValueError, match="EDL references 1 unknown clip ID.*vid_002"):
            edl.validate_clip_references({"vid_001"}, raise_on_error=True)

    def test_validate_clip_references_failure_without_raise(self, sample_entries):
        edl = EditDecisionList(entries=sample_entries)
        missing = edl.validate_clip_references({"vid_001"}, raise_on_error=False)
        assert missing == ["vid_002"]

    def test_serialization_dict_list_and_json(self, sample_entries):
        edl = EditDecisionList(entries=sample_entries)
        dicts = edl.to_dict_list()
        assert len(dicts) == 3
        assert dicts[0]["file_reference"] == "vid_001"
        assert dicts[0]["transition_intent"] == "fade"

        json_str = edl.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_save_and_load_file_round_trip(self, sample_entries, tmp_path):
        edl = EditDecisionList(entries=sample_entries)
        out_file = tmp_path / "edl_test.json"

        saved_path = edl.save_json(out_file)
        assert saved_path.is_file()

        loaded_edl = EditDecisionList.from_file(saved_path)
        assert len(loaded_edl) == len(edl)
        assert pytest.approx(loaded_edl.total_duration, 0.001) == edl.total_duration
        assert loaded_edl[0].file_reference == edl[0].file_reference
        assert loaded_edl[0].transition_intent == edl[0].transition_intent

    def test_from_file_not_found_raises(self, tmp_path):
        non_existent = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError, match="EDL file not found"):
            EditDecisionList.from_file(non_existent)


class TestValidateEdlCliIntegration:
    """Test suite verifying the --validate-edl CLI command."""

    def test_cli_validate_edl_success(self, tmp_path, capsys, monkeypatch):
        test_file = tmp_path / "valid_edl.json"
        edl_data = [
            {
                "file_reference": "vid_101",
                "start_trim": 0.0,
                "end_trim": 4.5,
                "scene_rationale": "High-action hook",
                "transition_intent": "slide",
            },
            {
                "file_reference": "vid_102",
                "start_trim": 2.0,
                "end_trim": 8.0,
                "scene_rationale": "Reaction shot",
                "transition_intent": "fade",
            },
        ]
        test_file.write_text(json.dumps(edl_data), encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            ["main.py", "--validate-edl", str(test_file)],
        )

        main()
        captured = capsys.readouterr()
        assert "Validating Edit Decision List (SPEC-007)" in captured.out
        assert "EDL Valid: 2 cut(s) loaded successfully." in captured.out
        assert "Total Duration:    10.50 s" in captured.out
        assert "vid_101" in captured.out
        assert "vid_102" in captured.out

    def test_cli_validate_edl_failure(self, tmp_path, capsys, monkeypatch):
        bad_file = tmp_path / "invalid_edl.json"
        bad_file.write_text("{\"corrupted\": true}", encoding="utf-8")

        monkeypatch.setattr(
            sys,
            "argv",
            ["main.py", "--validate-edl", str(bad_file)],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "EDL validation failed" in captured.err
