"""Unit and Integration Tests for SPEC-024 EDL Schema Validation Callback."""

import json
import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from callbacks.schema_validation import (
    SchemaValidationError,
    create_schema_rejection_response,
    extract_json_payload,
    schema_validation_callback,
    validate_edl_data,
)
from schemas.media import ClipManifestEntry


@pytest.fixture
def sample_manifest():
    return [
        ClipManifestEntry(
            clip_id="vid_01",
            absolute_path="C:/media/clip_01.mp4",
            file_name="clip_01.mp4",
            duration_seconds=10.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
        ),
        ClipManifestEntry(
            clip_id="vid_02",
            absolute_path="C:/media/clip_02.mp4",
            file_name="clip_02.mp4",
            duration_seconds=20.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
        ),
    ]


@pytest.fixture
def valid_edl_raw():
    return [
        {
            "file_reference": "vid_01",
            "start_trim": 1.0,
            "end_trim": 5.5,
            "scene_rationale": "High energy intro establishing the hiking trail.",
            "transition_intent": "wipe",
        },
        {
            "file_reference": "vid_02",
            "start_trim": 2.0,
            "end_trim": 8.0,
            "scene_rationale": "Scenic mountain summit vista.",
            "transition_intent": "dissolve",
        },
    ]


class TestJsonExtraction:
    """Test suite for extract_json_payload."""

    def test_extract_raw_json(self):
        raw = '[{"file_reference": "vid_01", "start_trim": 0.0, "end_trim": 5.0}]'
        res = extract_json_payload(raw)
        assert isinstance(res, list)
        assert res[0]["file_reference"] == "vid_01"

    def test_extract_markdown_fenced_json(self):
        text = (
            "Here is the curated Edit Decision List:\n"
            "```json\n"
            '[{"file_reference": "vid_01", "start_trim": 1.0, "end_trim": 4.0}]\n'
            "```\n"
            "Let me know what you think!"
        )
        res = extract_json_payload(text)
        assert isinstance(res, list)
        assert res[0]["start_trim"] == 1.0

    def test_extract_invalid_text_returns_none(self):
        assert extract_json_payload("No JSON here at all") is None
        assert extract_json_payload("") is None


class TestValidateEdlData:
    """Test suite for validate_edl_data schema verification."""

    def test_valid_edl_passes(self, valid_edl_raw, sample_manifest):
        edl = validate_edl_data(valid_edl_raw, clip_manifest=sample_manifest)
        assert len(edl) == 2
        assert edl[0].file_reference == "vid_01"
        assert edl[1].transition_intent.value == "dissolve"

    def test_empty_edl_rejected(self):
        with pytest.raises(SchemaValidationError, match="empty"):
            validate_edl_data([])

    def test_invalid_timecode_rejected(self, valid_edl_raw):
        # end_trim <= start_trim
        bad_raw = list(valid_edl_raw)
        bad_raw[0] = dict(bad_raw[0])
        bad_raw[0]["end_trim"] = 0.5  # start is 1.0
        with pytest.raises(SchemaValidationError, match="greater than start_trim"):
            validate_edl_data(bad_raw)

    def test_invalid_transition_intent_rejected(self, valid_edl_raw):
        bad_raw = list(valid_edl_raw)
        bad_raw[0] = dict(bad_raw[0])
        bad_raw[0]["transition_intent"] = "spin_whirlwind"
        with pytest.raises(SchemaValidationError, match="Invalid transition_intent"):
            validate_edl_data(bad_raw)

    def test_unknown_clip_id_against_manifest_rejected(self, valid_edl_raw, sample_manifest):
        bad_raw = list(valid_edl_raw)
        bad_raw[0] = dict(bad_raw[0])
        bad_raw[0]["file_reference"] = "vid_999_nonexistent"
        with pytest.raises(SchemaValidationError, match="unknown clip_id"):
            validate_edl_data(bad_raw, clip_manifest=sample_manifest)

    def test_end_trim_exceeding_duration_rejected(self, valid_edl_raw, sample_manifest):
        # vid_01 duration is 10.0s
        bad_raw = list(valid_edl_raw)
        bad_raw[0] = dict(bad_raw[0])
        bad_raw[0]["end_trim"] = 15.0
        with pytest.raises(SchemaValidationError, match="exceeds source clip duration"):
            validate_edl_data(bad_raw, clip_manifest=sample_manifest)


class TestSchemaValidationCallback:
    """Test suite for ADK after_model_callback integration."""

    def _make_response(self, text: str) -> LlmResponse:
        return LlmResponse(
            content=types.Content(
                parts=[types.Part.from_text(text=text)]
            )
        )

    def test_callback_ignores_conversational_response(self):
        resp = self._make_response("Welcome to VideoGuru! What is the theme of your video today?")
        result = schema_validation_callback(None, resp)
        assert result is None  # Allowed to pass

    def test_callback_approves_valid_edl_response(self, valid_edl_raw, sample_manifest):
        text = f"Here is the finalized EDL:\n```json\n{json.dumps(valid_edl_raw)}\n```"
        resp = self._make_response(text)
        result = schema_validation_callback(None, resp, clip_manifest=sample_manifest)
        assert result is None  # Allowed to pass

    def test_callback_rejects_malformed_edl_and_requests_regeneration(self, sample_manifest):
        # EDL keywords present but JSON is malformed
        text = "start_trim and end_trim: [file_reference: not real json"
        resp = self._make_response(text)
        result = schema_validation_callback(None, resp, clip_manifest=sample_manifest)

        assert isinstance(result, LlmResponse)
        response_text = result.content.parts[0].text
        assert "[SCHEMA VALIDATION ERROR]" in response_text
        assert "Correction Instructions" in response_text

    def test_callback_rejects_schema_violation(self, valid_edl_raw, sample_manifest):
        bad_raw = list(valid_edl_raw)
        bad_raw[0] = dict(bad_raw[0])
        bad_raw[0]["end_trim"] = 0.2  # less than start_trim (1.0)
        text = f"```json\n{json.dumps(bad_raw)}\n```"
        resp = self._make_response(text)
        result = schema_validation_callback(None, resp, clip_manifest=sample_manifest)

        assert isinstance(result, LlmResponse)
        response_text = result.content.parts[0].text
        assert "[SCHEMA VALIDATION ERROR]" in response_text
        assert "greater than start_trim" in response_text

    def test_callback_raises_when_raise_exception_true(self, valid_edl_raw, sample_manifest):
        bad_raw = list(valid_edl_raw)
        bad_raw[0] = dict(bad_raw[0])
        bad_raw[0]["start_trim"] = 99.0
        bad_raw[0]["end_trim"] = 10.0
        text = json.dumps(bad_raw)
        resp = self._make_response(text)
        with pytest.raises(SchemaValidationError):
            schema_validation_callback(
                None,
                resp,
                clip_manifest=sample_manifest,
                raise_exception=True,
            )

    def test_callback_uses_manifest_from_context_state(self, valid_edl_raw, sample_manifest):
        class MockSession:
            state = {"clip_manifest": sample_manifest}

        class MockContext:
            session = MockSession()

        text = json.dumps(valid_edl_raw)
        resp = self._make_response(text)
        # Should cross-reference manifest from mock context session state
        result = schema_validation_callback(MockContext(), resp)
        assert result is None  # Valid
