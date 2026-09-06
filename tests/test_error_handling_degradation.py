"""Unit Tests for VideoGuru Error Handling & Graceful Degradation (SPEC-029)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from services.exceptions import (
    AudioDuckingError,
    GeminiApiError,
    IngestionError,
    RenderError,
    SchemaValidationError,
    VideoGuruError,
    WhisperError,
)
from services.resilience import (
    record_session_error,
    record_session_warning,
    safe_execute,
)
from tools.audio_ducking import apply_audio_ducking
from tools.curation_tools import assemble_edl_from_manifest
from tools.transition_renderer import render_with_transitions
from tools.whisper_captioning import generate_captions


class TestExceptionHierarchy:
    """Tests for VideoGuru exception taxonomy and serialization."""

    def test_base_error_to_dict(self):
        err = VideoGuruError(
            message="Test internal failure",
            category="custom",
            details={"debug_id": 123},
            user_message="Friendly message",
        )
        d = err.to_dict()
        assert d["error_type"] == "VideoGuruError"
        assert d["category"] == "custom"
        assert d["message"] == "Test internal failure"
        assert d["user_message"] == "Friendly message"
        assert d["details"]["debug_id"] == 123

    def test_ingestion_error(self):
        err = IngestionError(
            message="Corrupt header",
            file_path="media/clip.mp4",
        )
        assert isinstance(err, VideoGuruError)
        assert err.category == "ingestion"
        assert err.details["file_path"] == "media/clip.mp4"
        assert "Failed to ingest" in err.user_message

    def test_gemini_api_error(self):
        err = GeminiApiError(
            message="ResourceExhausted: Quota exceeded",
            status_code=429,
        )
        assert isinstance(err, VideoGuruError)
        assert err.category == "gemini_api"
        assert err.details["status_code"] == 429
        assert "heuristic curation" in err.user_message

    def test_schema_validation_error(self):
        err = SchemaValidationError(
            message="Missing transition intent",
            validation_errors=["transition_intent is required"],
        )
        assert isinstance(err, VideoGuruError)
        assert err.category == "schema_validation"
        assert len(err.details["validation_errors"]) == 1

    def test_render_error_subclasses(self):
        render_err = RenderError(message="Filter error", stderr="Error initializing filter")
        duck_err = AudioDuckingError(message="No audio stream in video")
        whisper_err = WhisperError(message="CUDA out of memory")

        assert isinstance(duck_err, RenderError)
        assert isinstance(whisper_err, RenderError)
        assert isinstance(duck_err, VideoGuruError)
        assert duck_err.category == "audio_ducking"
        assert whisper_err.category == "captioning"


class TestResilienceUtilities:
    """Tests for safe_execute and session warnings recorder."""

    def test_record_session_warning_dict_state(self):
        state: dict = {}
        record_session_warning(
            state=state,
            warning_message="Whisper GPU unavailable, using base model",
            category="captioning",
            details={"fallback": "base"},
        )
        assert "warnings" in state
        assert len(state["warnings"]) == 1
        w = state["warnings"][0]
        assert w["category"] == "captioning"
        assert "Whisper GPU unavailable" in w["message"]
        assert w["details"]["fallback"] == "base"
        assert "timestamp" in w

    def test_record_session_error_dict_state(self):
        state: dict = {}
        err = VideoGuruError("Fatal FFmpeg crash", user_message="Render failed")
        record_session_error(state=state, error=err, category="rendering")
        assert state["error"] == "Fatal FFmpeg crash"
        assert state["error_details"]["user_message"] == "Render failed"
        assert state["error_details"]["category"] == "rendering"

    def test_safe_execute_success(self):
        result = safe_execute(action=lambda: 40 + 2, default_value=0)
        assert result == 42

    def test_safe_execute_fallback(self):
        state: dict = {}
        result = safe_execute(
            action=lambda: 1 / 0,
            fallback=lambda exc: "fallback_value",
            state=state,
            category="math",
            warning_message="Division by zero occurred",
        )
        assert result == "fallback_value"
        assert len(state["warnings"]) == 1
        assert state["warnings"][0]["message"] == "Division by zero occurred"


class TestGracefulDegradationInTools:
    """Tests for graceful fallbacks in rendering, ducking, captions, and curation."""

    def test_audio_ducking_degrades_on_ffmpeg_failure(self, tmp_path):
        dummy_video = tmp_path / "video.mp4"
        dummy_video.write_bytes(b"dummy video")
        dummy_music = tmp_path / "music.mp3"
        dummy_music.write_bytes(b"dummy music")

        tool_state: dict = {}
        tool_ctx = MagicMock()
        tool_ctx.state = tool_state

        with patch("tools.audio_ducking.execute_ffmpeg_command", side_effect=RuntimeError("FFmpeg sidechain error")):
            out_video = apply_audio_ducking(
                video_path=dummy_video,
                music_path=dummy_music,
                output_path=tmp_path / "ducked.mp4",
                tool_context=tool_ctx,
            )

            # Returns source video path as graceful fallback
            assert out_video == str(dummy_video.resolve())
            assert tool_state.get("ducked_video_path") == str(dummy_video.resolve())
            assert "warnings" in tool_state
            assert any(w["category"] == "ducking" for w in tool_state["warnings"])

    def test_whisper_captioning_degrades_on_failure(self, tmp_path):
        dummy_video = tmp_path / "video.mp4"
        dummy_video.write_bytes(b"dummy video")

        tool_state: dict = {}
        tool_ctx = MagicMock()
        tool_ctx.state = tool_state

        with patch("tools.whisper_captioning.transcribe_audio_whisper", side_effect=RuntimeError("Whisper OOM")):
            cap_res = generate_captions(
                video_path=dummy_video,
                tool_context=tool_ctx,
                offline=True,
            )

            # Returns uncaptioned video as graceful fallback
            assert cap_res["captioned_video_path"] == str(dummy_video.resolve())
            assert cap_res["srt_path"] is None
            assert tool_state.get("captioned_video_path") == str(dummy_video.resolve())
            assert "warnings" in tool_state
            assert any(w["category"] == "captioning" for w in tool_state["warnings"])

    def test_transition_renderer_degrades_on_xfade_failure(self, tmp_path):
        edl = EditDecisionList(
            entries=[
                EDLEntry(file_reference="clip1", start_trim=0.0, end_trim=3.0, transition_intent=TransitionIntent.FADE, scene_rationale="Intro clip"),
                EDLEntry(file_reference="clip2", start_trim=0.0, end_trim=3.0, transition_intent=TransitionIntent.WIPE, scene_rationale="Action clip"),
            ]
        )
        manifest = [
            ClipManifestEntry(clip_id="clip1", file_name="c1.mp4", absolute_path=str(tmp_path / "c1.mp4"), duration_seconds=5.0, frame_rate=30.0, resolution="1920x1080", width=1920, height=1080, video_codec="h264"),
            ClipManifestEntry(clip_id="clip2", file_name="c2.mp4", absolute_path=str(tmp_path / "c2.mp4"), duration_seconds=5.0, frame_rate=30.0, resolution="1920x1080", width=1920, height=1080, video_codec="h264"),
        ]

        target_output = tmp_path / "final_out.mp4"
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        tool_state: dict = {}
        tool_ctx = MagicMock()
        tool_ctx.state = tool_state

        # Mock execute_ffmpeg_command: pre-trim calls succeed, xfade call fails, fallback concat succeeds
        def fake_ffmpeg(cmd):
            # Check if this is the xfade command
            cmd_str = " ".join(cmd)
            if "xfade" in cmd_str:
                raise RuntimeError("FFmpeg xfade filter failed!")
            # Otherwise normal pre-trim or fallback concat
            return cmd

        with patch("tools.transition_renderer.execute_ffmpeg_command", side_effect=fake_ffmpeg):
            rendered_path = render_with_transitions(
                edl=edl,
                clip_manifest=manifest,
                output_path=target_output,
                transition_duration=1.0,
                tool_context=tool_ctx,
                staging_dir=staging_dir,
            )

            assert rendered_path == str(target_output.resolve())
            assert "warnings" in tool_state
            assert any("Transition rendering failed" in w["message"] for w in tool_state["warnings"])

    def test_curation_degrades_on_single_clip_gemini_failure(self, tmp_path):
        manifest = [
            ClipManifestEntry(clip_id="clip1", file_name="c1.mp4", absolute_path=str(tmp_path / "c1.mp4"), duration_seconds=4.0, frame_rate=30.0, resolution="1920x1080", width=1920, height=1080, video_codec="h264"),
            ClipManifestEntry(clip_id="clip2", file_name="c2.mp4", absolute_path=str(tmp_path / "c2.mp4"), duration_seconds=6.0, frame_rate=30.0, resolution="1920x1080", width=1920, height=1080, video_codec="h264"),
        ]

        call_count = 0

        def flaky_analyze(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Gemini Quota Exceeded (429)")
            return EDLEntry(
                file_reference="clip2",
                start_trim=0.5,
                end_trim=4.5,
                transition_intent=TransitionIntent.CUT,
                scene_rationale="Live analysis successful",
            )

        with patch("tools.curation_tools.analyze_clip", side_effect=flaky_analyze):
            edl = assemble_edl_from_manifest(
                manifest=manifest,
                theme="Travel Highlights",
                offline=False,
            )

            assert len(edl) == 2
            assert edl.entries[0].file_reference == "clip1"
            assert edl.entries[1].file_reference == "clip2"
            assert edl.entries[1].scene_rationale == "Live analysis successful"
