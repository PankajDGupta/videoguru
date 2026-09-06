"""Unit and integration tests for SPEC-008: Gemini Video Analysis Tool.

Covers:
- Input validation (file paths, video extensions, theme strings)
- Prompt generation and structured schema configuration
- Offline/mock execution mode and duration boundary clamping
- Gemini API client lifecycle mocking (files.upload, polling, files.delete, generate_content)
- Guaranteed cleanup of uploaded Files API resources on success and failure
- ToolContext session state integration
- CLI dispatch via main.py (--analyze-clip, --theme, --clip-id, --offline)
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from google.adk.events import EventActions
from google.adk.tools import ToolContext


def _create_mock_tool_context(initial_state: dict[str, Any] | None = None) -> ToolContext:
    actions = EventActions()
    mock_invocation = MagicMock()
    mock_invocation.session.state = initial_state if initial_state is not None else {}
    return ToolContext(invocation_context=mock_invocation, event_actions=actions)

from main import main
from schemas.edl import EDLEntry, TransitionIntent
from tools.clip_metadata import find_ffprobe_executable
from tools.video_analysis import (
    ANALYSIS_SYSTEM_INSTRUCTION,
    analyze_clip,
    build_analysis_prompt,
    mock_clip_analysis,
    poll_file_active,
)


@pytest.fixture(scope="module")
def sample_video_clip(tmp_path_factory) -> Path:
    """Generate a valid short MP4 clip for analysis testing using FFmpeg."""
    ffprobe_bin = find_ffprobe_executable()
    ffmpeg_bin = str(Path(ffprobe_bin).parent / "ffmpeg.exe")
    if not Path(ffmpeg_bin).exists():
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    temp_dir = tmp_path_factory.mktemp("video_analysis_media")
    clip_path = temp_dir / "sample_clip.mp4"

    cmd = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2.0:size=1280x720:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=2.0",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(clip_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return clip_path


class TestPromptAndMockHelpers:
    """Test suite for prompt construction and mock analysis helper."""

    def test_build_analysis_prompt_contents(self):
        prompt = build_analysis_prompt(
            theme="Scenic road trip",
            clip_id="vid_test_123",
            duration=12.5,
            resolution="1920x1080",
            frame_rate=29.97,
        )
        assert "vid_test_123" in prompt
        assert "12.50 seconds" in prompt
        assert "1920x1080" in prompt
        assert "29.97 fps" in prompt
        assert "Scenic road trip" in prompt
        assert "file_reference" in prompt
        assert "start_trim" in prompt
        assert "end_trim" in prompt
        assert "scene_rationale" in prompt
        assert "transition_intent" in prompt
        assert "engagement_score" in prompt

    def test_mock_clip_analysis_short_clip(self):
        entry = mock_clip_analysis(clip_id="vid_short", duration=0.8, theme="Quick flash")
        assert isinstance(entry, EDLEntry)
        assert entry.file_reference == "vid_short"
        assert entry.start_trim == 0.0
        assert entry.end_trim <= 0.8
        assert entry.duration > 0
        assert entry.engagement_score == 8.5
        assert "Quick flash" in entry.scene_rationale

    def test_mock_clip_analysis_medium_clip(self):
        entry = mock_clip_analysis(clip_id="vid_med", duration=15.0, theme="Mountain biking")
        assert isinstance(entry, EDLEntry)
        assert entry.file_reference == "vid_med"
        assert entry.start_trim >= 0.0
        assert entry.end_trim <= 15.0
        assert entry.end_trim > entry.start_trim
        assert entry.engagement_score == 8.5
        assert "Mountain biking" in entry.scene_rationale


class TestVideoAnalysisValidation:
    """Test suite validating input boundaries and errors."""

    def test_empty_clip_path_raises_value_error(self):
        with pytest.raises(ValueError, match="clip_path must be a non-empty string"):
            analyze_clip(clip_path="   ", theme="Vlog")

    def test_nonexistent_clip_raises_file_not_found(self, tmp_path):
        ghost_path = tmp_path / "ghost_clip.mp4"
        with pytest.raises(FileNotFoundError, match="Video clip does not exist"):
            analyze_clip(clip_path=str(ghost_path), theme="Vlog")

    def test_directory_path_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="not a regular file"):
            analyze_clip(clip_path=str(tmp_path), theme="Vlog")

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        text_file = tmp_path / "notes.txt"
        text_file.write_text("not a video")
        with pytest.raises(ValueError, match="Unsupported video format"):
            analyze_clip(clip_path=str(text_file), theme="Vlog")

    def test_empty_theme_raises_value_error(self, sample_video_clip):
        with pytest.raises(ValueError, match="Theme must be a non-empty string"):
            analyze_clip(clip_path=str(sample_video_clip), theme="   ")

    def test_theme_resolved_from_tool_context_state(self, sample_video_clip):
        context = _create_mock_tool_context({"theme": "Surfing in Bali"})

        entry = analyze_clip(
            clip_path=str(sample_video_clip),
            theme="",
            offline=True,
            tool_context=context,
        )
        assert isinstance(entry, EDLEntry)
        assert "Surfing in Bali" in entry.scene_rationale
        assert "clip_analyses" in context.state
        assert entry.file_reference in context.state["clip_analyses"]

    def test_missing_api_key_without_offline_raises_value_error(self, sample_video_clip, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is not set"):
            analyze_clip(clip_path=str(sample_video_clip), theme="Travel vlog", offline=False)


class TestOfflineAnalysis:
    """Test offline execution mode."""

    def test_analyze_clip_offline_mode(self, sample_video_clip):
        entry = analyze_clip(
            clip_path=str(sample_video_clip),
            theme="Urban exploration",
            clip_id="vid_custom_001",
            offline=True,
        )
        assert isinstance(entry, EDLEntry)
        assert entry.file_reference == "vid_custom_001"
        assert entry.start_trim >= 0.0
        assert entry.end_trim <= 2.0
        assert entry.end_trim > entry.start_trim
        assert entry.transition_intent in TransitionIntent
        assert entry.engagement_score is not None
        assert 0.0 <= entry.engagement_score <= 10.0
        assert "Urban exploration" in entry.scene_rationale

    def test_offline_mode_auto_generates_clip_id(self, sample_video_clip):
        entry = analyze_clip(
            clip_path=str(sample_video_clip),
            theme="Cooking masterclass",
            offline=True,
        )
        assert entry.file_reference.startswith("vid_")


class TestGeminiLiveAnalysisWithMockClient:
    """Test live multimodal analysis with mocked google.genai.Client."""

    def _create_mock_client(self, response_text: str, file_state="ACTIVE"):
        mock_client = MagicMock()

        # Mock files.upload and files.get
        mock_file = MagicMock()
        mock_file.name = "files/test_clip_upload_987"
        mock_file.state = file_state

        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        mock_client.files.delete.return_value = None

        # Mock models.generate_content
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_response.parsed = None
        mock_client.models.generate_content.return_value = mock_response

        return mock_client, mock_file

    def test_successful_gemini_analysis(self, sample_video_clip):
        valid_response_json = json.dumps(
            {
                "file_reference": "vid_sample_123",
                "start_trim": 0.2,
                "end_trim": 1.8,
                "scene_rationale": "High visual kinetic energy and vibrant framing perfectly match the adventure theme.",
                "transition_intent": "wipe",
                "engagement_score": 9.4,
            }
        )
        mock_client, mock_file = self._create_mock_client(valid_response_json)

        entry = analyze_clip(
            clip_path=str(sample_video_clip),
            theme="Mountain biking",
            clip_id="vid_sample_123",
            client=mock_client,
        )

        assert isinstance(entry, EDLEntry)
        assert entry.file_reference == "vid_sample_123"
        assert entry.start_trim == 0.2
        assert entry.end_trim == 1.8
        assert entry.duration == 1.6
        assert entry.transition_intent == TransitionIntent.WIPE
        assert entry.engagement_score == 9.4
        assert "kinetic energy" in entry.scene_rationale

        # Verify Files API upload & deletion lifecycle
        mock_client.files.upload.assert_called_once()
        mock_client.models.generate_content.assert_called_once()
        mock_client.files.delete.assert_called_once_with(name="files/test_clip_upload_987")

    def test_guaranteed_file_deletion_on_generation_failure(self, sample_video_clip):
        mock_client, mock_file = self._create_mock_client("{}")
        mock_client.models.generate_content.side_effect = RuntimeError("Gemini API quota exceeded")

        with pytest.raises(RuntimeError, match="Gemini video analysis failed"):
            analyze_clip(
                clip_path=str(sample_video_clip),
                theme="Hiking",
                client=mock_client,
            )

        # Guaranteed cleanup in finally block
        mock_client.files.delete.assert_called_once_with(name="files/test_clip_upload_987")

    def test_file_polling_processing_to_active(self, sample_video_clip):
        mock_client = MagicMock()

        mock_file_processing = MagicMock()
        mock_file_processing.name = "files/test_poll"
        mock_file_processing.state = "PROCESSING"

        mock_file_active = MagicMock()
        mock_file_active.name = "files/test_poll"
        mock_file_active.state = "ACTIVE"

        mock_client.files.upload.return_value = mock_file_processing
        mock_client.files.get.return_value = mock_file_active

        valid_response_json = json.dumps(
            {
                "file_reference": "vid_polled",
                "start_trim": 0.0,
                "end_trim": 1.5,
                "scene_rationale": "Smooth pan across the horizon.",
                "transition_intent": "fade",
                "engagement_score": 8.0,
            }
        )
        mock_response = MagicMock()
        mock_response.text = valid_response_json
        mock_client.models.generate_content.return_value = mock_response

        with patch("time.sleep", return_value=None):
            entry = analyze_clip(
                clip_path=str(sample_video_clip),
                theme="Sunrise",
                clip_id="vid_polled",
                client=mock_client,
            )

        assert entry.transition_intent == TransitionIntent.FADE
        assert mock_client.files.get.called
        mock_client.files.delete.assert_called_once_with(name="files/test_poll")

    def test_file_processing_failed_state_raises_runtime_error(self):
        mock_client = MagicMock()
        mock_file_failed = MagicMock()
        mock_file_failed.name = "files/corrupted"
        mock_file_failed.state = "FAILED"
        mock_file_failed.error = "Corrupted MP4 container"

        with pytest.raises(RuntimeError, match="Gemini Files API failed to process video"):
            poll_file_active(mock_client, mock_file_failed)

    def test_trim_clamping_to_physical_duration(self, sample_video_clip):
        # Clip duration is 2.0s, but model returned 5.0s end_trim
        overshot_json = json.dumps(
            {
                "file_reference": "vid_overshot",
                "start_trim": 0.5,
                "end_trim": 5.0,
                "scene_rationale": "Extensive segment selected by model.",
                "transition_intent": "cut",
                "engagement_score": 7.5,
            }
        )
        mock_client, _ = self._create_mock_client(overshot_json)

        entry = analyze_clip(
            clip_path=str(sample_video_clip),
            theme="Speed run",
            clip_id="vid_overshot",
            client=mock_client,
        )

        assert entry.start_trim == 0.5
        assert entry.end_trim <= 2.0
        assert entry.duration <= 1.5


class TestMainCLIIntegration:
    """Test suite for CLI --analyze-clip integration."""

    def test_cli_analyze_clip_offline(self, sample_video_clip, capsys):
        test_args = [
            "main.py",
            "--analyze-clip",
            str(sample_video_clip),
            "--theme",
            "Pacific coast drive",
            "--clip-id",
            "vid_cli_001",
            "--offline",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        out = capsys.readouterr().out
        assert "VideoGuru: Analyzing Video Clip via Gemini 2.0 Flash (SPEC-008)" in out
        assert "File Reference:   vid_cli_001" in out
        assert "Pacific coast drive" in out
        assert "Engagement Score: 8.5/10.0" in out
        assert "Start Trim:" in out
        assert "End Trim:" in out

    def test_cli_analyze_clip_missing_file_exits_with_error(self, tmp_path, capsys):
        missing = tmp_path / "not_found.mp4"
        test_args = [
            "main.py",
            "--analyze-clip",
            str(missing),
            "--offline",
        ]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        err = capsys.readouterr().err
        assert "Clip analysis failed:" in err
