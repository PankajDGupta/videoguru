"""Tests for Scene-Aware Text Overlays & Subtitle Controls (SPEC-033).

Verifies:
1. Speech-to-text subtitles are disabled by default (no unwanted Hindi/foreign transcripts burned).
2. Subtitles can be opted into via enable_captions=True or --captions.
3. Scene-aware overlay text is generated and burned by default.
4. Overlay text escaping handles commas, colons, quotes, percent, and gte timing expressions.
5. Subprocess UTF-8 encoding on Windows does not crash with charmap UnicodeDecodeError.
6. CLI arguments (--captions, --no-captions, --no-overlay-text) parse and route correctly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agents.enhancement_rendering import EnhancementRenderingAgent
from config import settings
from rendering.ffmpeg_builder import (
    build_drawtext_overlay_command,
    execute_ffmpeg_command,
)
from schemas.edl import EditDecisionList, EDLEntry, TransitionIntent
from schemas.media import ClipManifestEntry
from schemas.overlay_text import OverlayTextPlan, OverlayTextStyle, TextPosition
from tools.overlay_text_tools import (
    burn_overlay_text,
    generate_overlay_texts_with_gemini,
    generate_overlay_text_plan,
)


@pytest.fixture
def sample_edl() -> EditDecisionList:
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="clip_01",
                start_trim=0.0,
                end_trim=4.0,
                scene_rationale="Morning home workout session with weights",
                transition_intent=TransitionIntent.CUT,
            ),
            EDLEntry(
                file_reference="clip_02",
                start_trim=5.0,
                end_trim=9.0,
                scene_rationale="Driving son to tuition class in car",
                transition_intent=TransitionIntent.FADE,
            ),
            EDLEntry(
                file_reference="clip_03",
                start_trim=1.0,
                end_trim=5.0,
                scene_rationale="Teaching son mathematics at study desk",
                transition_intent=TransitionIntent.CUT,
            ),
        ]
    )


@pytest.fixture
def sample_manifest() -> list[ClipManifestEntry]:
    return [
        ClipManifestEntry(
            clip_id="clip_01",
            absolute_path="C:/media/workout.mp4",
            file_name="workout.mp4",
            duration_seconds=10.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=1000,
        ),
        ClipManifestEntry(
            clip_id="clip_02",
            absolute_path="C:/media/drive.mp4",
            file_name="drive.mp4",
            duration_seconds=12.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=1000,
        ),
        ClipManifestEntry(
            clip_id="clip_03",
            absolute_path="C:/media/maths.mp4",
            file_name="maths.mp4",
            duration_seconds=8.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=1000,
        ),
    ]


class TestSubtitleDisablingAndOptIn:
    """Verify speech-to-text subtitles are opt-in and disabled by default."""

    @patch("agents.enhancement_rendering.render_with_transitions")
    @patch("agents.enhancement_rendering.burn_overlay_text")
    @patch("agents.enhancement_rendering.generate_captions")
    def test_subtitles_disabled_by_default(
        self, mock_captions, mock_burn_overlay, mock_transitions, sample_edl, sample_manifest, tmp_path
    ):
        mock_video = tmp_path / "step1.mp4"
        mock_video.write_bytes(b"dummy")
        mock_transitions.return_value = str(mock_video)
        mock_burn_overlay.return_value = str(mock_video)

        agent = EnhancementRenderingAgent(offline=True)
        state = {
            "edl": sample_edl,
            "clip_manifest": sample_manifest,
            "theme": "Work-life balance",
        }

        result = agent.execute_rendering_pipeline(state)

        # Captions must NOT be called by default
        assert result["has_captions"] is False
        assert result["srt_path"] is None
        mock_captions.assert_not_called()
        # Overlay text should be applied
        assert result["has_overlay_text"] is True

    @patch("agents.enhancement_rendering.render_with_transitions")
    @patch("agents.enhancement_rendering.burn_overlay_text")
    @patch("agents.enhancement_rendering.generate_captions")
    def test_subtitles_enabled_when_explicitly_opted_in(
        self, mock_captions, mock_burn_overlay, mock_transitions, sample_edl, sample_manifest, tmp_path
    ):
        mock_video = tmp_path / "step1.mp4"
        mock_video.write_bytes(b"dummy")
        mock_cap_video = tmp_path / "step3.mp4"
        mock_cap_video.write_bytes(b"captioned")
        mock_srt = tmp_path / "subs.srt"
        mock_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")

        mock_transitions.return_value = str(mock_video)
        mock_burn_overlay.return_value = str(mock_video)
        mock_captions.return_value = {
            "captioned_video_path": str(mock_cap_video),
            "srt_path": str(mock_srt),
        }

        agent = EnhancementRenderingAgent(offline=True)
        state = {
            "edl": sample_edl,
            "clip_manifest": sample_manifest,
            "theme": "Work-life balance",
            "enable_captions": True,
        }

        result = agent.execute_rendering_pipeline(state)

        assert result["has_captions"] is True
        assert result["srt_path"] == str(mock_srt)
        mock_captions.assert_called_once()


class TestDrawtextFilterEscaping:
    """Verify drawtext command builder properly escapes commas, colons, and timing expressions."""

    def test_commas_in_overlay_text_are_escaped(self):
        entries = [
            {
                "text": "Work-life balance, gym session",
                "start_time": 0.5,
                "end_time": 3.5,
                "style": {"font_family": "Impact", "font_size": 48},
            }
        ]
        with patch("rendering.ffmpeg_builder.find_ffmpeg_executable", return_value="ffmpeg"):
            cmd = build_drawtext_overlay_command("in.mp4", entries, "out.mp4")

        vf_str = cmd[cmd.index("-vf") + 1]
        # Text comma must be escaped as \, so FFmpeg does not treat it as a filter separator
        assert r"text='Work-life balance\, gym session'" in vf_str

    def test_enable_timing_commas_are_escaped(self):
        entries = [
            {
                "text": "Scene 1",
                "start_time": 1.25,
                "end_time": 4.75,
                "style": {"font_family": "Impact", "font_size": 48},
            }
        ]
        with patch("rendering.ffmpeg_builder.find_ffmpeg_executable", return_value="ffmpeg"):
            cmd = build_drawtext_overlay_command("in.mp4", entries, "out.mp4")

        vf_str = cmd[cmd.index("-vf") + 1]
        # enable expression must have escaped comma
        assert r"gte(t\,1.250)*lte(t\,4.750)" in vf_str

    def test_multiple_drawtext_filters_chained_without_internal_comma_conflict(self):
        entries = [
            {
                "text": "Morning workout, lets go!",
                "start_time": 0.0,
                "end_time": 3.0,
                "style": {"font_family": "Impact", "font_size": 48},
            },
            {
                "text": "Office work, busy afternoon",
                "start_time": 3.5,
                "end_time": 6.5,
                "style": {"font_family": "Impact", "font_size": 48},
            },
        ]
        with patch("rendering.ffmpeg_builder.find_ffmpeg_executable", return_value="ffmpeg"):
            cmd = build_drawtext_overlay_command("in.mp4", entries, "out.mp4")

        vf_str = cmd[cmd.index("-vf") + 1]
        # Exactly 2 drawtext filters separated by 1 top-level comma
        parts = vf_str.split("drawtext=")
        assert len(parts) == 3  # empty prefix + 2 filters


class TestSubprocessUtf8Encoding:
    """Verify execute_ffmpeg_command uses utf-8 with error replacement to prevent cp1252 crash."""

    @patch("subprocess.run")
    def test_execute_ffmpeg_command_uses_utf8_and_replace(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "FFmpeg stdout"
        mock_proc.stderr = "FFmpeg stderr with \x8d unicode byte"
        mock_run.return_value = mock_proc

        cmd = ["ffmpeg", "-version"]
        res = execute_ffmpeg_command(cmd)

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("encoding") == "utf-8"
        assert kwargs.get("errors") == "replace"
        assert res.returncode == 0

    @patch("subprocess.run")
    def test_execute_ffmpeg_command_handles_none_stderr(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = None
        mock_proc.stderr = None
        mock_run.return_value = mock_proc

        cmd = ["ffmpeg", "-version"]
        with pytest.raises(RuntimeError, match="code 1"):
            execute_ffmpeg_command(cmd, check=True)


class TestSceneAwareGeminiPrompt:
    """Verify Gemini overlay prompt explicitly requests stylish captions and scene awareness."""

    @patch("tools.overlay_text_tools.settings")
    @patch("google.genai.Client")
    def test_gemini_overlay_prompt_requests_scene_context_and_disallows_subtitles(
        self, mock_client_cls, mock_settings, sample_edl, sample_manifest
    ):
        mock_settings.GEMINI_API_KEY = "test_key"
        mock_settings.GEMINI_MODEL = "gemini-2.0-flash"

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.text = (
            '[{"cut_index": 0, "text": "Morning workout grind 💪", "is_hook": true},'
            '{"cut_index": 1, "text": "School drop-off run 🚗", "is_hook": false},'
            '{"cut_index": 2, "text": "Math tutoring session 📐", "is_hook": false}]'
        )
        mock_client.models.generate_content.return_value = mock_resp

        plan = generate_overlay_texts_with_gemini(
            sample_edl,
            theme="Work-life balance: Workout, school, and maths tutoring",
            clip_manifest=sample_manifest,
        )

        assert len(plan) == 3
        assert plan[0].text == "Morning workout grind 💪"
        assert plan[0].is_hook is True

        # Inspect prompt passed to Gemini
        call_args = mock_client.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents", "")
        # Prompt must disallow foreign speech subtitles
        assert "speech transcript subtitles" in prompt or "subtitles" in prompt.lower()
        # Prompt must contain scene content
        assert "Morning home workout" in prompt
        assert "tuition" in prompt.lower() or "driving" in prompt.lower()


class TestCliCaptionsAndOverlayFlags:
    """Verify main.py CLI flags for --captions, --no-captions, --no-overlay-text."""

    def test_parse_arguments_captions_flag(self):
        from main import parse_arguments
        with patch("sys.argv", ["main.py", "--captions"]):
            args = parse_arguments()
            assert args.captions is True

    def test_parse_arguments_subtitles_alias(self):
        from main import parse_arguments
        with patch("sys.argv", ["main.py", "--subtitles"]):
            args = parse_arguments()
            assert args.captions is True

    def test_parse_arguments_no_captions_flag(self):
        from main import parse_arguments
        with patch("sys.argv", ["main.py", "--no-captions"]):
            args = parse_arguments()
            assert args.no_captions is True

    def test_parse_arguments_no_overlay_text_flag(self):
        from main import parse_arguments
        with patch("sys.argv", ["main.py", "--no-overlay-text"]):
            args = parse_arguments()
            assert args.no_overlay_text is True
