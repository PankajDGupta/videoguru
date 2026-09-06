"""Unit tests for SPEC-019: Whisper Captioning Tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from config import settings
from tools.whisper_captioning import (
    burn_subtitles_to_video,
    format_timestamp_srt,
    generate_captions,
    transcribe_audio_whisper,
    write_srt_file,
)


class TestFormatTimestampSrt:
    """Validate timestamp formatting into SubRip (SRT) timecode format."""

    def test_zero_seconds(self):
        assert format_timestamp_srt(0.0) == "00:00:00,000"

    def test_negative_seconds_clamped_to_zero(self):
        assert format_timestamp_srt(-5.5) == "00:00:00,000"

    def test_sub_second_values(self):
        assert format_timestamp_srt(0.456) == "00:00:00,456"
        assert format_timestamp_srt(0.005) == "00:00:00,005"
        assert format_timestamp_srt(0.999) == "00:00:00,999"

    def test_multi_minute_values(self):
        # 83.456s = 1m 23s 456ms
        assert format_timestamp_srt(83.456) == "00:01:23,456"
        # 600.0s = 10m 00s 000ms
        assert format_timestamp_srt(600.0) == "00:10:00,000"
        # 3599.999s = 59m 59s 999ms
        assert format_timestamp_srt(3599.999) == "00:59:59,999"

    def test_multi_hour_values(self):
        # 3600.0s = 1h 00m 00s 000ms
        assert format_timestamp_srt(3600.0) == "01:00:00,000"
        # 3661.123s = 1h 01m 01s 123ms
        assert format_timestamp_srt(3661.123) == "01:01:01,123"
        # 7325.789s = 2h 02m 05s 789ms
        assert format_timestamp_srt(7325.789) == "02:02:05,789"
        # 36000.0s = 10h 00m 00s 000ms
        assert format_timestamp_srt(36000.0) == "10:00:00,000"


class TestWriteSrtFile:
    """Validate writing subtitle segments to a standard .srt file."""

    def test_write_srt_standard_formatting(self, tmp_path: Path):
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.5,
                "text": "Hello world.",
            },
            {
                "id": 1,
                "start": 3.0,
                "end": 6.123,
                "text": "Welcome to VideoGuru!",
            },
        ]
        out_file = tmp_path / "subtitles.srt"
        result_path = write_srt_file(segments, out_file)

        assert result_path == out_file
        assert out_file.exists()

        content = out_file.read_text(encoding="utf-8")
        expected = (
            "1\n"
            "00:00:00,000 --> 00:00:02,500\n"
            "Hello world.\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:06,123\n"
            "Welcome to VideoGuru!\n"
        )
        assert content == expected

    def test_write_srt_creates_parent_directories(self, tmp_path: Path):
        nested_out = tmp_path / "deep" / "nested" / "dir" / "test.srt"
        segments = [{"start": 1.0, "end": 2.0, "text": "Testing creation"}]
        result_path = write_srt_file(segments, nested_out)

        assert result_path.exists()
        assert "Testing creation" in result_path.read_text(encoding="utf-8")

    def test_write_srt_skips_empty_text_segments(self, tmp_path: Path):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "  "},
            {"start": 1.0, "end": 2.0, "text": "Valid line"},
            {"start": 2.0, "end": 3.0, "text": ""},
        ]
        out_file = tmp_path / "filtered.srt"
        write_srt_file(segments, out_file)

        content = out_file.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        # Should only have one block, numbered 1
        assert lines[0] == "1"
        assert "Valid line" in content
        assert "2\n" not in content

    def test_write_srt_empty_segments_list(self, tmp_path: Path):
        out_file = tmp_path / "empty.srt"
        result_path = write_srt_file([], out_file)
        assert result_path.exists()
        assert result_path.read_text(encoding="utf-8") == ""

    def test_write_srt_empty_output_path_raises(self):
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            write_srt_file([], "")

    def test_write_srt_handles_missing_or_invalid_timestamps(self, tmp_path: Path):
        segments = [
            {"text": "No start or end"},
            {"start": "invalid", "end": "invalid", "text": "Invalid numbers"},
            {"start": 5.0, "end": 2.0, "text": "End before start"},
        ]
        out_file = tmp_path / "timestamps.srt"
        write_srt_file(segments, out_file)

        content = out_file.read_text(encoding="utf-8")
        assert "1\n00:00:00,000 --> 00:00:00,000\nNo start or end" in content
        assert "2\n00:00:00,000 --> 00:00:00,000\nInvalid numbers" in content
        assert "3\n00:00:05,000 --> 00:00:05,000\nEnd before start" in content


class TestTranscribeAudioWhisper:
    """Validate Whisper transcription with mocked model and offline mode."""

    def test_transcribe_empty_path_raises_value_error(self):
        with pytest.raises(ValueError, match="media_path cannot be empty"):
            transcribe_audio_whisper("")

    def test_transcribe_nonexistent_file_raises_not_found(self, tmp_path: Path):
        nonexistent = tmp_path / "missing_video.mp4"
        with pytest.raises(FileNotFoundError, match="Media file does not exist"):
            transcribe_audio_whisper(nonexistent, offline=False)

    def test_transcribe_offline_mode_returns_mock_data(self, tmp_path: Path):
        dummy_media = tmp_path / "offline_clip.mp4"
        result = transcribe_audio_whisper(dummy_media, offline=True)

        assert "text" in result
        assert "segments" in result
        assert "language" in result
        assert result["language"] == "en"
        assert len(result["segments"]) > 0
        assert "offline_clip" in result["text"]

    def test_transcribe_with_env_var_offline(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("VIDEOGURU_OFFLINE", "1")
        dummy_media = tmp_path / "clip.mp4"
        result = transcribe_audio_whisper(dummy_media)
        assert result["language"] == "en"
        assert len(result["segments"]) == 2

    @patch("whisper.load_model")
    def test_transcribe_with_mocked_whisper_success(self, mock_load_model, tmp_path: Path):
        video_file = tmp_path / "sample.mp4"
        video_file.write_bytes(b"mock video data")

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Transcribed text from whisper.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "Transcribed text"},
                {"id": 1, "start": 2.0, "end": 4.0, "text": "from whisper."},
            ],
            "language": "es",
        }
        mock_load_model.return_value = mock_model

        result = transcribe_audio_whisper(video_file, model_name="small")

        mock_load_model.assert_called_once_with("small")
        mock_model.transcribe.assert_called_once_with(str(video_file.resolve()))
        assert result["language"] == "es"
        assert result["text"] == "Transcribed text from whisper."
        assert len(result["segments"]) == 2

    @patch("whisper.load_model")
    def test_transcribe_whisper_error_raises_runtime_error(self, mock_load_model, tmp_path: Path):
        video_file = tmp_path / "sample.mp4"
        video_file.write_bytes(b"mock video data")

        mock_load_model.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(RuntimeError, match="Whisper transcription failed"):
            transcribe_audio_whisper(video_file, offline=False)


class TestBurnSubtitlesToVideo:
    """Validate FFmpeg subtitle burn-in execution and argument verification."""

    def test_burn_empty_paths_raise_value_error(self):
        with pytest.raises(ValueError, match="video_path cannot be empty"):
            burn_subtitles_to_video("", "sub.srt", "out.mp4")
        with pytest.raises(ValueError, match="srt_path cannot be empty"):
            burn_subtitles_to_video("vid.mp4", "", "out.mp4")
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            burn_subtitles_to_video("vid.mp4", "sub.srt", "")

    def test_burn_nonexistent_files_raise_file_not_found(self, tmp_path: Path):
        vid = tmp_path / "exists.mp4"
        vid.write_bytes(b"data")
        srt = tmp_path / "missing.srt"

        with pytest.raises(FileNotFoundError, match="Subtitle file not found"):
            burn_subtitles_to_video(vid, srt, tmp_path / "out.mp4")

        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        missing_vid = tmp_path / "missing.mp4"
        with pytest.raises(FileNotFoundError, match="Source video not found"):
            burn_subtitles_to_video(missing_vid, srt, tmp_path / "out.mp4")

    @patch("tools.whisper_captioning.execute_ffmpeg_command")
    @patch("tools.whisper_captioning.build_caption_burn_command")
    def test_burn_subtitles_executes_ffmpeg(
        self,
        mock_build_cmd,
        mock_execute,
        tmp_path: Path,
    ):
        vid = tmp_path / "in.mp4"
        vid.write_bytes(b"video")
        srt = tmp_path / "sub.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\ntest\n")
        out = tmp_path / "rendered" / "out.mp4"

        mock_build_cmd.return_value = ["ffmpeg", "-i", str(vid), "-vf", "subtitles=...", str(out)]

        result_path = burn_subtitles_to_video(vid, srt, out, font_size=20)

        assert result_path == str(out.resolve())
        mock_build_cmd.assert_called_once_with(
            video_path=vid.resolve(),
            srt_path=srt.resolve(),
            output_path=out.resolve(),
            font_size=20,
        )
        mock_execute.assert_called_once()


class TestGenerateCaptionsTool:
    """Validate ADK Tool generate_captions orchestration and session state updates."""

    def test_generate_captions_empty_video_raises_value_error(self):
        with pytest.raises(ValueError, match="video_path cannot be empty"):
            generate_captions("")

    def test_generate_captions_nonexistent_video_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            generate_captions(tmp_path / "missing.mp4", offline=False)

    @patch("tools.whisper_captioning.burn_subtitles_to_video")
    @patch("tools.whisper_captioning.write_srt_file")
    @patch("tools.whisper_captioning.transcribe_audio_whisper")
    def test_generate_captions_orchestration_with_tool_context(
        self,
        mock_transcribe,
        mock_write_srt,
        mock_burn,
        tmp_path: Path,
    ):
        vid_path = tmp_path / "vacation.mp4"
        vid_path.write_bytes(b"video content")

        mock_transcribe.return_value = {
            "text": "Transcribed dialogue.",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Transcribed dialogue."}],
            "language": "en",
        }

        # Mock ToolContext with a state dict
        mock_tool_context = MagicMock()
        mock_tool_context.state = {}

        custom_srt = tmp_path / "custom.srt"
        custom_out = tmp_path / "custom_out.mp4"

        result = generate_captions(
            video_path=vid_path,
            output_srt_path=custom_srt,
            output_video_path=custom_out,
            model_name="base",
            tool_context=mock_tool_context,
            offline=True,
            font_size=18,
        )

        mock_transcribe.assert_called_once_with(
            media_path=vid_path.resolve(),
            model_name="base",
            offline=True,
        )
        mock_write_srt.assert_called_once_with(
            segments=[{"start": 0.0, "end": 2.0, "text": "Transcribed dialogue."}],
            output_path=custom_srt.resolve(),
        )
        mock_burn.assert_called_once_with(
            video_path=vid_path.resolve(),
            srt_path=custom_srt.resolve(),
            output_path=custom_out.resolve(),
            font_size=18,
        )

        assert result["srt_path"] == str(custom_srt.resolve())
        assert result["captioned_video_path"] == str(custom_out.resolve())

        # Verify session state was populated
        assert mock_tool_context.state["srt_path"] == str(custom_srt.resolve())
        assert mock_tool_context.state["captioned_video_path"] == str(custom_out.resolve())

    @patch("tools.whisper_captioning.burn_subtitles_to_video")
    @patch("tools.whisper_captioning.write_srt_file")
    @patch("tools.whisper_captioning.transcribe_audio_whisper")
    def test_generate_captions_default_paths(
        self,
        mock_transcribe,
        mock_write_srt,
        mock_burn,
        tmp_path: Path,
    ):
        vid_path = tmp_path / "family_trip.mp4"
        vid_path.write_bytes(b"content")

        mock_transcribe.return_value = {"segments": [], "text": "", "language": "en"}

        result = generate_captions(
            video_path=vid_path,
            offline=True,
        )

        expected_srt = str(settings.STAGING_DIR / "family_trip.srt")
        expected_video = str(settings.OUTPUT_DIR / "family_trip_captioned.mp4")

        assert result["srt_path"] == expected_srt
        assert result["captioned_video_path"] == expected_video

    @patch("tools.whisper_captioning.execute_ffmpeg_command")
    def test_generate_captions_end_to_end_offline(
        self,
        mock_ffmpeg,
        tmp_path: Path,
    ):
        vid_path = tmp_path / "vlog.mp4"
        vid_path.write_bytes(b"mock video bytes")

        srt_path = tmp_path / "vlog.srt"
        out_video = tmp_path / "vlog_captioned.mp4"

        mock_tool_context = MagicMock()
        mock_tool_context.state = {}

        result = generate_captions(
            video_path=vid_path,
            output_srt_path=srt_path,
            output_video_path=out_video,
            tool_context=mock_tool_context,
            offline=True,
        )

        assert srt_path.exists()
        srt_content = srt_path.read_text(encoding="utf-8")
        assert "Welcome to VideoGuru" in srt_content
        assert "00:00:00,000 --> 00:00:03,000" in srt_content

        mock_ffmpeg.assert_called_once()
        assert mock_tool_context.state["srt_path"] == str(srt_path.resolve())
        assert mock_tool_context.state["captioned_video_path"] == str(out_video.resolve())
        assert result["srt_path"] == str(srt_path.resolve())
        assert result["captioned_video_path"] == str(out_video.resolve())
