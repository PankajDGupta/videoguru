"""Unit tests for SPEC-018: Audio Ducking Tool."""

from __future__ import annotations

import logging
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from config import settings
from tools.audio_ducking import apply_audio_ducking


class MockToolContext:
    """Mock ADK ToolContext for testing state access."""

    def __init__(self, state: dict | None = None):
        self.state = state if state is not None else {}


class TestApplyAudioDucking:
    """Unit tests for apply_audio_ducking ADK tool."""

    @pytest.fixture
    def dummy_video(self, tmp_path: Path) -> Path:
        """Create a temporary dummy video file."""
        video_file = tmp_path / "rendered_transition.mp4"
        video_file.write_text("dummy video content")
        return video_file

    @pytest.fixture
    def dummy_music(self, tmp_path: Path) -> Path:
        """Create a temporary dummy music file."""
        music_file = tmp_path / "background.mp3"
        music_file.write_text("dummy music content")
        return music_file

    @patch("tools.audio_ducking.execute_ffmpeg_command")
    def test_apply_ducking_explicit_paths(
        self,
        mock_exec: MagicMock,
        dummy_video: Path,
        dummy_music: Path,
        tmp_path: Path,
    ):
        """Test apply_audio_ducking with explicit video_path, music_path, and output_path."""
        out_file = tmp_path / "custom_ducked.mp4"

        result = apply_audio_ducking(
            video_path=str(dummy_video),
            music_path=str(dummy_music),
            output_path=str(out_file),
        )

        assert result == str(out_file.resolve())
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]

        # Verify command contains inputs and sidechaincompress filter
        assert "-i" in cmd
        assert str(dummy_video.resolve()) in cmd
        assert str(dummy_music.resolve()) in cmd
        assert str(out_file.resolve()) in cmd

        # Check filter_complex content
        filter_idx = cmd.index("-filter_complex") + 1
        filter_str = cmd[filter_idx]
        assert "sidechaincompress" in filter_str
        assert "volume=0.3" in filter_str
        assert "threshold=0.05" in filter_str
        assert "ratio=4.0" in filter_str

    @patch("tools.audio_ducking.execute_ffmpeg_command")
    def test_apply_ducking_default_output_path(
        self,
        mock_exec: MagicMock,
        dummy_video: Path,
        dummy_music: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test generating default output_path in settings.STAGING_DIR when output_path is None."""
        staging_dir = tmp_path / "custom_staging"
        monkeypatch.setattr(settings, "STAGING_DIR", staging_dir)

        result = apply_audio_ducking(
            video_path=dummy_video,
            music_path=dummy_music,
            output_path=None,
        )

        assert staging_dir.exists()
        assert Path(result).parent == staging_dir.resolve()
        assert Path(result).name.startswith("ducked_video_")
        assert Path(result).name.endswith(".mp4")
        mock_exec.assert_called_once()

    @patch("tools.audio_ducking.execute_ffmpeg_command")
    def test_retrieve_paths_from_tool_context_transition_rendered(
        self,
        mock_exec: MagicMock,
        dummy_video: Path,
        dummy_music: Path,
        tmp_path: Path,
    ):
        """Test retrieving video_path and music_path from tool_context.state."""
        context = MockToolContext(
            state={
                "transition_rendered_path": str(dummy_video),
                "music_path": str(dummy_music),
            }
        )
        out_file = tmp_path / "output_from_context.mp4"

        result = apply_audio_ducking(
            output_path=str(out_file),
            tool_context=context,
        )

        assert result == str(out_file.resolve())
        assert context.state.get("ducked_video_path") == str(out_file.resolve())
        mock_exec.assert_called_once()

    @patch("tools.audio_ducking.execute_ffmpeg_command")
    def test_retrieve_paths_from_tool_context_fallback_video_path(
        self,
        mock_exec: MagicMock,
        dummy_video: Path,
        dummy_music: Path,
        tmp_path: Path,
    ):
        """Test retrieving video_path from state['video_path'] when transition_rendered_path is absent."""
        context = MockToolContext(
            state={
                "video_path": str(dummy_video),
                "music_path": str(dummy_music),
            }
        )
        out_file = tmp_path / "output_fallback.mp4"

        result = apply_audio_ducking(
            output_path=str(out_file),
            tool_context=context,
        )

        assert result == str(out_file.resolve())
        assert context.state.get("ducked_video_path") == str(out_file.resolve())

    @patch("tools.audio_ducking.execute_ffmpeg_command")
    def test_dict_tool_context_support(
        self,
        mock_exec: MagicMock,
        dummy_video: Path,
        dummy_music: Path,
        tmp_path: Path,
    ):
        """Test passing a plain dict as tool_context."""
        context_dict = {
            "transition_rendered_path": str(dummy_video),
            "music_path": str(dummy_music),
        }
        out_file = tmp_path / "output_dict.mp4"

        result = apply_audio_ducking(
            output_path=str(out_file),
            tool_context=context_dict,
        )

        assert result == str(out_file.resolve())
        assert context_dict.get("ducked_video_path") == str(out_file.resolve())

    @patch("tools.audio_ducking.execute_ffmpeg_command")
    def test_auto_discover_music_in_media_dir(
        self,
        mock_exec: MagicMock,
        dummy_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test auto-discovering music.mp3 in settings.MEDIA_INPUT_DIR."""
        media_dir = tmp_path / "media_store"
        media_dir.mkdir(parents=True)
        auto_music = media_dir / "music.mp3"
        auto_music.write_text("auto music content")
        monkeypatch.setattr(settings, "MEDIA_INPUT_DIR", media_dir)

        out_file = tmp_path / "out_auto.mp4"
        result = apply_audio_ducking(
            video_path=dummy_video,
            music_path=None,
            output_path=out_file,
        )

        assert result == str(out_file.resolve())
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]
        assert str(auto_music.resolve()) in cmd

    def test_behavior_when_no_music_file_exists(
        self,
        dummy_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that when no music file exists, original video_path is returned and warning logged."""
        empty_media_dir = tmp_path / "empty_media"
        empty_media_dir.mkdir(parents=True)
        monkeypatch.setattr(settings, "MEDIA_INPUT_DIR", empty_media_dir)

        context = MockToolContext(state={})

        with caplog.at_level(logging.WARNING):
            result = apply_audio_ducking(
                video_path=dummy_video,
                music_path=None,
                tool_context=context,
            )

        assert result == str(dummy_video.resolve())
        assert context.state.get("ducked_video_path") == str(dummy_video.resolve())
        assert any("No background music file provided" in record.message for record in caplog.records)

    @patch("tools.audio_ducking.execute_ffmpeg_command")
    def test_parameter_customization(
        self,
        mock_exec: MagicMock,
        dummy_video: Path,
        dummy_music: Path,
        tmp_path: Path,
    ):
        """Test parameter customization: threshold, ratio, attack, release, music_volume."""
        out_file = tmp_path / "custom_params.mp4"

        apply_audio_ducking(
            video_path=dummy_video,
            music_path=dummy_music,
            output_path=out_file,
            threshold=0.12,
            ratio=8.0,
            attack=15.0,
            release=350.0,
            music_volume=0.45,
        )

        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]
        filter_idx = cmd.index("-filter_complex") + 1
        filter_str = cmd[filter_idx]

        assert "threshold=0.12" in filter_str
        assert "ratio=8.0" in filter_str
        assert "attack=15.0" in filter_str
        assert "release=350.0" in filter_str
        assert "volume=0.45" in filter_str

    def test_missing_video_path_raises_value_error(self):
        """Test raising ValueError when video_path is None and not in state."""
        with pytest.raises(ValueError, match="video_path must be provided"):
            apply_audio_ducking(video_path=None, tool_context=None)

        context = MockToolContext(state={})
        with pytest.raises(ValueError, match="video_path must be provided"):
            apply_audio_ducking(video_path=None, tool_context=context)

    def test_empty_video_path_string_raises_value_error(self):
        """Test raising ValueError when video_path is empty string."""
        with pytest.raises(ValueError, match="video_path cannot be empty"):
            apply_audio_ducking(video_path="   ")

    def test_nonexistent_video_path_raises_file_not_found(self, tmp_path: Path):
        """Test raising FileNotFoundError when video_path does not exist on disk."""
        missing_video = tmp_path / "nonexistent_video.mp4"
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            apply_audio_ducking(video_path=missing_video)

    def test_explicit_nonexistent_music_raises_file_not_found(
        self,
        dummy_video: Path,
        tmp_path: Path,
    ):
        """Test raising FileNotFoundError when explicit music_path does not exist on disk."""
        missing_music = tmp_path / "missing_track.mp3"
        with pytest.raises(FileNotFoundError, match="Specified music file not found"):
            apply_audio_ducking(
                video_path=dummy_video,
                music_path=missing_music,
            )

    def test_explicit_music_in_state_not_found_raises_file_not_found(
        self,
        dummy_video: Path,
        tmp_path: Path,
    ):
        """Test raising FileNotFoundError when explicit music_path in state does not exist."""
        missing_music = tmp_path / "missing_state_music.mp3"
        context = MockToolContext(state={"music_path": str(missing_music)})

        with pytest.raises(FileNotFoundError, match="Specified music file not found"):
            apply_audio_ducking(
                video_path=dummy_video,
                tool_context=context,
            )

    def test_empty_music_path_string_raises_value_error(self, dummy_video: Path):
        """Test raising ValueError when music_path is empty string."""
        with pytest.raises(ValueError, match="music_path cannot be empty"):
            apply_audio_ducking(video_path=dummy_video, music_path="   ")

    def test_empty_output_path_string_raises_value_error(
        self,
        dummy_video: Path,
        dummy_music: Path,
    ):
        """Test raising ValueError when output_path is empty string."""
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            apply_audio_ducking(
                video_path=dummy_video,
                music_path=dummy_music,
                output_path="   ",
            )

    def test_invalid_parameters_raise_value_error(
        self,
        dummy_video: Path,
        dummy_music: Path,
        tmp_path: Path,
    ):
        """Test parameter validation errors propagated from build_ducking_command."""
        out = tmp_path / "out.mp4"

        # threshold <= 0 or > 1.0
        with pytest.raises(ValueError, match="threshold must be in range"):
            apply_audio_ducking(dummy_video, dummy_music, out, threshold=0.0)
        with pytest.raises(ValueError, match="threshold must be in range"):
            apply_audio_ducking(dummy_video, dummy_music, out, threshold=1.5)

        # ratio < 1.0
        with pytest.raises(ValueError, match="ratio must be >= 1.0"):
            apply_audio_ducking(dummy_video, dummy_music, out, ratio=0.5)

        # attack <= 0
        with pytest.raises(ValueError, match="attack must be positive"):
            apply_audio_ducking(dummy_video, dummy_music, out, attack=-1.0)

        # release <= 0
        with pytest.raises(ValueError, match="release must be positive"):
            apply_audio_ducking(dummy_video, dummy_music, out, release=0.0)

        # music_volume < 0
        with pytest.raises(ValueError, match="music_volume must be non-negative"):
            apply_audio_ducking(dummy_video, dummy_music, out, music_volume=-0.2)

    @patch("tools.audio_ducking.execute_ffmpeg_command")
    def test_ffmpeg_execution_failure_propagates(
        self,
        mock_exec: MagicMock,
        dummy_video: Path,
        dummy_music: Path,
        tmp_path: Path,
    ):
        """Test that RuntimeError from execute_ffmpeg_command is propagated."""
        mock_exec.side_effect = RuntimeError("FFmpeg command failed with code 1")
        out = tmp_path / "out_failed.mp4"

        with pytest.raises(RuntimeError, match="FFmpeg command failed"):
            apply_audio_ducking(
                video_path=dummy_video,
                music_path=dummy_music,
                output_path=out,
            )
