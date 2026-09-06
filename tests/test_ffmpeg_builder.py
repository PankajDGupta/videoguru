"""Unit tests for SPEC-016: FFmpeg Command Builder."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from rendering.ffmpeg_builder import (
    TRANSITION_MAP,
    VALID_XFADE_TRANSITIONS,
    build_audio_crossfade,
    build_caption_burn_command,
    build_ducking_command,
    build_trim_command,
    build_xfade_chain,
    calculate_xfade_offsets,
    escape_subtitles_path,
    execute_ffmpeg_command,
    find_ffmpeg_executable,
    map_transition_intent,
)
from schemas.edl import TransitionIntent


class TestFindFFmpegExecutable:
    """Validate locating the ffmpeg binary."""

    def test_find_ffmpeg_returns_valid_path(self):
        bin_path = find_ffmpeg_executable()
        assert bin_path is not None
        assert isinstance(bin_path, str)
        assert "ffmpeg" in bin_path.lower()

    @patch("shutil.which", return_value=None)
    @patch("os.name", "posix")
    def test_find_ffmpeg_not_found_raises_runtime_error(self, mock_which):
        with pytest.raises(RuntimeError, match="FFmpeg executable 'ffmpeg' not found"):
            find_ffmpeg_executable()


class TestMapTransitionIntent:
    """Validate mapping transition intents to FFmpeg xfade transition names."""

    def test_enum_intents_mapping(self):
        assert map_transition_intent(TransitionIntent.FADE) == "fade"
        assert map_transition_intent(TransitionIntent.WIPE) == "wipeleft"
        assert map_transition_intent(TransitionIntent.SLIDE) == "slideleft"
        assert map_transition_intent(TransitionIntent.DISSOLVE) == "dissolve"
        assert map_transition_intent(TransitionIntent.CUT) == "fade"

    def test_string_intents_mapping(self):
        assert map_transition_intent("fade") == "fade"
        assert map_transition_intent("wipe") == "wipeleft"
        assert map_transition_intent("slide") == "slideleft"
        assert map_transition_intent("dissolve") == "dissolve"
        assert map_transition_intent("wiperight") == "wiperight"
        assert map_transition_intent("circlecrop") == "circlecrop"

    def test_unknown_intent_fallback(self):
        assert map_transition_intent("nonexistent_transition") == "fade"

    def test_invalid_types_raise_error(self):
        with pytest.raises(ValueError, match="Transition string cannot be empty"):
            map_transition_intent("")
        with pytest.raises(TypeError, match="Expected str or TransitionIntent"):
            map_transition_intent(123)  # type: ignore


class TestTrimCommand:
    """Validate trimming command construction and validation."""

    def test_build_trim_command_defaults(self):
        cmd = build_trim_command(
            clip_path="input.mp4",
            start=1.5,
            end=6.5,
            output_path="trimmed.mp4",
        )
        assert "ffmpeg" in cmd[0].lower()
        assert "-y" in cmd
        assert "-ss" in cmd
        ss_idx = cmd.index("-ss")
        assert cmd[ss_idx + 1] == "1.500"
        assert "-to" in cmd
        to_idx = cmd.index("-to")
        assert cmd[to_idx + 1] == "6.500"
        assert "-i" in cmd
        i_idx = cmd.index("-i")
        assert cmd[i_idx + 1] == "input.mp4"
        assert "-vf" in cmd
        vf_idx = cmd.index("-vf")
        assert "scale=1920:1080" in cmd[vf_idx + 1]
        assert "fps=30.0" in cmd[vf_idx + 1]
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert cmd[-1] == "trimmed.mp4"

    def test_build_trim_command_custom_res_and_fps(self):
        cmd = build_trim_command(
            clip_path=Path("/tmp/clip.mov"),
            start=0.0,
            end=10.0,
            output_path=Path("/tmp/out.mov"),
            target_resolution="1280x720",
            target_fps=60.0,
        )
        vf_idx = cmd.index("-vf")
        assert "scale=1280:720" in cmd[vf_idx + 1]
        assert "fps=60.0" in cmd[vf_idx + 1]
        assert cmd[-1] == str(Path("/tmp/out.mov"))

    def test_build_trim_command_invalid_times(self):
        with pytest.raises(ValueError, match="start trim must be non-negative"):
            build_trim_command("in.mp4", -1.0, 5.0, "out.mp4")

        with pytest.raises(ValueError, match="end trim must be strictly greater than start trim"):
            build_trim_command("in.mp4", 5.0, 5.0, "out.mp4")

        with pytest.raises(ValueError, match="end trim must be strictly greater than start trim"):
            build_trim_command("in.mp4", 6.0, 5.0, "out.mp4")

    def test_build_trim_command_invalid_fps(self):
        with pytest.raises(ValueError, match="target_fps must be positive"):
            build_trim_command("in.mp4", 0.0, 5.0, "out.mp4", target_fps=0.0)

    def test_build_trim_command_invalid_resolution(self):
        with pytest.raises(ValueError, match="Invalid target_resolution format"):
            build_trim_command("in.mp4", 0.0, 5.0, "out.mp4", target_resolution="1080p")

    def test_build_trim_command_empty_paths(self):
        with pytest.raises(ValueError, match="clip_path cannot be empty"):
            build_trim_command("", 0.0, 5.0, "out.mp4")
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            build_trim_command("in.mp4", 0.0, 5.0, "")


class TestXfadeOffsetsCalculation:
    """Validate cumulative offset calculation for xfade transitions."""

    def test_mathematical_verification_5_5_5_transition_1(self):
        """Verify mathematically: for durations [5, 5, 5] and transition 1.0, offset 1 is 4.0, offset 2 is 8.0."""
        durations = [5.0, 5.0, 5.0]
        transition = 1.0
        offsets = calculate_xfade_offsets(durations, transition_duration=transition)
        assert len(offsets) == 2
        assert offsets[0] == 4.0
        assert offsets[1] == 8.0

    def test_offsets_four_clips(self):
        # Clip 0: 10s, Clip 1: 8s, Clip 2: 6s, Clip 3: 4s with 1.5s transition
        # offset 0 = 10 - 1.5 = 8.5, blended = 10 + 8 - 1.5 = 16.5
        # offset 1 = 16.5 - 1.5 = 15.0, blended = 16.5 + 6 - 1.5 = 21.0
        # offset 2 = 21.0 - 1.5 = 19.5, blended = 21.0 + 4 - 1.5 = 23.5
        durations = [10.0, 8.0, 6.0, 4.0]
        offsets = calculate_xfade_offsets(durations, transition_duration=1.5)
        assert offsets == [8.5, 15.0, 19.5]

    def test_offsets_single_clip_returns_empty(self):
        durations = [10.0]
        offsets = calculate_xfade_offsets(durations, transition_duration=1.0)
        assert offsets == []

    def test_offsets_empty_durations_raises_error(self):
        with pytest.raises(ValueError, match="durations list cannot be empty"):
            calculate_xfade_offsets([])

    def test_offsets_invalid_transition_duration(self):
        with pytest.raises(ValueError, match="transition_duration must be positive"):
            calculate_xfade_offsets([5.0, 5.0], transition_duration=0.0)

    def test_offsets_duration_less_than_transition(self):
        with pytest.raises(ValueError, match="must be strictly greater than transition duration"):
            calculate_xfade_offsets([0.8, 5.0], transition_duration=1.0)

    def test_offsets_negative_duration(self):
        with pytest.raises(ValueError, match="must be positive"):
            calculate_xfade_offsets([-5.0, 5.0], transition_duration=1.0)


class TestAudioCrossfade:
    """Validate FFmpeg audio acrossfade filter string construction."""

    def test_two_clips_audio_crossfade(self):
        clips = ["clip1.mp4", "clip2.mp4"]
        durations = [5.0, 5.0]
        filter_str = build_audio_crossfade(clips, durations, transition_duration=1.0)
        assert filter_str == "[0:a][1:a]acrossfade=d=1.0:c1=tri:c2=tri[a1]"

    def test_three_clips_audio_crossfade(self):
        clips = ["clip1.mp4", "clip2.mp4", "clip3.mp4"]
        durations = [5.0, 5.0, 5.0]
        filter_str = build_audio_crossfade(clips, durations, transition_duration=1.0)
        expected = (
            "[0:a][1:a]acrossfade=d=1.0:c1=tri:c2=tri[a1];"
            "[a1][2:a]acrossfade=d=1.0:c1=tri:c2=tri[a2]"
        )
        assert filter_str == expected

    def test_custom_curves(self):
        clips = ["c1.mp4", "c2.mp4"]
        durations = [4.0, 4.0]
        filter_str = build_audio_crossfade(
            clips, durations, transition_duration=1.5, curve1="qsin", curve2="hsin"
        )
        assert "c1=qsin:c2=hsin" in filter_str
        assert "d=1.5" in filter_str

    def test_mismatched_lengths_raises_error(self):
        with pytest.raises(ValueError, match="does not match number of durations"):
            build_audio_crossfade(["c1.mp4", "c2.mp4"], [5.0])

    def test_fewer_than_two_clips_raises_error(self):
        with pytest.raises(ValueError, match="At least 2 clips are required"):
            build_audio_crossfade(["c1.mp4"], [5.0])
        with pytest.raises(ValueError, match="clips sequence cannot be empty"):
            build_audio_crossfade([], [])

    def test_duration_less_than_transition_raises_error(self):
        with pytest.raises(ValueError, match="must be strictly greater than transition duration"):
            build_audio_crossfade(["c1.mp4", "c2.mp4"], [1.0, 5.0], transition_duration=1.5)


class TestXfadeChain:
    """Validate full xfade chain command construction."""

    def test_two_clips_xfade_chain(self):
        clips = ["a.mp4", "b.mp4"]
        transitions = ["fade"]
        durations = [5.0, 5.0]
        cmd = build_xfade_chain(
            clips=clips,
            transitions=transitions,
            durations=durations,
            output_path="final.mp4",
            transition_duration=1.0,
        )
        assert "-i" in cmd
        assert cmd.count("-i") == 2
        fc_idx = cmd.index("-filter_complex")
        fc = cmd[fc_idx + 1]
        assert "[0:v][1:v]xfade=transition=fade:duration=1.0:offset=4.0[v1]" in fc
        assert "[0:a][1:a]acrossfade=d=1.0:c1=tri:c2=tri[a1]" in fc
        assert cmd[cmd.index("-map") + 1] == "[v1]"
        assert "-map" in cmd[cmd.index("-map") + 2 :]
        assert cmd[-1] == "final.mp4"

    def test_three_clips_mixed_transitions_and_broadcast(self):
        clips = ["c1.mp4", "c2.mp4", "c3.mp4"]
        durations = [6.0, 6.0, 6.0]
        # Test transition intent enum
        transitions = [TransitionIntent.WIPE, TransitionIntent.DISSOLVE]
        cmd = build_xfade_chain(
            clips=clips,
            transitions=transitions,
            durations=durations,
            output_path="out.mp4",
            transition_duration=1.0,
        )
        fc = cmd[cmd.index("-filter_complex") + 1]
        # First transition at 6 - 1 = 5.0
        # Second transition at (6+6-1) - 1 = 10.0
        assert "[0:v][1:v]xfade=transition=wipeleft:duration=1.0:offset=5.0[v1]" in fc
        assert "[v1][2:v]xfade=transition=dissolve:duration=1.0:offset=10.0[v2]" in fc
        assert "[a1][2:a]acrossfade=d=1.0:c1=tri:c2=tri[a2]" in fc

    def test_broadcast_single_transition(self):
        clips = ["c1.mp4", "c2.mp4", "c3.mp4"]
        durations = [5.0, 5.0, 5.0]
        cmd = build_xfade_chain(
            clips=clips,
            transitions="slide",
            durations=durations,
            output_path="out.mp4",
            transition_duration=1.0,
        )
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "transition=slideleft" in fc
        assert fc.count("transition=slideleft") == 2

    def test_transitions_count_mismatch(self):
        clips = ["c1.mp4", "c2.mp4", "c3.mp4"]
        durations = [5.0, 5.0, 5.0]
        with pytest.raises(ValueError, match="Expected 2 transitions"):
            build_xfade_chain(clips, ["fade"], durations, "out.mp4")

    def test_fewer_than_two_clips_raises_error(self):
        with pytest.raises(ValueError, match="At least 2 clips are required"):
            build_xfade_chain(["c1.mp4"], [], [5.0], "out.mp4")
        with pytest.raises(ValueError, match="clips sequence cannot be empty"):
            build_xfade_chain([], [], [], "out.mp4")

    def test_empty_output_raises_error(self):
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            build_xfade_chain(["c1.mp4", "c2.mp4"], ["fade"], [5.0, 5.0], "")


class TestDuckingCommand:
    """Validate audio ducking command construction and parameter bounds."""

    def test_build_ducking_command_default(self):
        cmd = build_ducking_command(
            video_path="input_video.mp4",
            music_path="bg_music.mp3",
            output_path="ducked_output.mp4",
        )
        assert "ffmpeg" in cmd[0].lower()
        assert "-y" in cmd
        assert cmd[cmd.index("-i") + 1] == "input_video.mp4"
        i2_idx = cmd.index("-i", cmd.index("-i") + 2)
        assert cmd[i2_idx + 1] == "bg_music.mp3"

        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "[1:a]volume=0.3[music]" in fc
        assert "sidechaincompress=threshold=0.05:ratio=4.0:attack=20.0:release=250.0[ducked]" in fc
        assert "[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]" in fc

        assert "-map" in cmd
        assert "0:v" in cmd
        assert "[aout]" in cmd
        assert "-c:v" in cmd
        assert "copy" in cmd
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert cmd[-1] == "ducked_output.mp4"

    def test_build_ducking_command_custom_parameters(self):
        cmd = build_ducking_command(
            video_path=Path("vid.mp4"),
            music_path=Path("music.aac"),
            output_path=Path("out.mp4"),
            threshold=0.1,
            ratio=6.0,
            attack=10.0,
            release=500.0,
            music_volume=0.25,
        )
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "volume=0.25" in fc
        assert "threshold=0.1:ratio=6.0:attack=10.0:release=500.0" in fc

    def test_ducking_invalid_parameters(self):
        with pytest.raises(ValueError, match="threshold must be in range"):
            build_ducking_command("v.mp4", "m.mp3", "out.mp4", threshold=0.0)
        with pytest.raises(ValueError, match="threshold must be in range"):
            build_ducking_command("v.mp4", "m.mp3", "out.mp4", threshold=1.5)

        with pytest.raises(ValueError, match="ratio must be >= 1.0"):
            build_ducking_command("v.mp4", "m.mp3", "out.mp4", ratio=0.8)

        with pytest.raises(ValueError, match="attack must be positive"):
            build_ducking_command("v.mp4", "m.mp3", "out.mp4", attack=-5.0)

        with pytest.raises(ValueError, match="release must be positive"):
            build_ducking_command("v.mp4", "m.mp3", "out.mp4", release=0.0)

        with pytest.raises(ValueError, match="music_volume must be non-negative"):
            build_ducking_command("v.mp4", "m.mp3", "out.mp4", music_volume=-0.1)

    def test_ducking_empty_paths(self):
        with pytest.raises(ValueError, match="video_path cannot be empty"):
            build_ducking_command("", "m.mp3", "out.mp4")
        with pytest.raises(ValueError, match="music_path cannot be empty"):
            build_ducking_command("v.mp4", "", "out.mp4")
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            build_ducking_command("v.mp4", "m.mp3", "")


class TestCaptionBurnCommand:
    """Validate caption burn command and path escaping on Windows/POSIX."""

    def test_escape_subtitles_path_windows_drive_and_backslashes(self):
        windows_path = r"C:\Users\Pankaj\Videos\captions.srt"
        escaped = escape_subtitles_path(windows_path)
        assert escaped == r"C\:/Users/Pankaj/Videos/captions.srt"

    def test_escape_subtitles_path_spaces_and_quotes(self):
        path_with_specials = r"D:\My Projects\Video's\sub title.srt"
        escaped = escape_subtitles_path(path_with_specials)
        assert r"D\:/My Projects/Video\'s/sub title.srt" == escaped

    def test_escape_subtitles_path_empty_raises_error(self):
        with pytest.raises(ValueError, match="Subtitle path cannot be empty"):
            escape_subtitles_path("")

    def test_build_caption_burn_command_structure(self):
        cmd = build_caption_burn_command(
            video_path=r"C:\videos\input.mp4",
            srt_path=r"C:\videos\captions.srt",
            output_path=r"C:\videos\output.mp4",
            font_size=18,
        )
        assert "ffmpeg" in cmd[0].lower()
        assert "-y" in cmd
        assert cmd[cmd.index("-i") + 1] == r"C:\videos\input.mp4"
        vf = cmd[cmd.index("-vf") + 1]
        assert r"subtitles='C\:/videos/captions.srt':force_style='FontSize=18'" == vf
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-c:a" in cmd
        assert "copy" in cmd
        assert cmd[-1] == r"C:\videos\output.mp4"

    def test_build_caption_burn_command_invalid_font_size(self):
        with pytest.raises(ValueError, match="font_size must be a positive integer"):
            build_caption_burn_command("in.mp4", "subs.srt", "out.mp4", font_size=0)
        with pytest.raises(ValueError, match="font_size must be a positive integer"):
            build_caption_burn_command("in.mp4", "subs.srt", "out.mp4", font_size=-10)

    def test_build_caption_burn_command_empty_paths(self):
        with pytest.raises(ValueError, match="video_path cannot be empty"):
            build_caption_burn_command("", "subs.srt", "out.mp4")
        with pytest.raises(ValueError, match="output_path cannot be empty"):
            build_caption_burn_command("in.mp4", "subs.srt", "")


class TestExecuteFFmpegCommand:
    """Validate subprocess execution, sandboxing, and error handling."""

    @patch("subprocess.run")
    def test_execute_success(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg", "-version"],
            returncode=0,
            stdout="ffmpeg version 7.0",
            stderr="",
        )
        cmd = ["ffmpeg", "-version"]
        result = execute_ffmpeg_command(cmd, timeout=60.0)
        assert result.returncode == 0
        assert "ffmpeg version" in result.stdout
        mock_run.assert_called_once_with(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60.0,
            check=False,
        )

    @patch("subprocess.run")
    def test_execute_failure_raises_runtime_error(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg", "-i", "missing.mp4"],
            returncode=1,
            stdout="",
            stderr="No such file or directory",
        )
        with pytest.raises(RuntimeError, match="FFmpeg command execution failed.*No such file"):
            execute_ffmpeg_command(["ffmpeg", "-i", "missing.mp4"], check=True)

    @patch("subprocess.run")
    def test_execute_failure_without_check_returns_process(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ffmpeg", "-i", "missing.mp4"],
            returncode=1,
            stdout="",
            stderr="No such file or directory",
        )
        result = execute_ffmpeg_command(["ffmpeg", "-i", "missing.mp4"], check=False)
        assert result.returncode == 1

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10.0))
    def test_execute_timeout_raises_timeout_error(self, mock_run):
        with pytest.raises(TimeoutError, match="FFmpeg command timed out after 10.0 seconds"):
            execute_ffmpeg_command(["ffmpeg", "-i", "vid.mp4"], timeout=10.0)

    @patch("subprocess.run", side_effect=OSError("Exec format error"))
    def test_execute_os_error_raises_runtime_error(self, mock_run):
        with pytest.raises(RuntimeError, match="FFmpeg execution failed: Exec format error"):
            execute_ffmpeg_command(["ffmpeg", "-i", "vid.mp4"])

    def test_execute_empty_cmd_raises_error(self):
        with pytest.raises(ValueError, match="Command cannot be empty"):
            execute_ffmpeg_command([])

    def test_execute_non_string_arg_raises_type_error(self):
        with pytest.raises(TypeError, match="All command arguments must be strings"):
            execute_ffmpeg_command(["ffmpeg", 123])  # type: ignore

    def test_execute_invalid_timeout_raises_error(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            execute_ffmpeg_command(["ffmpeg", "-version"], timeout=0.0)

    def test_execute_disallowed_binary_raises_error(self):
        with pytest.raises(ValueError, match="Permitted binary must be ffmpeg or ffprobe"):
            execute_ffmpeg_command(["bash", "-c", "echo test"])


class TestEdgeCasesAndPathTypes:
    """Validate boundary cases: 1 clip, 0 clips, negative durations, Path vs str types."""

    def test_single_clip_xfade_chain_boundary(self):
        with pytest.raises(ValueError, match="At least 2 clips are required"):
            build_xfade_chain(["clip1.mp4"], [], [10.0], "out.mp4")

    def test_single_clip_audio_crossfade_boundary(self):
        with pytest.raises(ValueError, match="At least 2 clips are required"):
            build_audio_crossfade(["clip1.mp4"], [10.0])

    def test_zero_clips_everywhere(self):
        with pytest.raises(ValueError, match="clips sequence cannot be empty"):
            build_xfade_chain([], [], [], "out.mp4")
        with pytest.raises(ValueError, match="clips sequence cannot be empty"):
            build_audio_crossfade([], [])
        with pytest.raises(ValueError, match="durations list cannot be empty"):
            calculate_xfade_offsets([])

    def test_path_objects_handled_properly(self):
        cmd = build_trim_command(
            clip_path=Path("folder/clip.mp4"),
            start=1.0,
            end=3.0,
            output_path=Path("folder/trimmed.mp4"),
        )
        assert str(Path("folder/clip.mp4")) in cmd
        assert str(Path("folder/trimmed.mp4")) in cmd

    def test_exact_clip_duration_equal_to_transition_duration(self):
        # A clip of 1.0s cannot have a 1.0s transition because offset would be 0 or stream exhausts
        with pytest.raises(ValueError, match="must be strictly greater than transition duration"):
            calculate_xfade_offsets([1.0, 5.0], transition_duration=1.0)
