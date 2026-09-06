"""Unit and integration tests for SPEC-005: Clip Metadata Extraction Tool."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import typing
import pytest
from google.adk.agents import Agent
from pydantic import ValidationError


from main import main
from schemas.media import ClipManifestEntry
from tools.clip_metadata import (
    _parse_duration,
    _parse_frame_rate,
    _parse_rational_fps,
    extract_clip_metadata,
    find_ffprobe_executable,
    probe_video_file,
)


class TestClipManifestEntrySchema:
    """Test suite validating the ClipManifestEntry Pydantic schema."""

    def test_valid_instantiation(self):
        entry = ClipManifestEntry(
            clip_id="vid_8f3a9b21",
            absolute_path="C:/Users/Media/clip_01.mp4",
            file_name="clip_01.mp4",
            duration_seconds=14.35,
            frame_rate=29.97,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=1048576,
        )
        assert entry.clip_id == "vid_8f3a9b21"
        assert entry.absolute_path == "C:/Users/Media/clip_01.mp4"
        assert entry.file_name == "clip_01.mp4"
        assert entry.duration_seconds == 14.35
        assert entry.frame_rate == 29.97
        assert entry.resolution == "1920x1080"
        assert entry.width == 1920
        assert entry.height == 1080
        assert entry.video_codec == "h264"
        assert entry.audio_codec == "aac"
        assert entry.has_audio is True
        assert entry.file_size_bytes == 1048576

    def test_convenience_property_aliases(self):
        entry = ClipManifestEntry(
            clip_id="vid_12345678",
            absolute_path="/tmp/video.mov",
            file_name="video.mov",
            duration_seconds=5.0,
            frame_rate=30.0,
            resolution="1280x720",
            width=1280,
            height=720,
            video_codec="hevc",
        )
        assert entry.file_path == "/tmp/video.mov"
        assert entry.codec == "hevc"
        assert entry.has_audio is False
        assert entry.audio_codec is None

    def test_alias_handling_in_model_validate(self):
        data = {
            "clip_id": "vid_alias01",
            "file_path": "/var/media/test.mp4",
            "file_name": "test.mp4",
            "duration_seconds": 10.0,
            "frame_rate": 60.0,
            "resolution": "3840x2160",
            "width": 3840,
            "height": 2160,
            "codec": "vp9",
        }
        entry = ClipManifestEntry.model_validate(data)
        assert entry.absolute_path == "/var/media/test.mp4"
        assert entry.video_codec == "vp9"

    def test_invalid_resolution_formats(self):
        # Missing 'x'
        with pytest.raises(ValidationError):
            ClipManifestEntry(
                clip_id="vid_err01",
                absolute_path="/tmp/v.mp4",
                file_name="v.mp4",
                duration_seconds=1.0,
                frame_rate=30.0,
                resolution="19201080",
                width=1920,
                height=1080,
                video_codec="h264",
            )

        # Non-digit
        with pytest.raises(ValidationError):
            ClipManifestEntry(
                clip_id="vid_err02",
                absolute_path="/tmp/v.mp4",
                file_name="v.mp4",
                duration_seconds=1.0,
                frame_rate=30.0,
                resolution="1920xabc",
                width=1920,
                height=1080,
                video_codec="h264",
            )

        # Zero or negative dimension
        with pytest.raises(ValidationError):
            ClipManifestEntry(
                clip_id="vid_err03",
                absolute_path="/tmp/v.mp4",
                file_name="v.mp4",
                duration_seconds=1.0,
                frame_rate=30.0,
                resolution="1920x0",
                width=1920,
                height=0,
                video_codec="h264",
            )

    def test_invalid_duration_and_frame_rate(self):
        # Negative duration
        with pytest.raises(ValidationError):
            ClipManifestEntry(
                clip_id="vid_err04",
                absolute_path="/tmp/v.mp4",
                file_name="v.mp4",
                duration_seconds=-1.5,
                frame_rate=30.0,
                resolution="1280x720",
                width=1280,
                height=720,
                video_codec="h264",
            )

        # Zero frame rate
        with pytest.raises(ValidationError):
            ClipManifestEntry(
                clip_id="vid_err05",
                absolute_path="/tmp/v.mp4",
                file_name="v.mp4",
                duration_seconds=1.0,
                frame_rate=0.0,
                resolution="1280x720",
                width=1280,
                height=720,
                video_codec="h264",
            )

    def test_generate_clip_id(self):
        cid1 = ClipManifestEntry.generate_clip_id()
        cid2 = ClipManifestEntry.generate_clip_id()
        assert cid1.startswith("vid_")
        assert len(cid1) == 12  # 'vid_' (4) + 8 hex chars
        assert cid1 != cid2

        custom_prefix = ClipManifestEntry.generate_clip_id(prefix="clip_")
        assert custom_prefix.startswith("clip_")

    def test_serialization(self):
        entry = ClipManifestEntry(
            clip_id="vid_ser01",
            absolute_path="/tmp/clip.mp4",
            file_name="clip.mp4",
            duration_seconds=3.14,
            frame_rate=25.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
        )
        d = entry.model_dump()
        assert d["clip_id"] == "vid_ser01"
        assert d["resolution"] == "1920x1080"

        json_str = entry.model_dump_json()
        assert "vid_ser01" in json_str


class TestInternalHelperParsers:
    """Test parsing logic for frame rates and durations."""

    def test_parse_rational_fps(self):
        assert _parse_rational_fps("30/1") == 30.0
        assert _parse_rational_fps("24/1") == 24.0
        assert _parse_rational_fps("30000/1001") == 29.97
        assert _parse_rational_fps("24000/1001") == 23.98
        assert _parse_rational_fps("60/1") == 60.0
        assert _parse_rational_fps("25") == 25.0
        assert _parse_rational_fps("0/0") is None
        assert _parse_rational_fps("") is None
        assert _parse_rational_fps("invalid") is None
        assert _parse_rational_fps("10/0") is None

    def test_parse_frame_rate(self):
        assert _parse_frame_rate("30/1", "0/0") == 30.0
        assert _parse_frame_rate("0/0", "25/1") == 25.0
        with pytest.raises(ValueError):
            _parse_frame_rate("0/0", "0/0")

    def test_parse_duration(self):
        # From video stream
        assert _parse_duration({"duration": "14.35"}, {"duration": "14.4"}) == 14.35
        # Fallback to container format
        assert _parse_duration({"duration": "N/A"}, {"duration": "10.5"}) == 10.5
        # From tag HH:MM:SS
        assert _parse_duration({"tags": {"DURATION": "00:01:30.500"}}, {}) == 90.5
        # Unresolvable duration
        with pytest.raises(ValueError):
            _parse_duration({}, {})


class TestProbeVideoFile:
    """Test probe_video_file error handling and validation."""

    def test_find_ffprobe_executable(self):
        ffprobe_bin = find_ffprobe_executable()
        assert ffprobe_bin is not None
        assert Path(ffprobe_bin).exists()

    def test_empty_path_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty string"):
            probe_video_file("")
        with pytest.raises(ValueError, match="non-empty string"):
            probe_video_file("   ")

    def test_non_existent_file_raises_not_found(self, tmp_path):
        missing = tmp_path / "non_existent.mp4"
        with pytest.raises(FileNotFoundError):
            probe_video_file(str(missing))

    def test_directory_passed_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="regular file"):
            probe_video_file(str(tmp_path))

    def test_corrupt_file_raises_runtime_error(self, tmp_path):
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"This is not a valid mp4 header or media stream.")
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            probe_video_file(str(corrupt))

    def test_missing_ffprobe_simulation(self, monkeypatch, tmp_path):
        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"dummy")

        monkeypatch.setattr(shutil, "which", lambda *args, **kwargs: None)
        with pytest.raises(RuntimeError, match="ffprobe executable not found"):
            probe_video_file(str(test_file))


@pytest.fixture(scope="module")
def synthetic_media(tmp_path_factory):
    """Generate synthetic media clips using ffmpeg lavfi for end-to-end probing tests."""
    temp_dir = tmp_path_factory.mktemp("media_clips")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("ffmpeg binary not available for synthetic media creation")

    # Clip 1: 720p 30fps with audio (1.5s)
    clip_720p = temp_dir / "clip_720p.mp4"
    cmd_720p = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1.5:size=1280x720:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1.5",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        str(clip_720p),
    ]
    subprocess.run(cmd_720p, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Clip 2: 1080p 24fps silent (1.0s)
    clip_1080p = temp_dir / "clip_1080p.mp4"
    cmd_1080p = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1.0:size=1920x1080:rate=24",
        "-c:v",
        "libx264",
        "-an",
        "-y",
        str(clip_1080p),
    ]
    subprocess.run(cmd_1080p, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Clip 3: Audio only (no video stream)
    audio_only = temp_dir / "audio_only.mp4"
    cmd_audio = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1.0",
        "-c:a",
        "aac",
        "-vn",
        "-y",
        str(audio_only),
    ]
    subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    return {
        "clip_720p": clip_720p,
        "clip_1080p": clip_1080p,
        "audio_only": audio_only,
    }


class TestExtractClipMetadata:
    """Test extract_clip_metadata tool against real media files."""

    def test_extract_720p_with_audio(self, synthetic_media):
        clip_path = str(synthetic_media["clip_720p"])
        entry = extract_clip_metadata(clip_path)

        assert isinstance(entry, ClipManifestEntry)
        assert entry.clip_id.startswith("vid_")
        assert entry.file_name == "clip_720p.mp4"
        assert Path(entry.absolute_path) == synthetic_media["clip_720p"].resolve()
        assert entry.resolution == "1280x720"
        assert entry.width == 1280
        assert entry.height == 720
        assert entry.frame_rate == 30.0
        assert abs(entry.duration_seconds - 1.5) < 0.2
        assert "h264" in entry.video_codec
        assert entry.has_audio is True
        assert entry.audio_codec == "aac"
        assert entry.file_size_bytes > 0

    def test_extract_1080p_silent_video(self, synthetic_media):
        clip_path = str(synthetic_media["clip_1080p"])
        entry = extract_clip_metadata(clip_path)

        assert entry.resolution == "1920x1080"
        assert entry.width == 1920
        assert entry.height == 1080
        assert entry.frame_rate == 24.0
        assert abs(entry.duration_seconds - 1.0) < 0.2
        assert "h264" in entry.video_codec
        assert entry.has_audio is False
        assert entry.audio_codec is None

    def test_custom_clip_id_provided(self, synthetic_media):
        clip_path = str(synthetic_media["clip_720p"])
        entry = extract_clip_metadata(clip_path, clip_id="my_custom_clip_001")
        assert entry.clip_id == "my_custom_clip_001"

    def test_audio_only_file_raises_no_video_stream(self, synthetic_media):
        audio_path = str(synthetic_media["audio_only"])
        with pytest.raises(ValueError, match="No video stream found"):
            extract_clip_metadata(audio_path)

    def test_non_existent_file_raises_not_found(self, tmp_path):
        missing = tmp_path / "ghost.mp4"
        with pytest.raises(FileNotFoundError):
            extract_clip_metadata(str(missing))


class TestADKToolDeclaration:
    """Validate that extract_clip_metadata complies with Google ADK tool conventions."""

    def test_tool_callable_and_signature(self):
        assert callable(extract_clip_metadata)
        hints = typing.get_type_hints(extract_clip_metadata)
        assert hints["file_path"] is str
        assert hints["return"] is ClipManifestEntry
        assert extract_clip_metadata.__doc__ is not None
        assert "ClipManifestEntry" in extract_clip_metadata.__doc__

    def test_agent_tool_binding(self):
        agent = Agent(
            name="test_metadata_agent",
            model="gemini-2.0-flash",
            instruction="Test metadata extraction agent.",
            tools=[extract_clip_metadata],
        )
        assert len(agent.tools) == 1
        assert agent.tools[0] == extract_clip_metadata
        assert callable(agent.tools[0])
        assert extract_clip_metadata.__name__ == "extract_clip_metadata"



class TestMainCLIExtractMetadata:
    """Test CLI argument handling for --extract-metadata / -m in main.py."""

    def test_cli_extract_metadata_flag(self, synthetic_media, monkeypatch, capsys):
        clip_path = str(synthetic_media["clip_720p"])
        monkeypatch.setattr(sys, "argv", ["main.py", "--extract-metadata", clip_path])

        main()
        captured = capsys.readouterr()
        assert "Extracting Clip Metadata via ffprobe (SPEC-005)" in captured.out
        assert "Clip ID:" in captured.out
        assert "1280x720" in captured.out
        assert "clip_720p.mp4" in captured.out
        assert "Video Codec:    h264" in captured.out

    def test_cli_shorthand_m_flag(self, synthetic_media, monkeypatch, capsys):
        clip_path = str(synthetic_media["clip_1080p"])
        monkeypatch.setattr(sys, "argv", ["main.py", "-m", clip_path])

        main()
        captured = capsys.readouterr()
        assert "Extracting Clip Metadata via ffprobe (SPEC-005)" in captured.out
        assert "1920x1080" in captured.out
        assert "clip_1080p.mp4" in captured.out
        assert "Has Audio:      No" in captured.out

    def test_cli_extract_metadata_invalid_file(self, tmp_path, monkeypatch, capsys):
        missing = str(tmp_path / "does_not_exist.mp4")
        monkeypatch.setattr(sys, "argv", ["main.py", "-m", missing])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Metadata extraction failed" in captured.err
