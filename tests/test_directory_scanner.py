"""Unit and integration tests for SPEC-004: Local Directory Scanner Tool."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from google.adk.agents import Agent

from schemas.media import ScannedVideoFile
from tools.directory_scanner import (
    SUPPORTED_VIDEO_EXTENSIONS,
    extract_file_basic_metadata,
    scan_local_directory,
    scan_local_directory_with_metadata,
)


@pytest.fixture
def temp_media_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory tree with various files."""
    # Root level video files
    (tmp_path / "clip1.mp4").write_bytes(b"dummy_mp4_content_12345")
    (tmp_path / "clip2.MOV").write_bytes(b"dummy_mov_uppercase_content")
    (tmp_path / "clip3.avi").write_bytes(b"dummy_avi_content")
    (tmp_path / "clip4.mkv").write_bytes(b"dummy_mkv_content")

    # Non-video files
    (tmp_path / "notes.txt").write_text("not a video")
    (tmp_path / "thumbnail.jpg").write_bytes(b"image_bytes")
    (tmp_path / "audio.mp3").write_bytes(b"audio_bytes")

    # Nested directory with video files
    nested_dir = tmp_path / "day_01" / "gopro"
    nested_dir.mkdir(parents=True)
    (nested_dir / "action_shot.mp4").write_bytes(b"nested_action_bytes")
    (nested_dir / "drone_footage.MKV").write_bytes(b"nested_drone_bytes")
    (nested_dir / "script.doc").write_text("doc text")

    # Hidden files and hidden directory (should be ignored)
    (tmp_path / ".hidden_clip.mp4").write_bytes(b"hidden_bytes")
    hidden_dir = tmp_path / ".git" / "hooks"
    hidden_dir.mkdir(parents=True)
    (hidden_dir / "git_internal.mp4").write_bytes(b"git_bytes")

    return tmp_path


class TestScannedVideoFileSchema:
    """Test suite for ScannedVideoFile Pydantic schema."""

    def test_valid_schema_instantiation(self) -> None:
        model = ScannedVideoFile(
            path="C:/media/clip.mp4",
            file_name="clip.mp4",
            file_size_bytes=1024,
            last_modified=1700000000.0,
            last_modified_iso="2023-11-14T22:13:20+00:00",
            extension=".mp4",
        )
        assert model.path == "C:/media/clip.mp4"
        assert model.file_name == "clip.mp4"
        assert model.file_size_bytes == 1024
        assert model.extension == ".mp4"

    def test_extension_validator_auto_dot_and_lower(self) -> None:
        model = ScannedVideoFile(
            path="C:/media/clip.MP4",
            file_name="clip.MP4",
            file_size_bytes=500,
            last_modified=1700000000.0,
            last_modified_iso="2023-11-14T22:13:20+00:00",
            extension="MOV",
        )
        assert model.extension == ".mov"

    def test_from_path_classmethod(self, tmp_path: Path) -> None:
        sample_file = tmp_path / "sample.mp4"
        sample_file.write_bytes(b"0123456789")

        model = ScannedVideoFile.from_path(sample_file)
        assert model.path == str(sample_file.resolve())
        assert model.file_name == "sample.mp4"
        assert model.file_size_bytes == 10
        assert model.extension == ".mp4"
        assert model.last_modified > 0
        assert "T" in model.last_modified_iso

    def test_negative_size_validation_error(self) -> None:
        with pytest.raises(ValueError):
            ScannedVideoFile(
                path="/clip.mp4",
                file_name="clip.mp4",
                file_size_bytes=-10,
                last_modified=1700000000.0,
                last_modified_iso="2023-11-14T22:13:20+00:00",
                extension=".mp4",
            )


class TestExtractFileBasicMetadata:
    """Test suite for extract_file_basic_metadata helper."""

    def test_extract_metadata_success(self, tmp_path: Path) -> None:
        file = tmp_path / "test_extract.mov"
        payload = b"test_payload_mov_123"
        file.write_bytes(payload)

        meta = extract_file_basic_metadata(file)
        assert meta.file_name == "test_extract.mov"
        assert meta.file_size_bytes == len(payload)
        assert meta.extension == ".mov"
        assert meta.path == str(file.resolve())

    def test_extract_non_existent_file_raises_not_found(self, tmp_path: Path) -> None:
        fake_file = tmp_path / "non_existent.mp4"
        with pytest.raises(FileNotFoundError, match="Video file does not exist"):
            extract_file_basic_metadata(fake_file)

    def test_extract_directory_raises_value_error(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "sub_dir"
        sub_dir.mkdir()
        with pytest.raises(ValueError, match="Path is not a regular file"):
            extract_file_basic_metadata(sub_dir)


class TestScanLocalDirectory:
    """Test suite for scan_local_directory and scan_local_directory_with_metadata."""

    def test_scan_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = scan_local_directory(str(empty_dir))
        assert result == []

    def test_scan_discovers_all_supported_extensions(self, temp_media_dir: Path) -> None:
        paths = scan_local_directory(str(temp_media_dir))

        # We expect: clip1.mp4, clip2.MOV, clip3.avi, clip4.mkv,
        # day_01/gopro/action_shot.mp4, day_01/gopro/drone_footage.MKV
        assert len(paths) == 6

        basenames = [Path(p).name for p in paths]
        assert "clip1.mp4" in basenames
        assert "clip2.MOV" in basenames
        assert "clip3.avi" in basenames
        assert "clip4.mkv" in basenames
        assert "action_shot.mp4" in basenames
        assert "drone_footage.MKV" in basenames

    def test_scan_ignores_non_video_and_hidden_files(self, temp_media_dir: Path) -> None:
        paths = scan_local_directory(str(temp_media_dir))
        basenames = [Path(p).name for p in paths]

        # Non-video files
        assert "notes.txt" not in basenames
        assert "thumbnail.jpg" not in basenames
        assert "audio.mp3" not in basenames
        assert "script.doc" not in basenames

        # Hidden files
        assert ".hidden_clip.mp4" not in basenames
        assert "git_internal.mp4" not in basenames

    def test_scan_returns_absolute_paths(self, temp_media_dir: Path) -> None:
        paths = scan_local_directory(str(temp_media_dir))
        for p in paths:
            assert os.path.isabs(p)
            assert Path(p).exists()

    def test_scan_deterministic_sorting(self, temp_media_dir: Path) -> None:
        paths_first_run = scan_local_directory(str(temp_media_dir))
        paths_second_run = scan_local_directory(str(temp_media_dir))
        assert paths_first_run == paths_second_run
        assert paths_first_run == sorted(paths_first_run)

    def test_scan_with_metadata_returns_models(self, temp_media_dir: Path) -> None:
        models = scan_local_directory_with_metadata(temp_media_dir)
        assert len(models) == 6
        for m in models:
            assert isinstance(m, ScannedVideoFile)
            assert m.file_size_bytes > 0
            assert m.extension in SUPPORTED_VIDEO_EXTENSIONS
            assert Path(m.path).exists()


class TestDirectoryScannerValidationErrors:
    """Test suite for error handling on invalid directory paths."""

    def test_empty_string_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="directory_path must be a non-empty string"):
            scan_local_directory("")

    def test_whitespace_string_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="directory_path must be a non-empty string"):
            scan_local_directory("   ")

    def test_non_existent_directory_raises_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="Target directory does not exist"):
            scan_local_directory(str(missing))

    def test_file_passed_as_directory_raises_not_a_directory(self, tmp_path: Path) -> None:
        real_file = tmp_path / "video.mp4"
        real_file.write_bytes(b"content")
        with pytest.raises(NotADirectoryError, match="Target path is not a directory"):
            scan_local_directory(str(real_file))


class TestADKToolDeclaration:
    """Test compatibility with Google ADK Agent tool definitions."""

    def test_agent_tool_binding(self) -> None:
        agent = Agent(
            name="test_scanner_agent",
            model="gemini-2.0-flash",
            instruction="Test agent instructions.",
            tools=[scan_local_directory],
        )
        assert len(agent.tools) == 1
        assert agent.tools[0] == scan_local_directory
        assert scan_local_directory.__name__ == "scan_local_directory"
        assert callable(scan_local_directory)


class TestMainCLIScanDir:
    """Test CLI --scan-dir option."""

    def test_cli_scan_dir_with_videos(self, temp_media_dir: Path) -> None:
        cmd = [
            sys.executable,
            "main.py",
            "--scan-dir",
            str(temp_media_dir),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "VideoGuru: Scanning Local Directory for Media (SPEC-004)" in result.stdout
        assert "Discovered 6 video file(s)" in result.stdout
        assert "clip1.mp4" in result.stdout
        assert "drone_footage.MKV" in result.stdout

    def test_cli_scan_dir_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_cli_test"
        empty.mkdir()
        cmd = [
            sys.executable,
            "main.py",
            "--scan-dir",
            str(empty),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "No supported video files (.mp4, .mov, .avi, .mkv) found in directory." in result.stdout

    def test_cli_scan_dir_invalid_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing_dir_123"
        cmd = [
            sys.executable,
            "main.py",
            "--scan-dir",
            str(missing),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "Scan failed:" in result.stderr
