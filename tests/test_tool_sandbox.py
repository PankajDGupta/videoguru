"""Unit and Integration Tests for SPEC-022 Tool Sandboxing & FFmpeg Validation Callback."""

import logging
from pathlib import Path
import pytest

from callbacks.tool_sandbox import (
    SecuritySandboxingError,
    before_tool_sandbox_callback,
    is_shell_safe,
    validate_ffmpeg_command,
    validate_path_confined,
)
from config import settings


@pytest.fixture
def sandbox_dirs(tmp_path):
    """Fixture providing sandboxed input, staging, and output directories."""
    media_dir = tmp_path / "media"
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "output"
    media_dir.mkdir()
    staging_dir.mkdir()
    output_dir.mkdir()
    return [media_dir, staging_dir, output_dir]


class TestShellSafetyAndPathConfinement:
    """Test suite for shell metacharacter checking and path sandboxing."""

    @pytest.mark.parametrize(
        "safe_value",
        [
            "clip_01.mp4",
            "1920x1080",
            "fast",
            "libx264",
            "23",
            "fade",
            "wipeleft",
            "sidechaincompress",
        ],
    )
    def test_safe_arguments_pass_shell_check(self, safe_value: str):
        assert is_shell_safe(safe_value) is True

    @pytest.mark.parametrize(
        "dangerous_value",
        [
            "clip.mp4; rm -rf /",
            "clip.mp4 && del /f",
            "clip.mp4 | bash",
            "clip.mp4 `whoami`",
            "clip.mp4 $(calc.exe)",
            "output.mp4 > /dev/null",
            "output.mp4 < input.txt",
            "clip.mp4\nwhoami",
        ],
    )
    def test_dangerous_arguments_fail_shell_check(self, dangerous_value: str):
        assert is_shell_safe(dangerous_value) is False

    def test_validate_path_confined_allowed_path(self, sandbox_dirs):
        media_dir = sandbox_dirs[0]
        test_file = media_dir / "clip_01.mp4"
        test_file.touch()

        resolved = validate_path_confined(str(test_file), allowed_dirs=sandbox_dirs)
        assert resolved == test_file.resolve()

    def test_validate_path_confined_traversal_blocked(self, sandbox_dirs):
        media_dir = sandbox_dirs[0]
        traversal_path = str(media_dir) + "/../secret.txt"
        with pytest.raises(SecuritySandboxingError, match="Path traversal"):
            validate_path_confined(traversal_path, allowed_dirs=sandbox_dirs)

    def test_validate_path_confined_unauthorized_dir_blocked(self, sandbox_dirs, tmp_path):
        unauthorized = tmp_path / "unauthorized" / "file.mp4"
        with pytest.raises(SecuritySandboxingError, match="outside designated sandbox"):
            validate_path_confined(str(unauthorized), allowed_dirs=sandbox_dirs)

    def test_validate_path_confined_forbidden_protocol_blocked(self, sandbox_dirs):
        with pytest.raises(SecuritySandboxingError, match="Network protocol"):
            validate_path_confined("http://example.com/clip.mp4", allowed_dirs=sandbox_dirs)

    def test_validate_path_confined_sensitive_system_path_blocked(self, sandbox_dirs):
        with pytest.raises(SecuritySandboxingError, match="sensitive system path"):
            validate_path_confined("/etc/passwd", allowed_dirs=sandbox_dirs)


class TestFFmpegCommandValidation:
    """Test suite for validate_ffmpeg_command CLI sandboxing."""

    def test_valid_ffmpeg_command_passes(self, sandbox_dirs):
        media_dir, staging_dir, output_dir = sandbox_dirs
        in_file = media_dir / "input.mp4"
        out_file = output_dir / "output.mp4"
        in_file.touch()

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(in_file),
            "-c:v",
            "libx264",
            "-crf",
            "23",
            str(out_file),
        ]
        # Should not raise
        validate_ffmpeg_command(cmd, allowed_dirs=sandbox_dirs)

    def test_disallowed_binary_blocked(self, sandbox_dirs):
        cmd = ["bash", "-c", "echo hello"]
        with pytest.raises(SecuritySandboxingError, match="Binary 'bash' is not permitted"):
            validate_ffmpeg_command(cmd, allowed_dirs=sandbox_dirs)

    def test_forbidden_flag_blocked(self, sandbox_dirs):
        media_dir, _, output_dir = sandbox_dirs
        cmd = [
            "ffmpeg",
            "-protocol_whitelist",
            "file,http",
            "-i",
            str(media_dir / "input.mp4"),
            str(output_dir / "out.mp4"),
        ]
        with pytest.raises(SecuritySandboxingError, match="Prohibited FFmpeg option flag"):
            validate_ffmpeg_command(cmd, allowed_dirs=sandbox_dirs)

    def test_unrecognized_flag_blocked(self, sandbox_dirs):
        media_dir, _, output_dir = sandbox_dirs
        cmd = [
            "ffmpeg",
            "-malicious_option",
            "true",
            "-i",
            str(media_dir / "input.mp4"),
            str(output_dir / "out.mp4"),
        ]
        with pytest.raises(SecuritySandboxingError, match="Unrecognized or unwhitelisted FFmpeg flag"):
            validate_ffmpeg_command(cmd, allowed_dirs=sandbox_dirs)

    def test_shell_injection_in_command_blocked(self, sandbox_dirs):
        media_dir, _, output_dir = sandbox_dirs
        cmd = [
            "ffmpeg",
            "-i",
            str(media_dir / "input.mp4; rm -rf /"),
            str(output_dir / "out.mp4"),
        ]
        with pytest.raises(SecuritySandboxingError, match="Shell injection characters detected"):
            validate_ffmpeg_command(cmd, allowed_dirs=sandbox_dirs)

    def test_empty_command_blocked(self, sandbox_dirs):
        with pytest.raises(SecuritySandboxingError, match="cannot be empty"):
            validate_ffmpeg_command([], allowed_dirs=sandbox_dirs)


class TestBeforeToolSandboxCallback:
    """Test suite for ADK before_tool_callback integration."""

    class MockTool:
        name = "execute_ffmpeg_command"

    def test_callback_allows_valid_cmd(self, sandbox_dirs):
        media_dir, _, output_dir = sandbox_dirs
        in_file = media_dir / "clip_01.mp4"
        out_file = output_dir / "final.mp4"
        in_file.touch()

        args = {
            "cmd": [
                "ffmpeg",
                "-y",
                "-i",
                str(in_file),
                "-c:v",
                "copy",
                str(out_file),
            ]
        }
        res = before_tool_sandbox_callback(
            tool=self.MockTool(),
            args=args,
            allowed_dirs=sandbox_dirs,
        )
        assert res is None  # Permitted

    def test_callback_raises_on_cmd_violation(self, sandbox_dirs):
        args = {"cmd": ["curl", "http://evil.com/video.mp4"]}
        with pytest.raises(SecuritySandboxingError):
            before_tool_sandbox_callback(
                tool=self.MockTool(),
                args=args,
                allowed_dirs=sandbox_dirs,
                raise_exception=True,
            )

    def test_callback_returns_error_dict_when_raise_exception_false(self, sandbox_dirs):
        args = {"cmd": ["powershell", "-c", "dir"]}
        result = before_tool_sandbox_callback(
            tool=self.MockTool(),
            args=args,
            allowed_dirs=sandbox_dirs,
            raise_exception=False,
        )
        assert isinstance(result, dict)
        assert result.get("status") == "error"
        assert result.get("blocked_by_sandbox") is True

    def test_callback_validates_path_kwargs(self, sandbox_dirs):
        media_dir = sandbox_dirs[0]
        valid_clip = media_dir / "test.mp4"
        valid_clip.touch()

        # Valid keyword path
        args = {"clip_path": str(valid_clip), "theme": "Surfing"}
        assert before_tool_sandbox_callback(
            tool=self.MockTool(),
            args=args,
            allowed_dirs=sandbox_dirs,
        ) is None

        # Traversal keyword path
        bad_args = {"clip_path": str(media_dir) + "/../secret.key"}
        with pytest.raises(SecuritySandboxingError, match="Path traversal"):
            before_tool_sandbox_callback(
                tool=self.MockTool(),
                args=bad_args,
                allowed_dirs=sandbox_dirs,
            )

    def test_callback_blocks_shell_metacharacters_in_kwargs(self, sandbox_dirs):
        bad_args = {"theme": "Vacation; rm -rf /"}
        with pytest.raises(SecuritySandboxingError, match="Shell metacharacters detected"):
            before_tool_sandbox_callback(
                tool=self.MockTool(),
                args=bad_args,
                allowed_dirs=sandbox_dirs,
            )
