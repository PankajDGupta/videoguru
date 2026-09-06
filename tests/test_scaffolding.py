"""Unit tests for SPEC-001: Project Scaffolding & Dependencies Verification."""

import os
import shutil
import subprocess
from pathlib import Path
import pytest


def get_refreshed_path():
    """Retrieve full system PATH including any newly installed tools."""
    # Combine current os.environ PATH with registry user/machine PATH if on Windows
    paths = os.environ.get("PATH", "").split(os.pathsep)
    if os.name == "nt":
        import winreg
        for root in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
            subkey = r"Environment" if root == winreg.HKEY_CURRENT_USER else r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
            try:
                with winreg.OpenKey(root, subkey) as key:
                    val, _ = winreg.QueryValueEx(key, "Path")
                    for p in val.split(";"):
                        if p and p not in paths:
                            paths.append(p)
            except Exception:
                pass
    return os.pathsep.join(paths)


class TestProjectStructure:
    """Validate directory structure and required files for SPEC-001."""

    @pytest.fixture(autouse=True)
    def setup_base_dir(self):
        self.root_dir = Path(__file__).resolve().parent.parent

    def test_required_directories_exist(self):
        required_dirs = [
            "agents",
            "tools",
            "schemas",
            "services",
            "rendering",
            "tests",
            "config",
            "context",
        ]
        for dirname in required_dirs:
            dir_path = self.root_dir / dirname
            assert dir_path.is_dir(), f"Expected directory '{dirname}' does not exist at {dir_path}"

    def test_required_files_exist(self):
        required_files = [
            "pyproject.toml",
            "requirements.txt",
            ".gitignore",
            "main.py",
            "config/settings.py",
        ]
        for filename in required_files:
            file_path = self.root_dir / filename
            assert file_path.is_file(), f"Expected file '{filename}' does not exist at {file_path}"


class TestInternalPackageImports:
    """Validate that all created modules are valid importable Python packages."""

    def test_import_internal_packages(self):
        import agents
        import tools
        import schemas
        import services
        import rendering
        import config
        from config import settings

        assert agents is not None
        assert tools is not None
        assert schemas is not None
        assert services is not None
        assert rendering is not None
        assert config is not None
        assert hasattr(settings, "BASE_DIR")
        assert hasattr(settings, "MEDIA_INPUT_DIR")
        assert hasattr(settings, "STAGING_DIR")
        assert hasattr(settings, "OUTPUT_DIR")


class TestCoreDependencyImports:
    """Validate that external core dependencies install and import cleanly."""

    def test_import_google_adk(self):
        import google.adk
        assert google.adk is not None

    def test_import_google_genai(self):
        import google.genai
        assert google.genai is not None

    def test_import_pydantic(self):
        import pydantic
        assert hasattr(pydantic, "BaseModel")

    def test_import_opentimelineio(self):
        import opentimelineio as otio
        assert hasattr(otio, "schema")
        timeline = otio.schema.Timeline(name="TestTimeline")
        assert timeline.name == "TestTimeline"

    def test_import_whisper(self):
        import whisper
        assert hasattr(whisper, "load_model")


class TestFFmpegAvailability:
    """Validate that FFmpeg and FFprobe binaries are accessible on PATH."""

    def test_ffmpeg_accessible(self):
        full_path = get_refreshed_path()
        ffmpeg_bin = shutil.which("ffmpeg", path=full_path)
        assert ffmpeg_bin is not None, "ffmpeg binary not found in PATH"

        # Verify executing ffmpeg -version
        result = subprocess.run(
            [ffmpeg_bin, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env={**os.environ, "PATH": full_path}
        )
        assert result.returncode == 0
        assert "ffmpeg version" in result.stdout

    def test_ffprobe_accessible(self):
        full_path = get_refreshed_path()
        ffprobe_bin = shutil.which("ffprobe", path=full_path)
        assert ffprobe_bin is not None, "ffprobe binary not found in PATH"

        # Verify executing ffprobe -version
        result = subprocess.run(
            [ffprobe_bin, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            env={**os.environ, "PATH": full_path}
        )
        assert result.returncode == 0
        assert "ffprobe version" in result.stdout
