"""Unit tests for SPEC-027: Configuration & Environment Setup."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch
import pytest

import config
from config import settings
from config.settings import (
    SUPPORTED_WHISPER_MODELS,
    _resolve_default_input_dir,
    _resolve_default_output_dir,
    ensure_directories,
    reload_settings,
    validate_settings,
)


@pytest.fixture(autouse=True)
def restore_settings_after_test():
    """Ensure settings are restored to clean environment state after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)
    reload_settings()


class TestDefaultSettings:
    """Validate default configuration values, types, and paths."""

    def test_base_dir_properties(self):
        """BASE_DIR must be an existing, absolute Path pointing to project root."""
        assert isinstance(settings.BASE_DIR, Path)
        assert settings.BASE_DIR.is_absolute()
        assert settings.BASE_DIR.exists()
        assert (settings.BASE_DIR / "config").is_dir()

    def test_default_directory_types(self):
        """MEDIA_INPUT_DIR, STAGING_DIR, and OUTPUT_DIR must be Path instances."""
        assert isinstance(settings.MEDIA_INPUT_DIR, Path)
        assert isinstance(settings.STAGING_DIR, Path)
        assert isinstance(settings.OUTPUT_DIR, Path)

    def test_default_directory_targets(self):
        """MEDIA_INPUT_DIR defaults to input_videos (or media), OUTPUT_DIR to output_video (or output)."""
        expected_input = settings.BASE_DIR / "input_videos"
        if not expected_input.exists() and (settings.BASE_DIR / "media").exists():
            expected_input = settings.BASE_DIR / "media"

        expected_output = settings.BASE_DIR / "output_video"
        if not expected_output.exists() and (settings.BASE_DIR / "output").exists():
            expected_output = settings.BASE_DIR / "output"

        assert settings.MEDIA_INPUT_DIR == expected_input
        assert settings.STAGING_DIR == settings.BASE_DIR / "staging"
        assert settings.OUTPUT_DIR == expected_output

    def test_default_models_and_algorithms(self):
        """Validate default models and loop limits."""
        assert isinstance(settings.GEMINI_MODEL, str)
        assert settings.GEMINI_MODEL in ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.6-flash")
        assert isinstance(settings.WHISPER_MODEL, str)
        assert settings.WHISPER_MODEL == "medium"
        assert isinstance(settings.MAX_LOOP_ITERATIONS, int)
        assert not isinstance(settings.MAX_LOOP_ITERATIONS, bool)
        assert settings.MAX_LOOP_ITERATIONS == 5

    def test_default_web_and_app_settings(self):
        """Validate web server and application identity defaults."""
        assert isinstance(settings.APP_NAME, str)
        assert settings.APP_NAME == "videoguru"
        assert isinstance(settings.WEB_HOST, str)
        assert settings.WEB_HOST == "127.0.0.1"
        assert isinstance(settings.WEB_PORT, int)
        assert not isinstance(settings.WEB_PORT, bool)
        assert settings.WEB_PORT == 8000
        assert isinstance(settings.DEFAULT_USER_ID, str)
        assert settings.DEFAULT_USER_ID == "videoguru_user"

    def test_supported_whisper_models_set(self):
        """Validate supported Whisper model definitions."""
        assert isinstance(SUPPORTED_WHISPER_MODELS, frozenset)
        expected_models = {"tiny", "base", "small", "medium", "large", "turbo"}
        assert expected_models.issubset(SUPPORTED_WHISPER_MODELS)

    def test_config_package_exports(self):
        """Validate that config package exports required symbols."""
        assert hasattr(config, "settings")
        assert hasattr(config, "ensure_directories")
        assert hasattr(config, "validate_settings")
        assert hasattr(config, "reload_settings")
        assert config.settings is settings


class TestEnvironmentOverrides:
    """Validate environment variable overriding and reload_settings behavior."""

    def test_override_directories(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Environment variables override directory paths."""
        custom_input = tmp_path / "my_inputs"
        custom_staging = tmp_path / "my_staging"
        custom_output = tmp_path / "my_outputs"

        monkeypatch.setenv("MEDIA_INPUT_DIR", str(custom_input))
        monkeypatch.setenv("STAGING_DIR", str(custom_staging))
        monkeypatch.setenv("OUTPUT_DIR", str(custom_output))

        reload_settings()

        assert settings.MEDIA_INPUT_DIR == custom_input
        assert settings.STAGING_DIR == custom_staging
        assert settings.OUTPUT_DIR == custom_output

    def test_override_models_and_limits(self, monkeypatch: pytest.MonkeyPatch):
        """Environment variables override model identifiers and loop iterations."""
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-pro-exp-02-05")
        monkeypatch.setenv("WHISPER_MODEL", "large")
        monkeypatch.setenv("MAX_LOOP_ITERATIONS", "10")

        reload_settings()

        assert settings.GEMINI_MODEL == "gemini-2.0-pro-exp-02-05"
        assert settings.WHISPER_MODEL == "large"
        assert settings.MAX_LOOP_ITERATIONS == 10

    def test_override_web_server_config(self, monkeypatch: pytest.MonkeyPatch):
        """Environment variables override web server parameters."""
        monkeypatch.setenv("APP_NAME", "videoguru_custom")
        monkeypatch.setenv("WEB_HOST", "0.0.0.0")
        monkeypatch.setenv("WEB_PORT", "9090")
        monkeypatch.setenv("DEFAULT_USER_ID", "creator_123")

        reload_settings()

        assert settings.APP_NAME == "videoguru_custom"
        assert settings.WEB_HOST == "0.0.0.0"
        assert settings.WEB_PORT == 9090
        assert settings.DEFAULT_USER_ID == "creator_123"

    def test_override_gemini_api_key(self, monkeypatch: pytest.MonkeyPatch):
        """Environment variables override GEMINI_API_KEY."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key-12345")
        reload_settings()
        assert settings.GEMINI_API_KEY == "test-secret-key-12345"

    def test_invalid_integer_environment_variables(self, monkeypatch: pytest.MonkeyPatch):
        """Invalid integer strings in env variables fall back safely."""
        monkeypatch.setenv("MAX_LOOP_ITERATIONS", "not_an_int")
        monkeypatch.setenv("WEB_PORT", "invalid_port")

        reload_settings()

        assert settings.MAX_LOOP_ITERATIONS == -1
        assert settings.WEB_PORT == -1
        assert validate_settings() is False

    def test_importlib_reload_re_evaluates_env(self, monkeypatch: pytest.MonkeyPatch):
        """importlib.reload(settings) correctly updates module attributes from environment."""
        monkeypatch.setenv("APP_NAME", "reloaded_app")
        importlib.reload(settings)
        assert settings.APP_NAME == "reloaded_app"


class TestEnsureDirectories:
    """Validate ensure_directories() creates directories properly."""

    def test_ensure_directories_creates_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """ensure_directories() creates MEDIA_INPUT_DIR, STAGING_DIR, and OUTPUT_DIR."""
        custom_input = tmp_path / "new_input_videos"
        custom_staging = tmp_path / "new_staging"
        custom_output = tmp_path / "new_output_video"

        assert not custom_input.exists()
        assert not custom_staging.exists()
        assert not custom_output.exists()

        monkeypatch.setattr(settings, "MEDIA_INPUT_DIR", custom_input)
        monkeypatch.setattr(settings, "STAGING_DIR", custom_staging)
        monkeypatch.setattr(settings, "OUTPUT_DIR", custom_output)

        result = ensure_directories()

        assert custom_input.is_dir()
        assert custom_staging.is_dir()
        assert custom_output.is_dir()
        assert result == (custom_input, custom_staging, custom_output)

    def test_ensure_directories_nested_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """ensure_directories() creates deeply nested paths with parents=True."""
        nested_input = tmp_path / "level1" / "level2" / "inputs"
        nested_staging = tmp_path / "deep" / "nested" / "staging"
        nested_output = tmp_path / "output" / "subfolder" / "videos"

        monkeypatch.setattr(settings, "MEDIA_INPUT_DIR", nested_input)
        monkeypatch.setattr(settings, "STAGING_DIR", nested_staging)
        monkeypatch.setattr(settings, "OUTPUT_DIR", nested_output)

        ensure_directories()

        assert nested_input.is_dir()
        assert nested_staging.is_dir()
        assert nested_output.is_dir()

    def test_ensure_directories_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Calling ensure_directories() multiple times succeeds without error."""
        custom_input = tmp_path / "inputs"
        custom_staging = tmp_path / "staging"
        custom_output = tmp_path / "output"

        monkeypatch.setattr(settings, "MEDIA_INPUT_DIR", custom_input)
        monkeypatch.setattr(settings, "STAGING_DIR", custom_staging)
        monkeypatch.setattr(settings, "OUTPUT_DIR", custom_output)

        # Call twice
        ensure_directories()
        ensure_directories()

        assert custom_input.is_dir()
        assert custom_staging.is_dir()
        assert custom_output.is_dir()


class TestValidateSettings:
    """Validate validate_settings() checks valid and invalid configurations."""

    def test_validate_settings_success(self):
        """Default settings pass validation."""
        assert validate_settings() is True
        assert validate_settings(raise_on_error=True) is True

    def test_validate_settings_require_api_key_when_set(self, monkeypatch: pytest.MonkeyPatch):
        """Validation passes when API key is required and present."""
        monkeypatch.setattr(settings, "GEMINI_API_KEY", "valid_key_123")
        assert validate_settings(require_api_key=True) is True
        assert validate_settings(require_api_key=True, raise_on_error=True) is True

    def test_validate_settings_require_api_key_when_missing(self, monkeypatch: pytest.MonkeyPatch):
        """Validation fails when API key is required but None or empty."""
        monkeypatch.setattr(settings, "GEMINI_API_KEY", None)
        assert validate_settings(require_api_key=True) is False

        with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
            validate_settings(require_api_key=True, raise_on_error=True)

        monkeypatch.setattr(settings, "GEMINI_API_KEY", "   ")
        assert validate_settings(require_api_key=True) is False

    def test_validate_settings_invalid_loop_iterations(self, monkeypatch: pytest.MonkeyPatch):
        """Non-positive or non-int MAX_LOOP_ITERATIONS fails validation."""
        for invalid_val in [0, -1, -10, "5", 3.14, True, False]:
            monkeypatch.setattr(settings, "MAX_LOOP_ITERATIONS", invalid_val)
            assert validate_settings() is False
            with pytest.raises(ValueError, match="MAX_LOOP_ITERATIONS"):
                validate_settings(raise_on_error=True)

    def test_validate_settings_invalid_web_port(self, monkeypatch: pytest.MonkeyPatch):
        """Port outside 1..65535 or non-int fails validation."""
        for invalid_port in [0, -80, 65536, 100000, "8080", True, False]:
            monkeypatch.setattr(settings, "WEB_PORT", invalid_port)
            assert validate_settings() is False
            with pytest.raises(ValueError, match="WEB_PORT"):
                validate_settings(raise_on_error=True)

    def test_validate_settings_empty_strings(self, monkeypatch: pytest.MonkeyPatch):
        """Empty string parameters fail validation."""
        for field in ["APP_NAME", "WEB_HOST", "DEFAULT_USER_ID", "GEMINI_MODEL"]:
            monkeypatch.setattr(settings, field, "")
            assert validate_settings() is False
            with pytest.raises(ValueError, match=field):
                validate_settings(raise_on_error=True)

            monkeypatch.setattr(settings, field, "   ")
            assert validate_settings() is False

            # Reset back to valid
            reload_settings()

    def test_validate_settings_invalid_whisper_model(self, monkeypatch: pytest.MonkeyPatch):
        """Unsupported Whisper model fails validation."""
        monkeypatch.setattr(settings, "WHISPER_MODEL", "super_whisper_ultra")
        assert validate_settings() is False
        with pytest.raises(ValueError, match="WHISPER_MODEL"):
            validate_settings(raise_on_error=True)

    def test_validate_settings_invalid_directory_type(self, monkeypatch: pytest.MonkeyPatch):
        """Directory paths that are not Path objects fail validation."""
        monkeypatch.setattr(settings, "MEDIA_INPUT_DIR", "/not/a/path/object")
        assert validate_settings() is False
        with pytest.raises(ValueError, match="MEDIA_INPUT_DIR must be a pathlib.Path"):
            validate_settings(raise_on_error=True)


class TestDirectoryFallbackBehavior:
    """Validate fallback logic when input_videos or output_video do not exist."""

    def test_legacy_media_dir_fallback(self, tmp_path: Path):
        """_resolve_default_input_dir falls back to 'media' if input_videos is absent."""
        fake_base = tmp_path / "mock_project"
        fake_base.mkdir()
        fake_media = fake_base / "media"
        fake_media.mkdir()

        with patch("config.settings.BASE_DIR", fake_base):
            resolved = _resolve_default_input_dir()
            assert resolved == fake_media

    def test_input_videos_preferred_over_media(self, tmp_path: Path):
        """_resolve_default_input_dir prefers input_videos if it exists."""
        fake_base = tmp_path / "mock_project"
        fake_base.mkdir()
        fake_input_videos = fake_base / "input_videos"
        fake_input_videos.mkdir()
        fake_media = fake_base / "media"
        fake_media.mkdir()

        with patch("config.settings.BASE_DIR", fake_base):
            resolved = _resolve_default_input_dir()
            assert resolved == fake_input_videos

    def test_legacy_output_dir_fallback(self, tmp_path: Path):
        """_resolve_default_output_dir falls back to 'output' if output_video is absent."""
        fake_base = tmp_path / "mock_project"
        fake_base.mkdir()
        fake_output = fake_base / "output"
        fake_output.mkdir()

        with patch("config.settings.BASE_DIR", fake_base):
            resolved = _resolve_default_output_dir()
            assert resolved == fake_output

    def test_output_video_preferred_over_output(self, tmp_path: Path):
        """_resolve_default_output_dir prefers output_video if it exists."""
        fake_base = tmp_path / "mock_project"
        fake_base.mkdir()
        fake_output_video = fake_base / "output_video"
        fake_output_video.mkdir()
        fake_output = fake_base / "output"
        fake_output.mkdir()

        with patch("config.settings.BASE_DIR", fake_base):
            resolved = _resolve_default_output_dir()
            assert resolved == fake_output_video
