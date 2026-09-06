"""VideoGuru Settings and Configuration (SPEC-027).

Centralizes all application settings, directory paths, model identifiers,
and web service parameters with support for environment overrides via .env.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local environment variables from .env if present
load_dotenv(BASE_DIR / ".env")


def _resolve_default_input_dir() -> Path:
    """Resolve default input directory, falling back to legacy 'media' if input_videos does not exist."""
    input_videos_dir = BASE_DIR / "input_videos"
    legacy_media_dir = BASE_DIR / "media"
    if not input_videos_dir.exists() and legacy_media_dir.exists():
        return legacy_media_dir
    return input_videos_dir


def _resolve_default_output_dir() -> Path:
    """Resolve default output directory, falling back to legacy 'output' if output_video does not exist."""
    output_video_dir = BASE_DIR / "output_video"
    legacy_output_dir = BASE_DIR / "output"
    if not output_video_dir.exists() and legacy_output_dir.exists():
        return legacy_output_dir
    return output_video_dir


# Primary Directory Paths
MEDIA_INPUT_DIR = Path(os.getenv("MEDIA_INPUT_DIR", str(_resolve_default_input_dir())))
STAGING_DIR = Path(os.getenv("STAGING_DIR", str(BASE_DIR / "staging")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(_resolve_default_output_dir())))

# Models, API Keys & Algorithms
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")

try:
    MAX_LOOP_ITERATIONS = int(os.getenv("MAX_LOOP_ITERATIONS", "5"))
except (ValueError, TypeError):
    MAX_LOOP_ITERATIONS = 5

# Application & Web Server Configuration
APP_NAME = os.getenv("APP_NAME", "videoguru")
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")

try:
    WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
except (ValueError, TypeError):
    WEB_PORT = 8000

DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "videoguru_user")

# Recognized / Supported Whisper Models
SUPPORTED_WHISPER_MODELS: frozenset[str] = frozenset({
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "turbo",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
})


def ensure_directories() -> tuple[Path, Path, Path]:
    """Create MEDIA_INPUT_DIR, STAGING_DIR, and OUTPUT_DIR if they do not exist.

    Returns:
        tuple[Path, Path, Path]: Tuple containing (MEDIA_INPUT_DIR, STAGING_DIR, OUTPUT_DIR).
    """
    MEDIA_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (MEDIA_INPUT_DIR, STAGING_DIR, OUTPUT_DIR)


def validate_settings(
    require_api_key: bool = False,
    raise_on_error: bool = False,
) -> bool:
    """Validate environment and configuration sanity.

    Checks:
    - MEDIA_INPUT_DIR, STAGING_DIR, OUTPUT_DIR are valid Path instances.
    - MAX_LOOP_ITERATIONS is an integer >= 1.
    - WEB_PORT is an integer between 1 and 65535.
    - WEB_HOST, APP_NAME, DEFAULT_USER_ID, GEMINI_MODEL are non-empty strings.
    - WHISPER_MODEL is a recognized Whisper model identifier.
    - If require_api_key is True, GEMINI_API_KEY must be set and non-empty.

    Args:
        require_api_key: Whether to strictly require GEMINI_API_KEY to be non-empty.
        raise_on_error: If True, raises ValueError with failure details. If False, returns False.

    Returns:
        True if all configuration settings are valid, False otherwise.

    Raises:
        ValueError: If raise_on_error is True and any validation check fails.
    """
    errors: list[str] = []

    # Check directory path objects
    for name, path_val in [
        ("MEDIA_INPUT_DIR", MEDIA_INPUT_DIR),
        ("STAGING_DIR", STAGING_DIR),
        ("OUTPUT_DIR", OUTPUT_DIR),
    ]:
        if not isinstance(path_val, Path):
            errors.append(f"{name} must be a pathlib.Path instance, got {type(path_val).__name__}")

    # Check loop iterations (note: bool is a subclass of int in Python)
    if not isinstance(MAX_LOOP_ITERATIONS, int) or isinstance(MAX_LOOP_ITERATIONS, bool) or MAX_LOOP_ITERATIONS < 1:
        errors.append(f"MAX_LOOP_ITERATIONS must be an integer >= 1, got {MAX_LOOP_ITERATIONS!r}")

    # Check web port
    if not isinstance(WEB_PORT, int) or isinstance(WEB_PORT, bool) or not (1 <= WEB_PORT <= 65535):
        errors.append(f"WEB_PORT must be an integer between 1 and 65535, got {WEB_PORT!r}")

    # Check non-empty strings
    for name, val in [
        ("APP_NAME", APP_NAME),
        ("WEB_HOST", WEB_HOST),
        ("DEFAULT_USER_ID", DEFAULT_USER_ID),
        ("GEMINI_MODEL", GEMINI_MODEL),
    ]:
        if not isinstance(val, str) or not val.strip():
            errors.append(f"{name} must be a non-empty string, got {val!r}")

    # Check whisper model
    if not isinstance(WHISPER_MODEL, str) or WHISPER_MODEL.strip().lower() not in SUPPORTED_WHISPER_MODELS:
        errors.append(
            f"WHISPER_MODEL must be one of {sorted(SUPPORTED_WHISPER_MODELS)}, got {WHISPER_MODEL!r}"
        )

    # Check GEMINI_API_KEY if required
    if require_api_key:
        if not GEMINI_API_KEY or not str(GEMINI_API_KEY).strip():
            errors.append("GEMINI_API_KEY is required but missing or empty.")

    if errors:
        error_msg = f"Configuration validation failed: {'; '.join(errors)}"
        logger.warning(error_msg)
        if raise_on_error:
            raise ValueError(error_msg)
        return False

    return True


def reload_settings(env_file: Optional[Union[str, Path]] = None) -> None:
    """Reload all configuration settings from the current environment variables.

    Args:
        env_file: Optional path to a .env file to load before reloading.
    """
    global MEDIA_INPUT_DIR, STAGING_DIR, OUTPUT_DIR
    global GEMINI_API_KEY, GEMINI_MODEL, WHISPER_MODEL, MAX_LOOP_ITERATIONS
    global APP_NAME, WEB_HOST, WEB_PORT, DEFAULT_USER_ID

    if env_file:
        load_dotenv(env_file, override=True)

    MEDIA_INPUT_DIR = Path(os.getenv("MEDIA_INPUT_DIR", str(_resolve_default_input_dir())))
    STAGING_DIR = Path(os.getenv("STAGING_DIR", str(BASE_DIR / "staging")))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(_resolve_default_output_dir())))

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")

    try:
        MAX_LOOP_ITERATIONS = int(os.getenv("MAX_LOOP_ITERATIONS", "5"))
    except (ValueError, TypeError):
        MAX_LOOP_ITERATIONS = -1

    APP_NAME = os.getenv("APP_NAME", "videoguru")
    WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")

    try:
        WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
    except (ValueError, TypeError):
        WEB_PORT = -1

    DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "videoguru_user")


