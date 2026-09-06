"""VideoGuru Settings and Configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Load local environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

MEDIA_INPUT_DIR = Path(os.getenv("MEDIA_INPUT_DIR", BASE_DIR / "media"))
STAGING_DIR = Path(os.getenv("STAGING_DIR", BASE_DIR / "staging"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))

# Models, API Keys & Algorithms
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
MAX_LOOP_ITERATIONS = int(os.getenv("MAX_LOOP_ITERATIONS", "5"))

# Application & Web Server Configuration
APP_NAME = os.getenv("APP_NAME", "videoguru")
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "videoguru_user")

