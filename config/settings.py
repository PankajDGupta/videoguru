"""VideoGuru Settings and Configuration."""

import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_INPUT_DIR = Path(os.getenv("MEDIA_INPUT_DIR", BASE_DIR / "media"))
STAGING_DIR = Path(os.getenv("STAGING_DIR", BASE_DIR / "staging"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))

# Models & Algorithms
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
MAX_LOOP_ITERATIONS = int(os.getenv("MAX_LOOP_ITERATIONS", "5"))
