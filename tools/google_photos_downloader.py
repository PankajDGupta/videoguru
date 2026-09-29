"""Google Photos downloader ADK tool (SPEC-034).

Provides the ``download_videos_from_google_photos`` ADK tool that VideoGuru's
Ingestion Agent can call to pull today's (or a custom date range's) videos
from Google Photos into the local ``input_videos`` directory before running
the editing pipeline.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from config import settings

import webbrowser

# Import for type reference and testability — actual usage stays inside function
# so startup cost is minimal, but the name is mockable at module level.
try:
    from services.google_photos_client import (  # noqa: F401
        GooglePhotosAuthError,
        GooglePhotosClient,
        GooglePhotosDownloadError,
    )
except ImportError:
    GooglePhotosAuthError = Exception  # type: ignore[assignment,misc]
    GooglePhotosClient = None  # type: ignore[assignment,misc]
    GooglePhotosDownloadError = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — resolve credentials and token paths
# ---------------------------------------------------------------------------

def _resolve_credentials_path() -> Path:
    """Return the ``credentials.json`` path from env or default location.

    Checks (in order):
    1. ``GOOGLE_PHOTOS_CREDENTIALS`` env var.
    2. ``credentials.json`` in the project root (``BASE_DIR``).

    Raises:
        FileNotFoundError: If neither location contains a credentials file.
    """
    env_val = os.getenv("GOOGLE_PHOTOS_CREDENTIALS")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p

    default = settings.BASE_DIR / "credentials.json"
    if default.exists():
        return default

    raise FileNotFoundError(
        "Google Photos credentials.json not found.\n"
        "Download it from Google Cloud Console → APIs & Services → Credentials\n"
        "(choose 'Desktop app' OAuth2 client) and place it at:\n"
        f"  {default}\n"
        "or set the GOOGLE_PHOTOS_CREDENTIALS environment variable."
    )


def _resolve_token_path() -> Path:
    """Return the ``token.json`` cache path from env or default (project root)."""
    env_val = os.getenv("GOOGLE_PHOTOS_TOKEN_FILE")
    if env_val:
        return Path(env_val)
    return settings.BASE_DIR / "token.json"


# ---------------------------------------------------------------------------
# Download result schema
# ---------------------------------------------------------------------------

class DownloadResult:
    """Result of a Google Photos batch download operation.

    Attributes:
        downloaded: Paths of newly-downloaded video files.
        skipped: Number of items skipped (already present on disk).
        failed: Number of items that failed to download.
        errors: Human-readable error messages for each failure.
        dest_dir: The destination directory used.
    """

    def __init__(
        self,
        downloaded: list[Path],
        skipped: int,
        failed: int,
        errors: list[str],
        dest_dir: Path,
    ) -> None:
        self.downloaded = downloaded
        self.skipped = skipped
        self.failed = failed
        self.errors = errors
        self.dest_dir = dest_dir

    @property
    def total_new(self) -> int:
        """Number of newly downloaded files."""
        return len(self.downloaded)

    def summary(self) -> str:
        """Return a compact human-readable summary string."""
        return (
            f"Downloaded {self.total_new} new video(s), "
            f"skipped {self.skipped} (already cached), "
            f"failed {self.failed}."
        )


# ---------------------------------------------------------------------------
# Core download function (used by the ADK tool and the CLI)
# ---------------------------------------------------------------------------

def download_from_google_photos(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    dest_dir: Optional[Path] = None,
    credentials_json: Optional[str | Path] = None,
    token_file: Optional[str | Path] = None,
    force_picker: bool = False,
) -> DownloadResult:
    """Download videos from Google Photos for a given date range.

    If *start_date* and *end_date* are both ``None``, defaults to today.
    If only *start_date* is provided, *end_date* defaults to *start_date*.

    Authentication uses the OAuth2 browser flow on first run; subsequent
    calls silently reuse the cached ``token.json``.

    If direct library search is restricted by Google's API policy (403),
    it automatically launches the interactive Google Photos Picker API session.

    Args:
        start_date: Start of date range (inclusive). Defaults to today.
        end_date: End of date range (inclusive). Defaults to *start_date*.
        dest_dir: Local destination folder. Defaults to
            ``settings.MEDIA_INPUT_DIR`` (``input_videos``).
        credentials_json: Path to OAuth2 ``credentials.json``.
        token_file: Path to cached ``token.json``.
        force_picker: If True, skips direct search and immediately launches
            the Google Photos Picker UI.

    Returns:
        :class:`DownloadResult` with download statistics and file paths.

    Raises:
        FileNotFoundError: If ``credentials.json`` cannot be located.
        ``services.google_photos_client.GooglePhotosAuthError``: On auth failure.
    """
    # Resolve defaults
    today = date.today()
    if start_date is None:
        start_date = today
    if end_date is None:
        end_date = start_date
    if dest_dir is None:
        dest_dir = settings.MEDIA_INPUT_DIR

    creds_path = Path(credentials_json) if credentials_json else _resolve_credentials_path()
    tok_path = Path(token_file) if token_file else _resolve_token_path()

    logger.info(
        "Connecting to Google Photos — date range: %s → %s, dest: %s",
        start_date.isoformat(),
        end_date.isoformat(),
        dest_dir,
    )

    client = GooglePhotosClient.from_credentials_file(
        credentials_json=creds_path,
        token_file=tok_path,
    )

    downloaded: list[Path] = []
    skipped = 0
    failed = 0
    errors: list[str] = []

    def _process_item(item: dict) -> None:
        nonlocal skipped, failed
        media_file = item.get("mediaFile", {})
        filename = media_file.get("filename") or item.get("filename", item.get("id", "video.mp4"))
        try:
            result_path = client.download_video(
                media_item=item,
                dest_dir=dest_dir,
                skip_if_id_cached=True,
            )
            if result_path is None:
                skipped += 1
            else:
                downloaded.append(result_path)
        except GooglePhotosDownloadError as exc:
            failed += 1
            msg = f"Failed to download '{filename}': {exc}"
            errors.append(msg)
            logger.warning(msg)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            msg = f"Unexpected error for '{filename}': {exc}"
            errors.append(msg)
            logger.warning(msg)

    items_to_download: list[dict] = []

    if not force_picker:
        try:
            # Try direct library search first
            for item in client.iter_videos_for_date_range(start_date, end_date):
                items_to_download.append(item)
        except GooglePhotosAuthError as auth_err:
            err_str = str(auth_err)
            if "403" in err_str or "insufficient" in err_str.lower() or "PERMISSION_DENIED" in err_str:
                logger.info(
                    "Photos Library search restricted by Google policy. Launching Google Photos Picker API..."
                )
                force_picker = True
            else:
                raise

    if force_picker:
        print("\n" + "=" * 60)
        print("Google Photos Picker: Interactive Video Selection")
        print("=" * 60)
        print("Google Photos direct library search is restricted by Google's API policy.")
        print("Starting an interactive Google Photos Picker session...")
        session = client.create_picker_session()
        session_id = session["id"]
        picker_uri = session.get("pickerUri", "")
        print("\nOpening Google Photos Picker in your browser...")
        print("👉 Select the video clips you want to use, then click Done.")
        if picker_uri:
            print(f"Picker URL: {picker_uri}\n")
            webbrowser.open(picker_uri)
        print("Waiting for your selection in the browser (up to 5 minutes)...", flush=True)
        completed = client.poll_picker_session(session_id, timeout_seconds=300)
        if completed:
            print("✅ Selection received! Downloading selected videos...\n")
            for item in client.list_picked_media_items(session_id):
                items_to_download.append(item)
            client.delete_picker_session(session_id)
        else:
            print("⚠️ Selection timed out or was cancelled.")
            client.delete_picker_session(session_id)

    for item in items_to_download:
        _process_item(item)

    result = DownloadResult(
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
        errors=errors,
        dest_dir=dest_dir,
    )
    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# ADK Tool
# ---------------------------------------------------------------------------

def download_videos_from_google_photos(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """ADK Tool — Download videos from Google Photos to the local input_videos folder.

    Fetches all videos within the specified date range from the authenticated
    Google Photos account and saves them to the ``input_videos`` directory so
    they are ready for the Ingestion Agent.

    This tool handles OAuth2 authentication automatically:
    - **First run**: Opens a browser consent screen (one-time setup).
    - **Subsequent runs**: Silently uses the cached ``token.json``.

    Args:
        start_date: Start date in ``YYYY-MM-DD`` format.
            Defaults to today's date if omitted.
        end_date: End date in ``YYYY-MM-DD`` format (inclusive).
            Defaults to ``start_date`` if omitted (single-day download).

    Returns:
        dict with keys:
        - ``success`` (bool): True if at least one video was processed.
        - ``downloaded_count`` (int): Number of new videos downloaded.
        - ``skipped_count`` (int): Videos skipped (already on disk).
        - ``failed_count`` (int): Videos that failed to download.
        - ``downloaded_paths`` (list[str]): Absolute paths of new files.
        - ``dest_dir`` (str): Destination directory path.
        - ``summary`` (str): Human-readable summary.
        - ``errors`` (list[str]): Error messages for failed downloads.
    """
    # Parse date strings
    parsed_start: Optional[date] = None
    parsed_end: Optional[date] = None

    if start_date:
        try:
            parsed_start = date.fromisoformat(start_date)
        except ValueError as exc:
            return {
                "success": False,
                "error": f"Invalid start_date format '{start_date}': {exc}. Use YYYY-MM-DD.",
            }

    if end_date:
        try:
            parsed_end = date.fromisoformat(end_date)
        except ValueError as exc:
            return {
                "success": False,
                "error": f"Invalid end_date format '{end_date}': {exc}. Use YYYY-MM-DD.",
            }

    try:
        result = download_from_google_photos(
            start_date=parsed_start,
            end_date=parsed_end,
        )
    except FileNotFoundError as exc:
        return {
            "success": False,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Google Photos download tool raised an unexpected error.")
        return {
            "success": False,
            "error": f"Google Photos download failed: {exc}",
        }

    return {
        "success": result.failed == 0,
        "downloaded_count": result.total_new,
        "skipped_count": result.skipped,
        "failed_count": result.failed,
        "downloaded_paths": [str(p) for p in result.downloaded],
        "dest_dir": str(result.dest_dir),
        "summary": result.summary(),
        "errors": result.errors,
    }
