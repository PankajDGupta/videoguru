"""Google Photos API client for VideoGuru (SPEC-034).

Handles OAuth 2.0 authentication and video download from Google Photos.
Uses the official Google Photos Library REST API v1 with token caching.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import requests

logger = logging.getLogger(__name__)

# Google Photos Library API v1 base URL
_GPHOTOS_BASE_URL = "https://photoslibrary.googleapis.com/v1"
# Google Photos Picker API v1 base URL (Google's official replacement since March 2025)
_PICKER_BASE_URL = "https://photospicker.googleapis.com/v1"

# OAuth2 token and Google API endpoints
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_AUTH_URL = "https://accounts.google.com/o/oauth2/auth"
_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

# Required OAuth2 scopes
_SCOPES = [
    "https://www.googleapis.com/auth/photospicker.mediaitems.readonly",
    "https://www.googleapis.com/auth/photoslibrary.readonly",
]

# Video MIME types to select
_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",   # .mov
    "video/x-msvideo",  # .avi
    "video/x-matroska", # .mkv
    "video/3gpp",
    "video/3gpp2",
    "video/webm",
    "video/mpeg",
}

# Download suffix: =dv fetches the video bytes; =d fetches images
_VIDEO_DOWNLOAD_SUFFIX = "=dv"

# Maximum Google Photos API page size
_MAX_PAGE_SIZE = 100


class GooglePhotosAuthError(Exception):
    """Raised when OAuth2 authentication or token refresh fails."""


class GooglePhotosDownloadError(Exception):
    """Raised when a video download fails."""


class _TokenCache:
    """Manages the OAuth2 token lifecycle with file persistence.

    Stores ``access_token``, ``refresh_token``, ``expires_at`` (epoch seconds),
    and ``client_id`` / ``client_secret`` in a JSON file so the browser
    authorization flow only runs once per device.
    """

    def __init__(self, token_file: Path) -> None:
        self._file = token_file
        self._data: dict = {}
        if token_file.exists():
            try:
                self._data = json.loads(token_file.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse token cache %s: %s", token_file, exc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def access_token(self) -> Optional[str]:
        return self._data.get("access_token")

    @property
    def refresh_token(self) -> Optional[str]:
        return self._data.get("refresh_token")

    @property
    def client_id(self) -> Optional[str]:
        return self._data.get("client_id")

    @property
    def client_secret(self) -> Optional[str]:
        return self._data.get("client_secret")

    def is_expired(self, buffer_seconds: int = 120) -> bool:
        """Return True if the access token will expire within *buffer_seconds*."""
        expires_at = self._data.get("expires_at")
        if expires_at is None:
            return True
        return time.time() >= (expires_at - buffer_seconds)

    def save(
        self,
        *,
        access_token: str,
        refresh_token: Optional[str],
        expires_in: int,
        client_id: str,
        client_secret: str,
    ) -> None:
        """Persist a new token set to disk."""
        self._data.update(
            access_token=access_token,
            expires_at=int(time.time()) + expires_in,
            client_id=client_id,
            client_secret=client_secret,
        )
        if refresh_token:
            self._data["refresh_token"] = refresh_token
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        logger.debug("Token cache written to %s", self._file)

    def clear(self) -> None:
        """Delete the cached token file and clear in-memory state."""
        self._data = {}
        if self._file.exists():
            self._file.unlink()
            logger.info("Token cache cleared: %s", self._file)


class GooglePhotosClient:
    """Authenticated client for Google Photos Library API v1.

    Usage::

        client = GooglePhotosClient.from_credentials_file(
            credentials_json="credentials.json",
            token_file="token.json",
        )
        for item in client.iter_videos_for_date(date(2026, 9, 27)):
            client.download_video(item, dest_dir=Path("input_videos"))

    The first call opens a browser for the OAuth2 consent flow.
    Subsequent calls reuse the cached ``token.json``.

    Args:
        client_id: OAuth2 client ID from Google Cloud Console.
        client_secret: OAuth2 client secret.
        token_file: Path for caching the access/refresh token.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_file: Path,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_cache = _TokenCache(token_file)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_credentials_file(
        cls,
        credentials_json: str | Path,
        token_file: str | Path | None = None,
    ) -> "GooglePhotosClient":
        """Construct a client from a Google Cloud OAuth2 credentials JSON file.

        The ``credentials_json`` file must be a *Desktop* or *Installed* OAuth2
        client credentials file downloaded from the Google Cloud Console.

        Args:
            credentials_json: Path to the ``credentials.json`` file.
            token_file: Path for persisting the OAuth2 token.
                Defaults to ``token.json`` alongside ``credentials_json``.

        Returns:
            Authenticated GooglePhotosClient instance.

        Raises:
            FileNotFoundError: If ``credentials_json`` does not exist.
            ValueError: If the JSON structure is not a desktop/installed app credential.
        """
        creds_path = Path(credentials_json)
        if not creds_path.exists():
            raise FileNotFoundError(
                f"Google OAuth2 credentials file not found: {creds_path}\n"
                "Download it from Google Cloud Console → APIs & Services → Credentials."
            )

        raw = json.loads(creds_path.read_text(encoding="utf-8"))
        # Credentials JSON wraps content under 'installed' or 'web' key
        cred_data = raw.get("installed") or raw.get("web")
        if not cred_data:
            raise ValueError(
                "Unsupported credentials.json format. Expected 'installed' or 'web' app type."
            )

        client_id = cred_data["client_id"]
        client_secret = cred_data["client_secret"]

        if token_file is None:
            token_file = creds_path.parent / "token.json"

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            token_file=Path(token_file),
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self) -> None:
        """Ensure a valid access token is available.

        - If a cached, non-expired token exists, does nothing.
        - If the token is expired and a refresh_token is cached, silently refreshes.
        - Otherwise, opens the browser OAuth2 consent flow.

        Raises:
            GooglePhotosAuthError: If authentication or token refresh fails.
        """
        cache = self._token_cache

        # Reuse existing non-expired token
        if cache.access_token and not cache.is_expired():
            logger.debug("Using cached Google Photos access token (still valid).")
            return

        # Attempt silent refresh
        if cache.refresh_token:
            logger.info("Access token expired. Attempting silent refresh…")
            try:
                self._refresh_access_token(cache.refresh_token)
                return
            except GooglePhotosAuthError as exc:
                logger.warning("Silent token refresh failed (%s). Falling back to browser flow.", exc)

        # Full browser-based OAuth2 flow
        self._run_browser_auth_flow()

    def _refresh_access_token(self, refresh_token: str) -> None:
        """Use the refresh_token to obtain a new access_token without browser interaction."""
        resp = requests.post(
            _TOKEN_URL,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            raise GooglePhotosAuthError(
                f"Token refresh failed ({resp.status_code}): {resp.text}"
            )
        token_data = resp.json()
        self._token_cache.save(
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token", refresh_token),
            expires_in=token_data.get("expires_in", 3600),
            client_id=self._client_id,
            client_secret=self._client_secret,
        )
        logger.info("Google Photos access token refreshed successfully.")

    def _run_browser_auth_flow(self) -> None:
        """Execute the full OAuth2 Authorization Code flow via the system browser.

        Opens the Google consent URL in the default browser, then starts a
        local HTTP server on port 8080 to capture the redirect code.

        Raises:
            GooglePhotosAuthError: If the authorization code exchange fails.
        """
        import socket
        import urllib.parse
        import webbrowser
        from http.server import BaseHTTPRequestHandler, HTTPServer

        redirect_uri = "http://localhost:8080"
        auth_params = {
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

        print("\n" + "=" * 60)
        print("Google Photos OAuth2 Authorization Required")
        print("=" * 60)
        print("Opening your browser to authorize VideoGuru…")
        print(f"\nIf the browser does not open, visit:\n  {auth_url}\n")
        webbrowser.open(auth_url)

        # Capture the authorization code via a temporary local HTTP server
        auth_code: list[str] = []

        class _OAuthHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                if "code" in params:
                    auth_code.append(params["code"][0])
                self.send_response(200)
                self.end_headers()
                self.wfile.write(
                    b"<html><body><h2>Authorization successful!</h2>"
                    b"<p>You can close this tab and return to VideoGuru.</p>"
                    b"</body></html>"
                )

            def log_message(self, *args):  # suppress noisy HTTP logs
                pass

        server = HTTPServer(("localhost", 8080), _OAuthHandler)
        server.timeout = 120  # Wait up to 2 minutes
        server.handle_request()  # Block until one request is received

        if not auth_code:
            raise GooglePhotosAuthError(
                "No authorization code received. Did you approve the Google Photos consent screen?"
            )

        # Exchange code for tokens
        resp = requests.post(
            _TOKEN_URL,
            data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "code": auth_code[0],
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            raise GooglePhotosAuthError(
                f"Authorization code exchange failed ({resp.status_code}): {resp.text}"
            )
        token_data = resp.json()
        self._token_cache.save(
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            expires_in=token_data.get("expires_in", 3600),
            client_id=self._client_id,
            client_secret=self._client_secret,
        )
        print("✅ Google Photos authorization successful! Token saved.\n")
        logger.info("Google Photos OAuth2 browser flow completed and token cached.")

    # ------------------------------------------------------------------
    # Internal API helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return Authorization headers for Google Photos API requests."""
        self.authenticate()
        return {"Authorization": f"Bearer {self._token_cache.access_token}"}

    def _api_get(self, path: str, params: Optional[dict] = None, base_url: str = _GPHOTOS_BASE_URL) -> dict:
        """Perform an authenticated GET against the Google Photos API."""
        url = f"{base_url}/{path}"
        resp = requests.get(url, headers=self._auth_headers(), params=params, timeout=30)
        if resp.status_code == 401:
            # Token may have just expired mid-session; force refresh once
            logger.info("Got 401, forcing token refresh and retrying…")
            self._token_cache._data.pop("expires_at", None)
            self.authenticate()
            resp = requests.get(url, headers=self._auth_headers(), params=params, timeout=30)
        if not resp.ok:
            raise GooglePhotosAuthError(
                f"Google Photos API GET {path} failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()

    def _api_post(self, path: str, json_body: dict, base_url: str = _GPHOTOS_BASE_URL) -> dict:
        """Perform an authenticated POST against the Google Photos API."""
        url = f"{base_url}/{path}"
        resp = requests.post(url, headers=self._auth_headers(), json=json_body, timeout=30)
        if resp.status_code == 401:
            logger.info("Got 401, forcing token refresh and retrying…")
            self._token_cache._data.pop("expires_at", None)
            self.authenticate()
            resp = requests.post(url, headers=self._auth_headers(), json=json_body, timeout=30)
        if not resp.ok:
            raise GooglePhotosAuthError(
                f"Google Photos API POST {path} failed ({resp.status_code}): {resp.text}"
            )
        return resp.json()

    def _api_delete(self, path: str, base_url: str = _GPHOTOS_BASE_URL) -> dict:
        """Perform an authenticated DELETE against the Google Photos API."""
        url = f"{base_url}/{path}"
        resp = requests.delete(url, headers=self._auth_headers(), timeout=30)
        if resp.status_code == 401:
            self._token_cache._data.pop("expires_at", None)
            self.authenticate()
            resp = requests.delete(url, headers=self._auth_headers(), timeout=30)
        if not resp.ok:
            logger.debug("Google Photos API DELETE %s returned %s: %s", path, resp.status_code, resp.text)
        return resp.json() if resp.text else {}

    # ------------------------------------------------------------------
    # Media item iteration
    # ------------------------------------------------------------------

    def iter_videos_for_date(self, target_date: date) -> Iterator[dict]:
        """Yield all video media items created on *target_date* (local calendar date).

        Uses the ``mediaItems:search`` endpoint with a date filter and
        ``mediaTypeFilter=VIDEO``.  Handles pagination automatically.

        Args:
            target_date: Calendar date to filter by (e.g. ``date.today()``).

        Yields:
            Raw Google Photos ``mediaItem`` dicts containing at least:
            ``id``, ``filename``, ``mimeType``, ``baseUrl``,
            ``mediaMetadata.video``.
        """
        date_filter = {
            "ranges": [{
                "startDate": {
                    "year": target_date.year,
                    "month": target_date.month,
                    "day": target_date.day,
                },
                "endDate": {
                    "year": target_date.year,
                    "month": target_date.month,
                    "day": target_date.day,
                },
            }]
        }
        yield from self._search_videos(date_filter)

    def iter_videos_for_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> Iterator[dict]:
        """Yield all video media items created between *start_date* and *end_date* inclusive.

        Args:
            start_date: Start of date range.
            end_date: End of date range (inclusive).

        Yields:
            Raw Google Photos ``mediaItem`` dicts.
        """
        date_filter = {
            "ranges": [{
                "startDate": {
                    "year": start_date.year,
                    "month": start_date.month,
                    "day": start_date.day,
                },
                "endDate": {
                    "year": end_date.year,
                    "month": end_date.month,
                    "day": end_date.day,
                },
            }]
        }
        yield from self._search_videos(date_filter)

    def _search_videos(self, date_filter: dict) -> Iterator[dict]:
        """Internal method — paginate mediaItems:search and yield video items."""
        page_token: Optional[str] = None
        page = 0

        while True:
            body: dict = {
                "pageSize": _MAX_PAGE_SIZE,
                "filters": {
                    "dateFilter": date_filter,
                    "mediaTypeFilter": {"mediaTypes": ["VIDEO"]},
                },
            }
            if page_token:
                body["pageToken"] = page_token

            data = self._api_post("mediaItems:search", body)
            items = data.get("mediaItems", [])
            page += 1
            logger.debug("Page %d: received %d media items from Google Photos.", page, len(items))

            for item in items:
                mime = item.get("mimeType", "")
                if mime in _VIDEO_MIME_TYPES or mime.startswith("video/"):
                    yield item

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    # ------------------------------------------------------------------
    # Google Photos Picker API (Google's official replacement since March 2025)
    # ------------------------------------------------------------------

    def create_picker_session(self) -> dict:
        """Create a new Photos Picker session.

        Returns:
            Dict containing session 'id' and 'pickerUri'.
        """
        return self._api_post("sessions", {}, base_url=_PICKER_BASE_URL)

    def get_picker_session(self, session_id: str) -> dict:
        """Fetch current status of a Photos Picker session."""
        return self._api_get(f"sessions/{session_id}", base_url=_PICKER_BASE_URL)

    def poll_picker_session(
        self,
        session_id: str,
        timeout_seconds: int = 300,
        poll_interval: int = 2,
    ) -> bool:
        """Wait for the user to select media items in the Picker UI.

        Args:
            session_id: The session ID returned by create_picker_session().
            timeout_seconds: Max seconds to wait before timing out (default 5 min).
            poll_interval: Seconds between status polls (default 2s).

        Returns:
            True if user completed selection, False if timed out.
        """
        start = time.time()
        while time.time() - start < timeout_seconds:
            data = self.get_picker_session(session_id)
            if data.get("mediaItemsSet"):
                return True
            time.sleep(poll_interval)
        return False

    def list_picked_media_items(self, session_id: str) -> Iterator[dict]:
        """Yield all video media items selected in a completed Picker session."""
        page_token: Optional[str] = None
        while True:
            params: dict = {"sessionId": session_id, "pageSize": 100}
            if page_token:
                params["pageToken"] = page_token
            data = self._api_get("mediaItems", params=params, base_url=_PICKER_BASE_URL)
            items = data.get("mediaItems", [])
            for item in items:
                media_file = item.get("mediaFile", {})
                mime = media_file.get("mimeType", item.get("mimeType", ""))
                filename = media_file.get("filename", item.get("filename", ""))
                is_video = (
                    mime in _VIDEO_MIME_TYPES
                    or mime.startswith("video/")
                    or any(filename.lower().endswith(f".{ext}") for ext in ["mp4", "mov", "avi", "mkv", "webm", "3gp"])
                )
                if is_video:
                    yield item
            page_token = data.get("nextPageToken")
            if not page_token:
                break

    def delete_picker_session(self, session_id: str) -> None:
        """Delete a completed Picker session to release resources."""
        try:
            self._api_delete(f"sessions/{session_id}", base_url=_PICKER_BASE_URL)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to delete picker session %s: %s", session_id, exc)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_video(
        self,
        media_item: dict,
        dest_dir: Path,
        *,
        skip_if_id_cached: bool = True,
        id_cache_file: Optional[Path] = None,
        chunk_size: int = 1024 * 1024,
    ) -> Optional[Path]:
        """Download a single video media item to *dest_dir*.

        Supports both the legacy Library API mediaItem schema and the
        modern Photos Picker API PickedMediaItem schema.

        Args:
            media_item: A ``mediaItem`` dict from the Google Photos API or Picker API.
            dest_dir: Destination folder (created if absent).
            skip_if_id_cached: If True, skips download when the ``media_item['id']``
                is already present in *id_cache_file*.
            id_cache_file: JSON file that persists the set of downloaded media item IDs.
                Defaults to ``dest_dir/.gphotos_downloaded_ids.json``.
            chunk_size: Streaming download chunk size in bytes.

        Returns:
            The ``Path`` of the saved file, or ``None`` if the item was skipped.

        Raises:
            GooglePhotosDownloadError: If the download stream returns a non-200 status.
        """
        media_file = media_item.get("mediaFile", {})
        item_id: str = media_item["id"]
        filename: str = media_file.get("filename") or media_item.get("filename", f"{item_id}.mp4")
        base_url: str = media_file.get("baseUrl") or media_item.get("baseUrl", "")

        if not base_url:
            logger.warning("Media item %s has no baseUrl — skipping.", item_id)
            return None

        # Resolve ID cache
        if id_cache_file is None:
            id_cache_file = dest_dir / ".gphotos_downloaded_ids.json"

        downloaded_ids: set[str] = set()
        if skip_if_id_cached and id_cache_file.exists():
            try:
                downloaded_ids = set(
                    json.loads(id_cache_file.read_text(encoding="utf-8"))
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read ID cache %s: %s", id_cache_file, exc)

        if skip_if_id_cached and item_id in downloaded_ids:
            logger.info("Skipping already-downloaded media item: %s (%s)", filename, item_id)
            return None

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / filename

        # Append =dv to request the full-resolution video bytes
        download_url = f"{base_url}{_VIDEO_DOWNLOAD_SUFFIX}"
        logger.info("Downloading: %s → %s", filename, dest_path)

        # Include Authorization header (required by Picker API, compatible with Library API)
        headers = {}
        if self._token_cache.access_token:
            headers["Authorization"] = f"Bearer {self._token_cache.access_token}"

        resp = requests.get(download_url, headers=headers, stream=True, timeout=60)
        if resp.status_code != 200:
            raise GooglePhotosDownloadError(
                f"Failed to download '{filename}' (HTTP {resp.status_code}): {resp.text[:200]}"
            )

        total_bytes = int(resp.headers.get("content-length", 0))
        written = 0
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)

        size_mb = written / (1024 * 1024)
        logger.info("✅ Downloaded '%s' (%.2f MB) → %s", filename, size_mb, dest_path)

        # Update ID cache
        downloaded_ids.add(item_id)
        try:
            id_cache_file.write_text(
                json.dumps(sorted(downloaded_ids), indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update ID cache: %s", exc)

        return dest_path

    def revoke_token(self) -> None:
        """Revoke the stored access token and clear the local cache."""
        token = self._token_cache.access_token
        if token:
            try:
                requests.post(_REVOKE_URL, params={"token": token}, timeout=10)
                logger.info("Google Photos access token revoked.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Token revocation request failed: %s", exc)
        self._token_cache.clear()
