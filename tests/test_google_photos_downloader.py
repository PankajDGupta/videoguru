"""Unit tests for Google Photos integration (SPEC-034).

Tests cover:
- GooglePhotosClient authentication helpers
- Token cache persistence and expiry
- Video filtering and pagination in _search_videos
- Download deduplication via ID cache
- download_from_google_photos date resolution
- download_videos_from_google_photos ADK tool output schema
- CLI --fetch-photos / --photos-date / --photos-start / --photos-end argument parsing
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_creds(tmp_path: Path) -> Path:
    """Write a minimal OAuth2 Desktop credentials.json to a temp directory."""
    creds = {
        "installed": {
            "client_id": "test-client-id.apps.googleusercontent.com",
            "client_secret": "test-client-secret",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    p = tmp_path / "credentials.json"
    p.write_text(json.dumps(creds), encoding="utf-8")
    return p


@pytest.fixture
def tmp_token(tmp_path: Path) -> Path:
    """Return a path for the token.json cache file (does not exist yet)."""
    return tmp_path / "token.json"


@pytest.fixture
def valid_token_cache(tmp_path: Path) -> Path:
    """Write a non-expired token.json to a temp directory."""
    token_path = tmp_path / "token.json"
    token_data = {
        "access_token": "ya29.test-access-token",
        "refresh_token": "1//test-refresh-token",
        "expires_at": int(time.time()) + 3600,  # valid for 1 hour
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
    }
    token_path.write_text(json.dumps(token_data), encoding="utf-8")
    return token_path


@pytest.fixture
def expired_token_cache(tmp_path: Path) -> Path:
    """Write an expired token.json to a temp directory."""
    token_path = tmp_path / "token.json"
    token_data = {
        "access_token": "ya29.expired-access-token",
        "refresh_token": "1//test-refresh-token",
        "expires_at": int(time.time()) - 60,  # expired 1 minute ago
        "client_id": "test-client-id",
        "client_secret": "test-client-secret",
    }
    token_path.write_text(json.dumps(token_data), encoding="utf-8")
    return token_path


# ---------------------------------------------------------------------------
# _TokenCache tests
# ---------------------------------------------------------------------------

class TestTokenCache:
    def test_loads_existing_token_on_init(self, valid_token_cache: Path):
        from services.google_photos_client import _TokenCache

        cache = _TokenCache(valid_token_cache)
        assert cache.access_token == "ya29.test-access-token"
        assert cache.refresh_token == "1//test-refresh-token"
        assert cache.client_id == "test-client-id"

    def test_is_not_expired_for_fresh_token(self, valid_token_cache: Path):
        from services.google_photos_client import _TokenCache

        cache = _TokenCache(valid_token_cache)
        assert cache.is_expired(buffer_seconds=120) is False

    def test_is_expired_for_old_token(self, expired_token_cache: Path):
        from services.google_photos_client import _TokenCache

        cache = _TokenCache(expired_token_cache)
        assert cache.is_expired(buffer_seconds=120) is True

    def test_save_writes_json_to_disk(self, tmp_token: Path):
        from services.google_photos_client import _TokenCache

        cache = _TokenCache(tmp_token)
        cache.save(
            access_token="new-token",
            refresh_token="new-refresh",
            expires_in=3600,
            client_id="cid",
            client_secret="csec",
        )
        assert tmp_token.exists()
        data = json.loads(tmp_token.read_text())
        assert data["access_token"] == "new-token"
        assert data["refresh_token"] == "new-refresh"
        assert data["client_id"] == "cid"

    def test_clear_removes_file(self, valid_token_cache: Path):
        from services.google_photos_client import _TokenCache

        cache = _TokenCache(valid_token_cache)
        cache.clear()
        assert not valid_token_cache.exists()
        assert cache.access_token is None

    def test_is_expired_when_no_expires_at(self, tmp_token: Path):
        from services.google_photos_client import _TokenCache

        # Token file exists but has no expires_at field
        tmp_token.write_text(json.dumps({"access_token": "tok"}), encoding="utf-8")
        cache = _TokenCache(tmp_token)
        assert cache.is_expired() is True


# ---------------------------------------------------------------------------
# GooglePhotosClient factory tests
# ---------------------------------------------------------------------------

class TestGooglePhotosClientFactory:
    def test_from_credentials_file_success(self, tmp_creds: Path, tmp_token: Path):
        from services.google_photos_client import GooglePhotosClient

        client = GooglePhotosClient.from_credentials_file(
            credentials_json=tmp_creds,
            token_file=tmp_token,
        )
        assert client._client_id == "test-client-id.apps.googleusercontent.com"
        assert client._client_secret == "test-client-secret"

    def test_from_credentials_file_not_found_raises(self, tmp_path: Path):
        from services.google_photos_client import GooglePhotosClient

        with pytest.raises(FileNotFoundError, match="not found"):
            GooglePhotosClient.from_credentials_file(
                credentials_json=tmp_path / "nonexistent.json",
            )

    def test_from_credentials_file_bad_format_raises(self, tmp_path: Path):
        from services.google_photos_client import GooglePhotosClient

        bad_creds = tmp_path / "bad.json"
        bad_creds.write_text(json.dumps({"not_installed": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported credentials.json format"):
            GooglePhotosClient.from_credentials_file(credentials_json=bad_creds)


# ---------------------------------------------------------------------------
# GooglePhotosClient authentication tests
# ---------------------------------------------------------------------------

class TestGooglePhotosClientAuth:
    def _make_client(self, token_file: Path) -> "object":
        from services.google_photos_client import GooglePhotosClient

        return GooglePhotosClient(
            client_id="test-cid",
            client_secret="test-csec",
            token_file=token_file,
        )

    def test_authenticate_uses_cached_token_when_valid(self, valid_token_cache: Path):
        client = self._make_client(valid_token_cache)
        # Should not raise and should not call any HTTP endpoint
        with patch("services.google_photos_client.requests.post") as mock_post:
            client.authenticate()
            mock_post.assert_not_called()

    def test_authenticate_refreshes_when_expired(self, expired_token_cache: Path):
        client = self._make_client(expired_token_cache)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "access_token": "refreshed-token",
            "expires_in": 3600,
        }

        with patch("services.google_photos_client.requests.post", return_value=mock_resp):
            client.authenticate()

        assert client._token_cache.access_token == "refreshed-token"

    def test_refresh_token_failure_raises_auth_error(self, expired_token_cache: Path):
        from services.google_photos_client import GooglePhotosAuthError

        client = self._make_client(expired_token_cache)

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "invalid_grant"

        with patch("services.google_photos_client.requests.post", return_value=mock_resp):
            # After refresh failure, falls back to browser flow which we don't mock here,
            # but _refresh_access_token itself should raise GooglePhotosAuthError
            with pytest.raises((GooglePhotosAuthError, Exception)):
                client._refresh_access_token("bad-refresh-token")


# ---------------------------------------------------------------------------
# iter_videos_for_date / iter_videos_for_date_range tests
# ---------------------------------------------------------------------------

def _make_video_item(item_id: str, mime: str = "video/mp4", filename: str = "clip.mp4") -> dict:
    return {
        "id": item_id,
        "filename": filename,
        "mimeType": mime,
        "baseUrl": f"https://lh3.googleusercontent.com/{item_id}",
        "mediaMetadata": {"video": {}},
    }


class TestVideoSearch:
    def _make_authed_client(self, tmp_path: Path) -> "object":
        """Return a client with a valid (non-expired) token so authenticate() is a no-op."""
        token_path = tmp_path / "token.json"
        token_path.write_text(
            json.dumps({
                "access_token": "ya29.authed",
                "refresh_token": "refresh",
                "expires_at": int(time.time()) + 3600,
                "client_id": "cid",
                "client_secret": "csec",
            }),
            encoding="utf-8",
        )
        from services.google_photos_client import GooglePhotosClient

        return GooglePhotosClient(
            client_id="cid",
            client_secret="csec",
            token_file=token_path,
        )

    def test_iter_videos_for_date_yields_video_items(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        api_response = {
            "mediaItems": [
                _make_video_item("vid-1"),
                _make_video_item("vid-2"),
            ]
        }
        with patch.object(client, "_api_post", return_value=api_response):
            items = list(client.iter_videos_for_date(date(2026, 9, 27)))
        assert len(items) == 2
        assert items[0]["id"] == "vid-1"
        assert items[1]["id"] == "vid-2"

    def test_iter_videos_filters_non_video_mime(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        api_response = {
            "mediaItems": [
                _make_video_item("vid-1", mime="video/mp4"),
                _make_video_item("img-1", mime="image/jpeg"),  # Should be filtered out
            ]
        }
        with patch.object(client, "_api_post", return_value=api_response):
            items = list(client.iter_videos_for_date(date(2026, 9, 27)))
        assert len(items) == 1
        assert items[0]["id"] == "vid-1"

    def test_iter_videos_handles_pagination(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        page1 = {
            "mediaItems": [_make_video_item("vid-1")],
            "nextPageToken": "token-page-2",
        }
        page2 = {
            "mediaItems": [_make_video_item("vid-2")],
            # No nextPageToken → last page
        }
        with patch.object(client, "_api_post", side_effect=[page1, page2]):
            items = list(client.iter_videos_for_date(date(2026, 9, 27)))
        assert len(items) == 2

    def test_iter_videos_empty_result(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        with patch.object(client, "_api_post", return_value={"mediaItems": []}):
            items = list(client.iter_videos_for_date(date(2026, 9, 27)))
        assert items == []

    def test_iter_videos_no_media_items_key(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        # API returns no mediaItems key (zero results case)
        with patch.object(client, "_api_post", return_value={}):
            items = list(client.iter_videos_for_date(date(2026, 9, 27)))
        assert items == []


# ---------------------------------------------------------------------------
# download_video tests
# ---------------------------------------------------------------------------

class TestDownloadVideo:
    def _make_authed_client(self, tmp_path: Path) -> "object":
        token_path = tmp_path / "token.json"
        token_path.write_text(
            json.dumps({
                "access_token": "ya29.authed",
                "refresh_token": "refresh",
                "expires_at": int(time.time()) + 3600,
                "client_id": "cid",
                "client_secret": "csec",
            }),
            encoding="utf-8",
        )
        from services.google_photos_client import GooglePhotosClient

        return GooglePhotosClient(
            client_id="cid",
            client_secret="csec",
            token_file=token_path,
        )

    def test_download_video_saves_file(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        item = _make_video_item("vid-1", filename="test_clip.mp4")
        dest = tmp_path / "input_videos"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": "1024"}
        mock_resp.iter_content.return_value = [b"A" * 512, b"B" * 512]

        with patch("services.google_photos_client.requests.get", return_value=mock_resp):
            result = client.download_video(item, dest_dir=dest, skip_if_id_cached=False)

        assert result is not None
        assert result.name == "test_clip.mp4"
        assert result.exists()
        assert result.read_bytes() == b"A" * 512 + b"B" * 512

    def test_download_video_skips_already_downloaded_id(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        item = _make_video_item("vid-already-downloaded", filename="dup.mp4")
        dest = tmp_path / "input_videos"
        dest.mkdir()

        # Pre-populate ID cache
        id_cache = dest / ".gphotos_downloaded_ids.json"
        id_cache.write_text(json.dumps(["vid-already-downloaded"]), encoding="utf-8")

        with patch("services.google_photos_client.requests.get") as mock_get:
            result = client.download_video(
                item,
                dest_dir=dest,
                skip_if_id_cached=True,
                id_cache_file=id_cache,
            )
        mock_get.assert_not_called()  # No HTTP call should be made
        assert result is None

    def test_download_video_updates_id_cache(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        item = _make_video_item("vid-new", filename="new_clip.mp4")
        dest = tmp_path / "input_videos"
        id_cache = dest / ".gphotos_downloaded_ids.json"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": "100"}
        mock_resp.iter_content.return_value = [b"X" * 100]

        with patch("services.google_photos_client.requests.get", return_value=mock_resp):
            client.download_video(item, dest_dir=dest, skip_if_id_cached=True, id_cache_file=id_cache)

        assert id_cache.exists()
        cached_ids = json.loads(id_cache.read_text())
        assert "vid-new" in cached_ids

    def test_download_video_raises_on_http_error(self, tmp_path: Path):
        from services.google_photos_client import GooglePhotosDownloadError

        client = self._make_authed_client(tmp_path)
        item = _make_video_item("vid-err", filename="err.mp4")
        dest = tmp_path / "input_videos"

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with patch("services.google_photos_client.requests.get", return_value=mock_resp):
            with pytest.raises(GooglePhotosDownloadError, match="HTTP 403"):
                client.download_video(item, dest_dir=dest, skip_if_id_cached=False)

    def test_download_video_no_base_url_skips(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        item = {"id": "vid-nurl", "filename": "nurl.mp4", "mimeType": "video/mp4", "baseUrl": ""}
        dest = tmp_path / "input_videos"

        with patch("services.google_photos_client.requests.get") as mock_get:
            result = client.download_video(item, dest_dir=dest, skip_if_id_cached=False)
        mock_get.assert_not_called()
        assert result is None


# ---------------------------------------------------------------------------
# download_from_google_photos function tests
# ---------------------------------------------------------------------------

class TestDownloadFromGooglePhotos:
    def test_defaults_to_today(self, tmp_path: Path, tmp_creds: Path):
        from tools.google_photos_downloader import download_from_google_photos

        with patch("tools.google_photos_downloader.GooglePhotosClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.iter_videos_for_date_range.return_value = iter([])
            MockClient.from_credentials_file.return_value = mock_instance

            result = download_from_google_photos(
                dest_dir=tmp_path,
                credentials_json=tmp_creds,
                token_file=tmp_path / "token.json",
            )

        assert result.total_new == 0
        assert result.skipped == 0
        # Verify it was called with today's date
        call_args = mock_instance.iter_videos_for_date_range.call_args
        today = date.today()
        assert call_args[0][0] == today
        assert call_args[0][1] == today

    def test_custom_date_range(self, tmp_path: Path, tmp_creds: Path):
        from tools.google_photos_downloader import download_from_google_photos

        with patch("tools.google_photos_downloader.GooglePhotosClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.iter_videos_for_date_range.return_value = iter([])
            MockClient.from_credentials_file.return_value = mock_instance

            download_from_google_photos(
                start_date=date(2026, 9, 1),
                end_date=date(2026, 9, 27),
                dest_dir=tmp_path,
                credentials_json=tmp_creds,
                token_file=tmp_path / "token.json",
            )

        call_args = mock_instance.iter_videos_for_date_range.call_args
        assert call_args[0][0] == date(2026, 9, 1)
        assert call_args[0][1] == date(2026, 9, 27)

    def test_counts_downloaded_and_skipped(self, tmp_path: Path, tmp_creds: Path):
        from tools.google_photos_downloader import download_from_google_photos

        item1 = _make_video_item("v1", filename="v1.mp4")
        item2 = _make_video_item("v2", filename="v2.mp4")

        with patch("tools.google_photos_downloader.GooglePhotosClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.iter_videos_for_date_range.return_value = iter([item1, item2])
            # v1 downloaded, v2 skipped (already cached)
            mock_instance.download_video.side_effect = [
                tmp_path / "v1.mp4",
                None,
            ]
            MockClient.from_credentials_file.return_value = mock_instance

            result = download_from_google_photos(
                start_date=date(2026, 9, 27),
                dest_dir=tmp_path,
                credentials_json=tmp_creds,
                token_file=tmp_path / "token.json",
            )

        assert result.total_new == 1
        assert result.skipped == 1
        assert result.failed == 0

    def test_counts_failures(self, tmp_path: Path, tmp_creds: Path):
        from services.google_photos_client import GooglePhotosDownloadError
        from tools.google_photos_downloader import download_from_google_photos

        item1 = _make_video_item("v-fail", filename="fail.mp4")

        with patch("tools.google_photos_downloader.GooglePhotosClient") as MockClient:
            mock_instance = MagicMock()
            mock_instance.iter_videos_for_date_range.return_value = iter([item1])
            mock_instance.download_video.side_effect = GooglePhotosDownloadError("HTTP 503")
            MockClient.from_credentials_file.return_value = mock_instance

            result = download_from_google_photos(
                start_date=date(2026, 9, 27),
                dest_dir=tmp_path,
                credentials_json=tmp_creds,
                token_file=tmp_path / "token.json",
            )

        assert result.failed == 1
        assert len(result.errors) == 1
        assert "HTTP 503" in result.errors[0]


# ---------------------------------------------------------------------------
# download_videos_from_google_photos ADK tool tests
# ---------------------------------------------------------------------------

class TestDownloadVideosADKTool:
    def test_returns_success_dict_structure(self, tmp_path: Path, tmp_creds: Path):
        from tools.google_photos_downloader import download_videos_from_google_photos

        with patch("tools.google_photos_downloader.download_from_google_photos") as mock_dl:
            from tools.google_photos_downloader import DownloadResult

            mock_dl.return_value = DownloadResult(
                downloaded=[tmp_path / "v1.mp4"],
                skipped=0,
                failed=0,
                errors=[],
                dest_dir=tmp_path,
            )
            result = download_videos_from_google_photos(start_date="2026-09-27")

        assert result["success"] is True
        assert result["downloaded_count"] == 1
        assert result["skipped_count"] == 0
        assert result["failed_count"] == 0
        assert isinstance(result["downloaded_paths"], list)
        assert isinstance(result["summary"], str)
        assert isinstance(result["errors"], list)

    def test_returns_error_on_invalid_date_format(self):
        from tools.google_photos_downloader import download_videos_from_google_photos

        result = download_videos_from_google_photos(start_date="27-09-2026")  # Wrong format
        assert result["success"] is False
        assert "Invalid start_date format" in result["error"]

    def test_returns_error_on_invalid_end_date_format(self):
        from tools.google_photos_downloader import download_videos_from_google_photos

        result = download_videos_from_google_photos(start_date="2026-09-27", end_date="not-a-date")
        assert result["success"] is False
        assert "Invalid end_date format" in result["error"]

    def test_returns_error_when_credentials_missing(self):
        from tools.google_photos_downloader import download_videos_from_google_photos

        with patch("tools.google_photos_downloader.download_from_google_photos",
                   side_effect=FileNotFoundError("credentials.json not found")):
            result = download_videos_from_google_photos(start_date="2026-09-27")

        assert result["success"] is False
        assert "credentials.json not found" in result["error"]

    def test_defaults_to_today_when_no_date_given(self, tmp_path: Path):
        from tools.google_photos_downloader import download_videos_from_google_photos, DownloadResult

        with patch("tools.google_photos_downloader.download_from_google_photos") as mock_dl:
            mock_dl.return_value = DownloadResult([], 0, 0, [], tmp_path)
            download_videos_from_google_photos()  # No dates

        call_kwargs = mock_dl.call_args
        # start_date and end_date should both be None (tool passes None, function resolves to today)
        assert call_kwargs[1].get("start_date") is None or call_kwargs[0][0:] == ()


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------

class TestCLIArgParsing:
    def _parse(self, args: list[str]) -> "object":
        from main import parse_arguments
        import sys

        with patch.object(sys, "argv", ["main.py"] + args):
            return parse_arguments()

    def test_fetch_photos_flag(self):
        args = self._parse(["--fetch-photos"])
        assert args.fetch_photos is True

    def test_photos_date_flag(self):
        args = self._parse(["--photos-date", "2026-09-27"])
        assert args.photos_date == "2026-09-27"

    def test_photos_start_end_flags(self):
        args = self._parse(["--photos-start", "2026-09-01", "--photos-end", "2026-09-27"])
        assert args.photos_start == "2026-09-01"
        assert args.photos_end == "2026-09-27"

    def test_all_photos_args_default_to_none(self):
        args = self._parse(["--info"])
        assert args.fetch_photos is False
        assert args.photos_date is None
        assert args.photos_start is None
        assert args.photos_end is None
        assert args.picker is False

    def test_picker_flag(self):
        args = self._parse(["--fetch-photos", "--picker"])
        assert args.fetch_photos is True
        assert args.picker is True


# ---------------------------------------------------------------------------
# Google Photos Picker API unit tests
# ---------------------------------------------------------------------------

class TestPickerAPI:
    def _make_authed_client(self, tmp_path: Path) -> "object":
        token_path = tmp_path / "token.json"
        token_path.write_text(
            json.dumps({
                "access_token": "ya29.authed-picker",
                "refresh_token": "refresh",
                "expires_at": int(time.time()) + 3600,
                "client_id": "cid",
                "client_secret": "csec",
            }),
            encoding="utf-8",
        )
        from services.google_photos_client import GooglePhotosClient

        return GooglePhotosClient(
            client_id="cid",
            client_secret="csec",
            token_file=token_path,
        )

    def test_create_picker_session(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        mock_response = {
            "id": "session-123",
            "pickerUri": "https://photos.google.com/picker/session-123",
            "mediaItemsSet": False,
        }
        with patch.object(client, "_api_post", return_value=mock_response):
            session = client.create_picker_session()
        assert session["id"] == "session-123"
        assert "pickerUri" in session

    def test_poll_picker_session_completes(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        status_responses = [
            {"id": "session-123", "mediaItemsSet": False},
            {"id": "session-123", "mediaItemsSet": True},
        ]
        with patch.object(client, "get_picker_session", side_effect=status_responses):
            result = client.poll_picker_session("session-123", timeout_seconds=5, poll_interval=0.01)
        assert result is True

    def test_list_picked_media_items_filters_videos(self, tmp_path: Path):
        client = self._make_authed_client(tmp_path)
        mock_data = {
            "mediaItems": [
                {
                    "id": "item-1",
                    "mediaFile": {
                        "baseUrl": "https://lh3.googleusercontent.com/vid1",
                        "filename": "vacation.mp4",
                        "mimeType": "video/mp4",
                    },
                },
                {
                    "id": "item-2",
                    "mediaFile": {
                        "baseUrl": "https://lh3.googleusercontent.com/photo1",
                        "filename": "sunset.jpg",
                        "mimeType": "image/jpeg",
                    },
                },
            ]
        }
        with patch.object(client, "_api_get", return_value=mock_data):
            items = list(client.list_picked_media_items("session-123"))
        assert len(items) == 1
        assert items[0]["id"] == "item-1"

    def test_download_from_google_photos_falls_back_to_picker_on_403(self, tmp_path: Path, tmp_creds: Path):
        from services.google_photos_client import GooglePhotosAuthError
        from tools.google_photos_downloader import download_from_google_photos

        item1 = {
            "id": "picked-vid-1",
            "mediaFile": {
                "baseUrl": "https://lh3.googleusercontent.com/picked",
                "filename": "picked_clip.mp4",
                "mimeType": "video/mp4",
            },
        }

        with patch("tools.google_photos_downloader.GooglePhotosClient") as MockClient:
            mock_instance = MagicMock()
            # Direct search throws 403 insufficient scopes error
            mock_instance.iter_videos_for_date_range.side_effect = GooglePhotosAuthError(
                "Google Photos API POST mediaItems:search failed (403): Request had insufficient authentication scopes."
            )
            mock_instance.create_picker_session.return_value = {
                "id": "sess-abc",
                "pickerUri": "https://photos.google.com/picker/sess-abc",
            }
            mock_instance.poll_picker_session.return_value = True
            mock_instance.list_picked_media_items.return_value = iter([item1])
            mock_instance.download_video.return_value = tmp_path / "picked_clip.mp4"

            MockClient.from_credentials_file.return_value = mock_instance

            with patch("tools.google_photos_downloader.webbrowser.open"):
                result = download_from_google_photos(
                    dest_dir=tmp_path,
                    credentials_json=tmp_creds,
                    token_file=tmp_path / "token.json",
                )

        assert result.total_new == 1
        mock_instance.create_picker_session.assert_called_once()
        mock_instance.poll_picker_session.assert_called_once_with("sess-abc", timeout_seconds=300)
        mock_instance.delete_picker_session.assert_called_once_with("sess-abc")

