"""Tests for the VideoGuru MCP server (SPEC-036)."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

import mcp_server
from config import settings

EXPECTED_TOOLS = {
    "start_pipeline",
    "get_job_status",
    "list_jobs",
    "list_outputs",
    "scan_media_directory",
    "get_clip_metadata",
    "build_clip_manifest",
    "apply_audio_ducking",
    "generate_captions",
    "download_from_google_photos",
}


def _make_clip(path: Path, color: str = "blue") -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d=3:r=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def clips_dir(tmp_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    d = tmp_path / "clips"
    d.mkdir()
    _make_clip(d / "a.mp4", "blue")
    _make_clip(d / "b.mp4", "red")
    return d


@pytest.fixture(autouse=True)
def fresh_jobs():
    mcp_server.JOBS.jobs.clear()
    mcp_server.JOBS._tasks.clear()
    yield


def test_tools_registered():
    server = mcp_server.create_server()
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS
    start = next(t for t in tools if t.name == "start_pipeline")
    assert {"theme", "input_dir"} <= set(start.input_schema["properties"])
    assert set(start.input_schema["required"]) == {"theme", "input_dir"}


def test_build_initial_state_shorts_and_flags(tmp_path):
    state = mcp_server.JOBS.build_initial_state(
        "Trip", tmp_path, "song.mp3", True, None, True, False, False
    )
    assert state["video_type"] == "shorts"
    assert state["target_resolution"] == "1080x1920"
    assert state["enable_captions"] is True
    assert state["enable_overlay_text"] is False
    assert state["enable_motion_graphics"] is False
    assert state["music_path"] == "song.mp3"
    assert state["auto_approve"] is True


def test_build_initial_state_omits_unset_flags(tmp_path):
    state = mcp_server.JOBS.build_initial_state("Trip", tmp_path, None, False, "1280x720", None, None, None)
    assert state["target_resolution"] == "1280x720"
    for key in ("enable_captions", "enable_overlay_text", "enable_motion_graphics", "music_path", "video_type"):
        assert key not in state


def test_start_pipeline_rejects_bad_input(tmp_path):
    res = asyncio.run(mcp_server.start_pipeline_impl("x", str(tmp_path / "missing")))
    assert res["success"] is False and "Directory not found" in res["error"]

    empty = tmp_path / "empty"
    empty.mkdir()
    res = asyncio.run(mcp_server.start_pipeline_impl("x", str(empty)))
    assert res["success"] is False and "No video files" in res["error"]

    (empty / "a.mp4").write_bytes(b"x")
    res = asyncio.run(mcp_server.start_pipeline_impl("  ", str(empty)))
    assert res["success"] is False and "theme" in res["error"]

    res = asyncio.run(mcp_server.start_pipeline_impl("x", str(empty), music_file=str(tmp_path / "nope.mp3")))
    assert res["success"] is False and "File not found" in res["error"]


def test_job_lifecycle_with_fake_runner(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")

    async def fake_runner(*, on_stage, initial_state, session_id, **_):
        on_stage("IngestionAgent")
        await asyncio.sleep(0.05)
        return {
            "success": True,
            "final_video_path": "out.mp4",
            "otio_file_path": "out.otio",
            "review_status": "approved",
            "rendering_complete": True,
            "clip_manifest": [1, 2],
            "edl": [1],
        }

    async def scenario():
        res = await mcp_server.start_pipeline_impl("Theme", str(tmp_path), runner=fake_runner)
        assert res["success"] and res["status"] == "running"
        job_id = res["job_id"]
        await asyncio.sleep(0.01)
        assert mcp_server.JOBS.describe(job_id)["status"] == "running"
        await mcp_server.JOBS._tasks[job_id]
        return mcp_server.JOBS.describe(job_id)

    final = asyncio.run(scenario())
    assert final["status"] == "completed"
    assert final["stage"] == "done"
    assert final["result"]["final_video_path"] == "out.mp4"
    assert final["result"]["clip_count"] == 2


def test_job_failure_is_captured(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")

    async def boom(**_):
        raise RuntimeError("kaput")

    async def scenario():
        res = await mcp_server.start_pipeline_impl("Theme", str(tmp_path), runner=boom)
        await mcp_server.JOBS._tasks[res["job_id"]]
        return mcp_server.JOBS.describe(res["job_id"])

    final = asyncio.run(scenario())
    assert final["status"] == "failed"
    assert "kaput" in final["error"]


def test_unknown_job():
    assert mcp_server.JOBS.describe("nope")["success"] is False


def test_safe_wrapper_keeps_stdout_clean(capsys):
    @mcp_server._safe
    def noisy() -> dict:
        print("stray library output")
        return {"success": True}

    assert noisy() == {"success": True}
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "stray library output" in captured.err


def test_safe_wrapper_converts_errors():
    @mcp_server._safe
    def bad() -> dict:
        raise RuntimeError("bad")

    res = bad()
    assert res["success"] is False and "RuntimeError" in res["error"]


def test_scan_and_manifest_tools(clips_dir):
    scan = mcp_server.scan_media_directory(str(clips_dir))
    assert scan["success"] and scan["count"] == 2

    manifest = mcp_server.build_manifest(str(clips_dir))
    assert manifest["success"] and manifest["count"] == 2

    meta = mcp_server.get_clip_metadata(str(clips_dir / "a.mp4"))
    assert meta["success"]


def test_file_tools_report_missing_paths(tmp_path):
    assert mcp_server.scan_media_directory(str(tmp_path / "x"))["success"] is False
    assert mcp_server.get_clip_metadata(str(tmp_path / "x.mp4"))["success"] is False
    assert mcp_server.duck_audio(str(tmp_path / "v.mp4"), str(tmp_path / "m.mp3"))["success"] is False


def test_google_photos_requires_token(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "GOOGLE_PHOTOS_TOKEN_FILE", str(tmp_path / "missing_token.json"))
    res = mcp_server.fetch_google_photos("2026-01-01", "2026-01-02")
    assert res["success"] is False
    assert "authorization required" in res["error"]


def test_list_outputs(tmp_path, monkeypatch):
    (tmp_path / "one.mp4").write_bytes(b"abc")
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path)
    res = mcp_server.list_outputs_impl()
    assert res["success"] and [Path(f["path"]).name for f in res["files"]] == ["one.mp4"]


def test_real_offline_pipeline_job(clips_dir, tmp_path, monkeypatch):
    """End-to-end: offline pipeline via the job manager on synthetic clips."""
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "out").mkdir()

    async def scenario():
        res = await mcp_server.start_pipeline_impl(
            "Test reel", str(clips_dir), offline=True,
            captions=False, overlay_text=False, motion_graphics=False,
        )
        assert res["success"], res
        await asyncio.wait_for(mcp_server.JOBS._tasks[res["job_id"]], timeout=300)
        return mcp_server.JOBS.describe(res["job_id"])

    final = asyncio.run(scenario())
    assert final["status"] == "completed", final
    video = final["result"]["final_video_path"]
    assert video and Path(video).is_file()
