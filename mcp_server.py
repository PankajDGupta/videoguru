"""VideoGuru MCP server (SPEC-036).

Exposes the VideoGuru pipeline to MCP clients (Claude Code, Claude Desktop) over stdio.

* ``start_pipeline`` runs the full multi-agent edit as a background job; poll it with
  ``get_job_status`` / ``list_jobs``.
* Smaller tools (scan, metadata, manifest, ducking, captions, Google Photos) run inline.

stdout is the MCP transport, so all logging goes to stderr and every tool body runs with
``sys.stdout`` redirected to stderr to swallow stray ``print`` calls from library code.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]

from config import settings

logger = logging.getLogger("videoguru.mcp")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet_stdout():
    """Redirect stdout to stderr so library prints cannot corrupt the MCP stream."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _fail(message: str) -> dict[str, Any]:
    return {"success": False, "error": message}


def _existing_dir(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ValueError(f"Directory not found: {path}")
    return p


def _existing_file(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"File not found: {path}")
    return p


def _safe(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Run *fn* with stdout guarded and convert exceptions into ``{success: False}``."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            with _quiet_stdout():
                return fn(*args, **kwargs)
        except ValueError as exc:
            return _fail(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP tool %s failed", fn.__name__)
            return _fail(f"{type(exc).__name__}: {exc}")

    return wrapper


# ---------------------------------------------------------------------------
# Pipeline job manager
# ---------------------------------------------------------------------------

class JobManager:
    """In-memory registry of background pipeline runs."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def build_initial_state(
        self,
        theme: str,
        input_dir: Path,
        music_file: Optional[str],
        shorts: bool,
        resolution: Optional[str],
        captions: Optional[bool],
        overlay_text: Optional[bool],
        motion_graphics: Optional[bool],
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "theme": theme,
            "media_dir": str(input_dir),
            "auto_approve": True,
        }
        if shorts or (resolution and resolution.strip().lower() == "1080x1920"):
            state["video_type"] = "shorts"
            state["target_resolution"] = "1080x1920"
        elif resolution:
            state["target_resolution"] = resolution
        if captions is not None:
            state["enable_captions"] = captions
        if overlay_text is not None:
            state["enable_overlay_text"] = overlay_text
        if motion_graphics is not None:
            state["enable_motion_graphics"] = motion_graphics
        if music_file:
            state["music_path"] = music_file
        return state

    def start(
        self,
        initial_state: dict[str, Any],
        offline: bool,
        runner: Optional[Callable[..., Any]] = None,
    ) -> str:
        """Create a job and schedule it on the running event loop."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "stage": "queued",
            "theme": initial_state.get("theme"),
            "started_at": time.time(),
            "finished_at": None,
            "result": None,
            "error": None,
        }
        self.jobs[job_id] = job
        self._tasks[job_id] = asyncio.create_task(
            self._run(job, initial_state, offline, runner)
        )
        return job_id

    async def _run(
        self,
        job: dict[str, Any],
        initial_state: dict[str, Any],
        offline: bool,
        runner: Optional[Callable[..., Any]],
    ) -> None:
        def on_stage(name: str) -> None:
            job["stage"] = name

        try:
            if runner is None:
                from agents.root_pipeline import create_root_workflow_agent

                agent = create_root_workflow_agent(offline=offline)
                runner = agent.run_pipeline_async
            with _quiet_stdout():
                res = await runner(
                    user_prompt=initial_state.get("theme") or "",
                    session_id=job["job_id"],
                    initial_state=initial_state,
                    auto_approve=True,
                    on_stage=on_stage,
                )
            job["result"] = {
                "final_video_path": res.get("final_video_path"),
                "otio_file_path": res.get("otio_file_path"),
                "review_status": res.get("review_status"),
                "rendering_complete": res.get("rendering_complete"),
                "clip_count": len(res.get("clip_manifest") or []),
                "edl_entries": len(res.get("edl") or []),
            }
            job["error"] = res.get("error")
            job["status"] = "completed" if res.get("success") else "failed"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline job %s crashed", job["job_id"])
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["stage"] = "done"
            job["finished_at"] = time.time()

    def describe(self, job_id: str) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job is None:
            return _fail(f"Unknown job_id: {job_id}")
        end = job["finished_at"] or time.time()
        return {
            "success": True,
            "job_id": job_id,
            "status": job["status"],
            "stage": job["stage"],
            "theme": job["theme"],
            "elapsed_seconds": round(end - job["started_at"], 1),
            "result": job["result"],
            "error": job["error"],
        }


JOBS = JobManager()


# ---------------------------------------------------------------------------
# Tool implementations (plain functions so they are unit-testable)
# ---------------------------------------------------------------------------

async def start_pipeline_impl(
    theme: str,
    input_dir: str,
    music_file: Optional[str] = None,
    shorts: bool = False,
    resolution: Optional[str] = None,
    captions: Optional[bool] = None,
    overlay_text: Optional[bool] = None,
    motion_graphics: Optional[bool] = None,
    offline: bool = False,
    runner: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    try:
        media_dir = _existing_dir(input_dir)
        if not any(f.suffix.lower() in VIDEO_EXTENSIONS for f in media_dir.rglob("*") if f.is_file()):
            raise ValueError(f"No video files ({', '.join(sorted(VIDEO_EXTENSIONS))}) found in {input_dir}")
        if music_file:
            music_file = str(_existing_file(music_file))
        if not theme.strip():
            raise ValueError("theme must not be empty")
    except ValueError as exc:
        return _fail(str(exc))
    state = JOBS.build_initial_state(
        theme.strip(), media_dir, music_file, shorts, resolution, captions, overlay_text, motion_graphics
    )
    job_id = JOBS.start(state, offline, runner)
    return {"success": True, "job_id": job_id, "status": "running", "output_dir": str(settings.OUTPUT_DIR)}


@_safe
def scan_media_directory(directory: str) -> dict[str, Any]:
    from tools.directory_scanner import scan_local_directory_with_metadata

    files = scan_local_directory_with_metadata(_existing_dir(directory))
    return {"success": True, "count": len(files), "files": [f.model_dump(mode="json") for f in files]}


@_safe
def get_clip_metadata(file_path: str) -> dict[str, Any]:
    from tools.clip_metadata import extract_clip_metadata

    entry = extract_clip_metadata(str(_existing_file(file_path)))
    return {"success": True, "clip": entry.model_dump(mode="json")}


@_safe
def build_manifest(directory: str) -> dict[str, Any]:
    from tools.ingestion_tools import build_clip_manifest

    manifest = build_clip_manifest(_existing_dir(directory))
    return {"success": True, "count": len(manifest), "clips": [c.model_dump(mode="json") for c in manifest]}


@_safe
def duck_audio(
    video_path: str,
    music_path: str,
    output_path: Optional[str] = None,
    music_volume: float = 0.3,
) -> dict[str, Any]:
    from tools.audio_ducking import apply_audio_ducking

    video = _existing_file(video_path)
    music = _existing_file(music_path)
    out = output_path or str(Path(settings.OUTPUT_DIR) / f"{video.stem}_ducked.mp4")
    result = apply_audio_ducking(video_path=video, music_path=music, output_path=out, music_volume=music_volume)
    return {"success": True, "output_path": str(result)}


@_safe
def make_captions(
    video_path: str,
    burn_in: bool = True,
    model_name: Optional[str] = None,
    offline: bool = False,
    shorts: bool = False,
) -> dict[str, Any]:
    from tools.whisper_captioning import generate_captions

    video = _existing_file(video_path)
    out_dir = Path(settings.OUTPUT_DIR)
    result = generate_captions(
        video_path=video,
        output_srt_path=out_dir / f"{video.stem}.srt",
        output_video_path=(out_dir / f"{video.stem}_captioned.mp4") if burn_in else None,
        model_name=model_name,
        offline=offline,
        is_shorts=shorts,
    )
    return {"success": True, **{k: str(v) for k, v in result.items()}}


@_safe
def fetch_google_photos(start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict[str, Any]:
    from tools.google_photos_downloader import download_videos_from_google_photos

    token = Path(settings.GOOGLE_PHOTOS_TOKEN_FILE)
    if not token.is_file():
        return _fail(
            "Google Photos authorization required: no token found at "
            f"{token}. Run `python main.py --fetch-photos` once in a terminal to complete the "
            "OAuth browser flow, then retry."
        )
    return download_videos_from_google_photos(start_date=start_date, end_date=end_date)


def list_outputs_impl() -> dict[str, Any]:
    out = Path(settings.OUTPUT_DIR)
    files = sorted(out.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if out.is_dir() else []
    return {
        "success": True,
        "output_dir": str(out),
        "files": [{"path": str(f), "size_bytes": f.stat().st_size} for f in files],
    }


# ---------------------------------------------------------------------------
# Server assembly
# ---------------------------------------------------------------------------

def create_server() -> Any:
    server = _Server(
        "videoguru",
        instructions=(
            "VideoGuru edits raw footage into a finished video. Call start_pipeline with a theme and a "
            "directory of clips, then poll get_job_status until status is 'completed'; the result contains "
            "final_video_path. Runs can take several minutes."
        ),
    )

    @server.tool()
    async def start_pipeline(
        theme: str,
        input_dir: str,
        music_file: Optional[str] = None,
        shorts: bool = False,
        resolution: Optional[str] = None,
        captions: Optional[bool] = None,
        overlay_text: Optional[bool] = None,
        motion_graphics: Optional[bool] = None,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Start the full VideoGuru edit pipeline in the background and return a job_id.

        Args:
            theme: Creative theme/prompt for the edit (e.g. "Goa beach trip highlights").
            input_dir: Absolute path to a directory of .mp4/.mov/.avi/.mkv clips.
            music_file: Optional background music path (auto-ducked under speech).
            shorts: Render vertical 1080x1920 YouTube Shorts.
            resolution: Optional WxH override, e.g. "1920x1080".
            captions: Burn in Whisper captions (default per settings).
            overlay_text: Enable overlay text (default per settings).
            motion_graphics: Enable motion graphics (default per settings).
            offline: Use deterministic heuristics instead of Gemini.
        """
        return await start_pipeline_impl(
            theme, input_dir, music_file, shorts, resolution, captions, overlay_text, motion_graphics, offline
        )

    @server.tool()
    def get_job_status(job_id: str) -> dict[str, Any]:
        """Get status, current stage and (when finished) result paths of a pipeline job."""
        return JOBS.describe(job_id)

    @server.tool()
    def list_jobs() -> dict[str, Any]:
        """List all pipeline jobs started in this server session."""
        return {"success": True, "jobs": [JOBS.describe(j) for j in JOBS.jobs]}

    @server.tool()
    def list_outputs() -> dict[str, Any]:
        """List rendered .mp4 files in the VideoGuru output directory, newest first."""
        return list_outputs_impl()

    server.tool(name="scan_media_directory")(scan_media_directory)
    server.tool(name="get_clip_metadata")(get_clip_metadata)
    server.tool(name="build_clip_manifest")(build_manifest)
    server.tool(name="apply_audio_ducking")(duck_audio)
    server.tool(name="generate_captions")(make_captions)
    server.tool(name="download_from_google_photos")(fetch_google_photos)
    return server


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from services.observability import configure_logging

    configure_logging()
    settings.ensure_directories()
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
