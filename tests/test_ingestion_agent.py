"""Unit and integration tests for SPEC-006: Ingestion Agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import pytest

from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

from agents import (
    INGESTION_INSTRUCTION,
    IngestionAgent,
    StubIngestionAgent,
    create_ingestion_agent,
    create_root_greeter_agent,
    ingestion_agent,
)
from config import settings
from main import main
from schemas.media import ClipManifestEntry
from tools.clip_metadata import extract_clip_metadata
from tools.directory_scanner import scan_local_directory
from tools.ingestion_tools import (
    build_clip_manifest,
    get_clip_manifest_from_state,
    ingest_media_directory,
)


@pytest.fixture(scope="module")
def sample_media_dir(tmp_path_factory):
    """Generate a temporary directory containing synthetic test video clips and non-video files."""
    media_dir = tmp_path_factory.mktemp("ingest_media")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("ffmpeg binary not found on PATH")

    # Clip 1: 720p 30fps with audio (1.5s)
    clip1 = media_dir / "vlog_intro.mp4"
    cmd1 = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1.5:size=1280x720:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=1000:duration=1.5",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        str(clip1),
    ]
    subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Clip 2: 1080p 24fps silent (1.0s)
    clip2 = media_dir / "drone_shot.mov"
    cmd2 = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1.0:size=1920x1080:rate=24",
        "-c:v",
        "libx264",
        "-an",
        "-y",
        str(clip2),
    ]
    subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Non-video files
    (media_dir / "notes.txt").write_text("Shoot notes from the day", encoding="utf-8")
    (media_dir / "thumbnail.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

    return media_dir


class TestBuildClipManifest:
    """Test suite for build_clip_manifest function."""

    def test_build_manifest_discovers_all_clips(self, sample_media_dir):
        manifest = build_clip_manifest(sample_media_dir)
        assert len(manifest) == 2
        file_names = {entry.file_name for entry in manifest}
        assert file_names == {"vlog_intro.mp4", "drone_shot.mov"}

        for entry in manifest:
            assert isinstance(entry, ClipManifestEntry)
            assert entry.clip_id.startswith("vid_")
            assert Path(entry.absolute_path).exists()
            assert entry.duration_seconds > 0
            assert entry.frame_rate > 0
            assert "x" in entry.resolution

    def test_build_manifest_deterministic_sorting(self, sample_media_dir):
        manifest1 = build_clip_manifest(sample_media_dir)
        manifest2 = build_clip_manifest(sample_media_dir)
        assert [e.absolute_path for e in manifest1] == [e.absolute_path for e in manifest2]

    def test_empty_directory_returns_empty_manifest(self, tmp_path):
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        manifest = build_clip_manifest(empty_dir)
        assert manifest == []

    def test_non_existent_directory_raises_not_found(self, tmp_path):
        missing = tmp_path / "missing_dir"
        with pytest.raises(FileNotFoundError):
            build_clip_manifest(missing)

    def test_file_passed_raises_not_a_directory(self, sample_media_dir):
        single_file = sample_media_dir / "vlog_intro.mp4"
        with pytest.raises(NotADirectoryError):
            build_clip_manifest(single_file)

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty string"):
            build_clip_manifest("")
        with pytest.raises(ValueError, match="non-empty string"):
            build_clip_manifest("   ")


from unittest.mock import MagicMock
from google.adk.events import Event, EventActions


def _create_mock_tool_context(initial_state: dict[str, Any] | None = None) -> ToolContext:
    actions = EventActions()
    mock_invocation = MagicMock()
    mock_invocation.session.state = initial_state if initial_state is not None else {}
    return ToolContext(invocation_context=mock_invocation, event_actions=actions)


class TestIngestMediaDirectoryTool:
    """Test suite for ingest_media_directory ADK tool."""

    def test_ingest_tool_updates_tool_context_state(self, sample_media_dir):
        tool_ctx = _create_mock_tool_context({})

        result_msg = ingest_media_directory(
            directory_path=str(sample_media_dir),
            tool_context=tool_ctx,
        )
        assert "[IngestionAgent] Successfully ingested 2 clip(s)" in result_msg
        assert "Total Duration:" in result_msg
        assert "vlog_intro.mp4" in result_msg
        assert "drone_shot.mov" in result_msg

        # Verify state persistence
        assert "clip_manifest" in tool_ctx.state
        assert len(tool_ctx.state["clip_manifest"]) == 2
        assert tool_ctx.state["media_dir"] == str(sample_media_dir.resolve())

        # Verify items are serializable dicts
        first_clip = tool_ctx.state["clip_manifest"][0]
        assert isinstance(first_clip, dict)
        assert "clip_id" in first_clip
        assert "resolution" in first_clip
        assert "duration_seconds" in first_clip

    def test_ingest_tool_uses_context_state_media_dir_fallback(self, sample_media_dir):
        tool_ctx = _create_mock_tool_context({"media_dir": str(sample_media_dir)})

        result_msg = ingest_media_directory(
            directory_path=None,
            tool_context=tool_ctx,
        )
        assert "Successfully ingested 2 clip(s)" in result_msg

    def test_ingest_tool_empty_dir_report(self, tmp_path):
        empty_dir = tmp_path / "empty_media"
        empty_dir.mkdir()
        tool_ctx = _create_mock_tool_context({})

        result_msg = ingest_media_directory(
            directory_path=str(empty_dir),
            tool_context=tool_ctx,
        )
        assert "No supported video clips found" in result_msg
        assert tool_ctx.state["clip_manifest"] == []



class TestGetClipManifestFromState:
    """Test parsing and reconstruction of ClipManifestEntry list from session state."""

    def test_get_manifest_from_valid_state(self):
        state = {
            "clip_manifest": [
                {
                    "clip_id": "vid_abc12345",
                    "absolute_path": "/media/clip1.mp4",
                    "file_name": "clip1.mp4",
                    "duration_seconds": 12.5,
                    "frame_rate": 30.0,
                    "resolution": "1920x1080",
                    "width": 1920,
                    "height": 1080,
                    "video_codec": "h264",
                    "has_audio": True,
                    "audio_codec": "aac",
                }
            ]
        }
        manifest = get_clip_manifest_from_state(state)
        assert len(manifest) == 1
        assert isinstance(manifest[0], ClipManifestEntry)
        assert manifest[0].clip_id == "vid_abc12345"
        assert manifest[0].resolution == "1920x1080"

    def test_get_manifest_from_empty_or_missing_state(self):
        assert get_clip_manifest_from_state({}) == []
        assert get_clip_manifest_from_state({"other_key": 123}) == []
        assert get_clip_manifest_from_state({"clip_manifest": None}) == []

    def test_get_manifest_with_existing_models(self):
        entry = ClipManifestEntry(
            clip_id="vid_premodel1",
            absolute_path="/media/pre.mp4",
            file_name="pre.mp4",
            duration_seconds=5.0,
            frame_rate=24.0,
            resolution="1280x720",
            width=1280,
            height=720,
            video_codec="h264",
        )
        state = {"clip_manifest": [entry]}
        manifest = get_clip_manifest_from_state(state)
        assert len(manifest) == 1
        assert manifest[0] is entry


class TestIngestionAgentDefinition:
    """Test IngestionAgent class properties, instructions, and tools."""

    def test_agent_attributes(self):
        agent = create_ingestion_agent()
        assert agent.name == "ingestion_agent"
        assert agent.model == settings.GEMINI_MODEL
        assert "Media Ingestion" in agent.instruction
        assert "Clip Manifest" in agent.instruction
        assert StubIngestionAgent is IngestionAgent

    def test_agent_tools_bound(self):
        agent = create_ingestion_agent()
        tool_names = [getattr(t, "__name__", str(t)) for t in agent.tools]
        assert "scan_local_directory" in tool_names
        assert "extract_clip_metadata" in tool_names
        assert "ingest_media_directory" in tool_names


class TestIngestionAgentRunnerTurn:
    """Integration test running IngestionAgent with ADK Runner and InMemorySessionService."""

    def test_offline_ingestion_turn_with_media_dir(self, sample_media_dir):
        async def _test():
            service = InMemorySessionService()
            user_id = "creator_ingest_01"
            session_id = "sess_ingest_01"

            # Create session with theme and media_dir in state
            await service.create_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state={
                    "theme": "Rocky Mountain Exploration",
                    "media_dir": str(sample_media_dir),
                },
            )

            agent = create_ingestion_agent(offline=True)
            runner = Runner(
                app_name=settings.APP_NAME,
                agent=agent,
                session_service=service,
            )

            events: list[Event] = []
            async for ev in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(parts=[types.Part.from_text(text="Please ingest the media")]),
            ):
                events.append(ev)

            assert len(events) >= 1
            response_text = events[0].content.parts[0].text
            assert "Rocky Mountain Exploration" in response_text
            assert "Successfully ingested 2 clip(s)" in response_text

            # Verify session state was updated with clip_manifest
            sess = await service.get_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            manifest = get_clip_manifest_from_state(sess.state)
            assert len(manifest) == 2
            assert all(isinstance(e, ClipManifestEntry) for e in manifest)

        asyncio.run(_test())

    def test_root_greeter_to_ingestion_agent_sequential_flow(self, sample_media_dir):
        """Verify sequential handoff from RootGreeterAgent to IngestionAgent."""
        async def _test():
            service = InMemorySessionService()
            user_id = "creator_seq_01"
            session_id = "sess_seq_01"

            await service.create_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state={"media_dir": str(sample_media_dir)},
            )

            # Wire root_greeter with ingestion_agent sub-agent
            ingestion = create_ingestion_agent(offline=True)
            root = create_root_greeter_agent(
                name="root_greeter",
                sub_agents=[ingestion],
                offline=True,
            )

            runner = Runner(
                app_name=settings.APP_NAME,
                agent=root,
                session_service=service,
            )

            # Send theme in turn 1
            turn_events: list[Event] = []
            async for ev in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(parts=[types.Part.from_text(text="Kayaking along Emerald Lake")]),
            ):
                turn_events.append(ev)

            texts = [e.content.parts[0].text for e in turn_events if e.content and e.content.parts]
            # Verify root greeter locked in theme and transferred
            assert any("Kayaking along Emerald Lake" in t for t in texts)
            assert any("Handing off to Ingestion Agent" in t for t in texts)
            # Verify ingestion agent picked it up
            assert any("[IngestionAgent] Ready!" in t for t in texts)

            # Verify session state has theme
            sess = await service.get_session(
                app_name=settings.APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            assert sess.state["theme"] == "Kayaking along Emerald Lake"

        asyncio.run(_test())


class TestMainCLIIngestDir:
    """Test CLI --ingest-dir flag execution in main.py."""

    def test_cli_ingest_dir_success(self, sample_media_dir, monkeypatch, capsys):
        monkeypatch.setattr(
            sys,
            "argv",
            ["main.py", "--ingest-dir", str(sample_media_dir)],
        )
        main()
        captured = capsys.readouterr()
        assert "Media Ingestion & Clip Manifest Construction (SPEC-006)" in captured.out
        assert "Ingested 2 clip(s)" in captured.out
        assert "vlog_intro.mp4" in captured.out
        assert "drone_shot.mov" in captured.out
        assert "Clip manifest successfully assembled with 2 entries." in captured.out

    def test_cli_ingest_dir_invalid_path(self, tmp_path, monkeypatch, capsys):
        missing = str(tmp_path / "non_existent_folder")
        monkeypatch.setattr(sys, "argv", ["main.py", "--ingest-dir", missing])

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Ingestion failed" in captured.err
