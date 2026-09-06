"""Unit and integration tests for SPEC-009: Curation Agent.

Covers:
- Assembling Edit Decision List (EDL) from Clip Manifest via assemble_edl_from_manifest
- Reference integrity and duration aggregation in curated EDLs
- Error handling for empty manifests and empty themes
- Integration of editorial/critic revision feedback in curation
- Custom ADK tool curate_edit_decision_list updating ToolContext session state
- Deserialization helper get_edl_from_state
- CurationAgent instantiation, tools registration, and ADK Runner execution
- .env loading verification via python-dotenv in config/settings
- CLI --curate dispatch via main.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.adk.events import EventActions
from google.adk.runners import Runner
from google.adk.tools import ToolContext
from google.genai import types

from agents.curation import (
    CURATION_INSTRUCTION,
    CurationAgent,
    create_curation_agent,
    curation_agent,
)
from config import settings
from main import main
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from services import get_session_manager, get_session_service
from tools.clip_metadata import find_ffprobe_executable
from tools.curation_tools import (
    assemble_edl_from_manifest,
    curate_edit_decision_list,
    get_edl_from_state,
)


@pytest.fixture(scope="module")
def sample_curation_media(tmp_path_factory) -> dict[str, Any]:
    """Generate two small synthetic MP4 video clips for curation testing."""
    ffprobe_bin = find_ffprobe_executable()
    ffmpeg_bin = str(Path(ffprobe_bin).parent / "ffmpeg.exe")
    if not Path(ffmpeg_bin).exists():
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    temp_dir = tmp_path_factory.mktemp("curation_media_dir")
    clip_1 = temp_dir / "clip_action.mp4"
    clip_2 = temp_dir / "clip_scenic.mp4"

    # Clip 1: 2.0s 1280x720 video with audio
    cmd_1 = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2.0:size=1280x720:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=800:duration=2.0",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(clip_1),
    ]
    subprocess.run(cmd_1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Clip 2: 1.5s 1920x1080 video with audio
    cmd_2 = [
        ffmpeg_bin,
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1.5:size=1920x1080:rate=24",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=400:duration=1.5",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(clip_2),
    ]
    subprocess.run(cmd_2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    manifest_entries = [
        ClipManifestEntry(
            clip_id="vid_action_001",
            absolute_path=str(clip_1),
            file_name="clip_action.mp4",
            duration_seconds=2.0,
            frame_rate=30.0,
            resolution="1280x720",
            width=1280,
            height=720,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=clip_1.stat().st_size,
        ),
        ClipManifestEntry(
            clip_id="vid_scenic_002",
            absolute_path=str(clip_2),
            file_name="clip_scenic.mp4",
            duration_seconds=1.5,
            frame_rate=24.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
            file_size_bytes=clip_2.stat().st_size,
        ),
    ]

    return {
        "dir": temp_dir,
        "clip_1": clip_1,
        "clip_2": clip_2,
        "manifest": manifest_entries,
    }


def _create_mock_tool_context(initial_state: dict[str, Any] | None = None) -> ToolContext:
    actions = EventActions()
    mock_invocation = MagicMock()
    mock_invocation.session.state = initial_state if initial_state is not None else {}
    return ToolContext(invocation_context=mock_invocation, event_actions=actions)


class TestAssembleEdlFromManifest:
    """Test assemble_edl_from_manifest logic."""

    def test_assemble_edl_valid_manifest(self, sample_curation_media):
        manifest = sample_curation_media["manifest"]
        edl = assemble_edl_from_manifest(
            manifest=manifest,
            theme="Extreme sports adrenaline",
            offline=True,
        )

        assert isinstance(edl, EditDecisionList)
        assert len(edl) == 2
        assert edl.clip_references == {"vid_action_001", "vid_scenic_002"}
        assert edl.total_duration > 0.0

        for cut in edl:
            assert cut.start_trim >= 0.0
            assert cut.end_trim > cut.start_trim
            assert cut.transition_intent in TransitionIntent
            assert cut.engagement_score is not None
            assert 0.0 <= cut.engagement_score <= 10.0
            assert "Extreme sports adrenaline" in cut.scene_rationale

    def test_assemble_edl_empty_manifest_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot assemble EDL from an empty clip manifest"):
            assemble_edl_from_manifest(manifest=[], theme="Vlog", offline=True)

    def test_assemble_edl_empty_theme_raises_value_error(self, sample_curation_media):
        manifest = sample_curation_media["manifest"]
        with pytest.raises(ValueError, match="Theme must be a non-empty string"):
            assemble_edl_from_manifest(manifest=manifest, theme="   ", offline=True)

    def test_assemble_edl_with_editorial_feedback(self, sample_curation_media):
        manifest = sample_curation_media["manifest"]
        edl = assemble_edl_from_manifest(
            manifest=manifest,
            theme="Cooking pasta",
            feedback="Cut faster during boiling step",
            offline=True,
        )

        for cut in edl:
            assert "Cooking pasta" in cut.scene_rationale
            assert "Cut faster during boiling step" in cut.scene_rationale


class TestCurateEditDecisionListTool:
    """Test curate_edit_decision_list ADK tool."""

    def test_tool_curates_and_updates_session_state(self, sample_curation_media):
        manifest_entries = sample_curation_media["manifest"]
        serialized_manifest = [e.model_dump() for e in manifest_entries]

        ctx = _create_mock_tool_context(
            {
                "clip_manifest": serialized_manifest,
                "theme": "Tokyo Night Walk",
            }
        )

        report = curate_edit_decision_list(tool_context=ctx, offline=True)

        assert "[CurationAgent] Successfully curated Edit Decision List" in report
        assert "Tokyo Night Walk" in report
        assert "vid_action_001" in report
        assert "vid_scenic_002" in report

        # Verify session state was populated
        assert "edl" in ctx.state
        assert isinstance(ctx.state["edl"], list)
        assert len(ctx.state["edl"]) == 2
        assert "edl_total_duration" in ctx.state
        assert ctx.state["edl_total_duration"] > 0.0

    def test_tool_missing_manifest_raises_value_error(self):
        ctx = _create_mock_tool_context({"theme": "Hiking"})
        with pytest.raises(ValueError, match="No clip manifest found in session state"):
            curate_edit_decision_list(tool_context=ctx, offline=True)

    def test_get_edl_from_state_helpers(self, sample_curation_media):
        manifest_entries = sample_curation_media["manifest"]
        serialized_manifest = [e.model_dump() for e in manifest_entries]

        ctx = _create_mock_tool_context(
            {
                "clip_manifest": serialized_manifest,
                "theme": "Desert safari",
            }
        )
        curate_edit_decision_list(tool_context=ctx, offline=True)

        edl = get_edl_from_state(ctx.state)
        assert isinstance(edl, EditDecisionList)
        assert len(edl) == 2
        assert edl.clip_references == {"vid_action_001", "vid_scenic_002"}

        # Empty state returns None
        assert get_edl_from_state({}) is None
        assert get_edl_from_state({"edl": None}) is None


class TestCurationAgentClass:
    """Test CurationAgent class, instructions, tools, and execution."""

    def test_curation_agent_attributes(self):
        agent = create_curation_agent(offline=True)
        assert agent.name == "curation_agent"
        assert agent.offline is True
        assert CURATION_INSTRUCTION in agent.instruction
        assert "edl" in agent.instruction

        tool_names = [t.__name__ if hasattr(t, "__name__") else str(t) for t in agent.tools]
        assert "curate_edit_decision_list" in tool_names
        assert "analyze_clip" in tool_names

    def test_singleton_curation_agent_instance(self):
        assert curation_agent is not None
        assert curation_agent.name == "curation_agent"

    def test_curation_agent_run_turn_success(self, sample_curation_media):
        async def _test():
            session_mgr = get_session_manager()
            session_id = "test_curation_turn_001"
            user_id = "test_creator"

            manifest_entries = sample_curation_media["manifest"]
            serialized_manifest = [e.model_dump() for e in manifest_entries]

            await session_mgr.create_session(
                user_id=user_id,
                session_id=session_id,
                state={
                    "theme": "Tropical island vacation",
                    "clip_manifest": serialized_manifest,
                },
            )

            agent = create_curation_agent(offline=True)
            runner = Runner(
                app_name=settings.APP_NAME,
                agent=agent,
                session_service=get_session_service(),
            )

            events = []
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(parts=[types.Part.from_text(text="Curate video")]),
            ):
                events.append(event)

            assert len(events) >= 1
            response_texts = [
                p.text
                for e in events
                if e.content and e.content.parts
                for p in e.content.parts
                if getattr(p, "text", None)
            ]
            full_response = " ".join(response_texts)
            assert "[CurationAgent] Successfully assembled initial Edit Decision List (EDL)" in full_response
            assert "Tropical island vacation" in full_response

            # Check that state was updated with EDL
            final_state = await session_mgr.get_state(user_id=user_id, session_id=session_id)
            assert "edl" in final_state
            assert len(final_state["edl"]) == 2
            assert final_state["edl_total_duration"] > 0

        asyncio.run(_test())

    def test_curation_agent_run_turn_missing_manifest(self):
        async def _test():
            session_mgr = get_session_manager()
            session_id = "test_curation_turn_missing_manifest"
            user_id = "test_creator"

            await session_mgr.create_session(
                user_id=user_id,
                session_id=session_id,
                state={"theme": "Snowboarding"},
            )

            agent = create_curation_agent(offline=True)
            runner = Runner(
                app_name=settings.APP_NAME,
                agent=agent,
                session_service=get_session_service(),
            )

            events = []
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(parts=[types.Part.from_text(text="Curate video")]),
            ):
                events.append(event)

            response_texts = [
                p.text
                for e in events
                if e.content and e.content.parts
                for p in e.content.parts
                if getattr(p, "text", None)
            ]
            full_response = " ".join(response_texts)
            assert "Cannot curate video: No clip manifest found" in full_response

        asyncio.run(_test())


class TestEnvLoading:
    """Test suite validating python-dotenv integration in settings.py."""

    def test_settings_exports_gemini_api_key_attribute(self):
        assert hasattr(settings, "GEMINI_API_KEY")

    def test_load_dotenv_loads_file(self, tmp_path, monkeypatch):
        test_env = tmp_path / ".env"
        test_env.write_text("GEMINI_API_KEY=test_secret_key_12345\n")

        from dotenv import load_dotenv

        load_dotenv(test_env, override=True)
        assert settings.os.getenv("GEMINI_API_KEY") == "test_secret_key_12345"


class TestMainCLICuration:
    """Test suite for CLI --curate command."""

    def test_cli_curate_offline(self, sample_curation_media, capsys):
        media_dir = str(sample_curation_media["dir"])
        test_args = [
            "main.py",
            "--curate",
            "--ingest-dir",
            media_dir,
            "--theme",
            "Weekend getaway in mountains",
            "--offline",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        out = capsys.readouterr().out
        assert "VideoGuru: Curating Edit Decision List (SPEC-009)" in out
        assert "Weekend getaway in mountains" in out
        assert "Curated EDL successfully assembled with 2 cut(s)" in out
        assert "clip_action.mp4" in out or "vid_action_001" in out or "vid_" in out
        assert "Total Duration:" in out

    def test_cli_curate_empty_dir_exits_with_error(self, tmp_path, capsys):
        empty_dir = tmp_path / "empty_folder"
        empty_dir.mkdir()
        test_args = [
            "main.py",
            "--curate",
            "--ingest-dir",
            str(empty_dir),
            "--offline",
        ]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "No video clips discovered" in combined
