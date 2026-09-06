"""Unit & Integration Tests for SPEC-025: Root Workflow Agent (agents/root_pipeline.py).

Tests:
1. Agent hierarchy and exact order of the 5 sub-agents in `sub_agents`:
   - RootGreeterAgent
   - IngestionAgent
   - LoopAgent (CurationAgent -> ReviewerAgent -> CriticAgent)
   - ReviewOrchestratorAgent
   - EnhancementRenderingAgent
2. Security callbacks registration:
   - Verifies input_guardrail_callback, before_tool_sandbox_callback,
     after_tool_error_recovery_callback, on_tool_error_recovery_callback,
     and schema_validation_callback across all constituent sub-agents.
3. Session state passthrough across boundaries in offline mode:
   `theme` -> `clip_manifest` -> `edl` -> `otio_file_path` / `review_status` -> `final_video_path`.
4. Helper functions and execution methods:
   - `execute_pipeline()` synchronous execution.
   - `run_pipeline_async()` asynchronous execution.
5. Error handling and graceful degradation:
   - Missing input media directory.
   - Missing EDL handling.
   - Exception handling with `raise_on_error=False` returning failure dict.
6. CLI integration:
   - `--pipeline` and `--run-pipeline` argument parsing and execution.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from google.adk.agents import Agent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types

from agents import (
    CurationAgent,
    EnhancementRenderingAgent,
    IngestionAgent,
    ReviewOrchestratorAgent,
    ReviewerAgent,
    RootGreeterAgent,
    RootWorkflowAgent,
    create_root_workflow_agent,
    execute_pipeline,
    get_root_workflow_agent,
    root_workflow_agent,
    run_pipeline_async,
)
from callbacks import (
    after_tool_error_recovery_callback,
    before_tool_sandbox_callback,
    input_guardrail_callback,
    on_tool_error_recovery_callback,
    schema_validation_callback,
)
from config import settings
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry


@pytest.fixture
def sample_manifest() -> list[ClipManifestEntry]:
    return [
        ClipManifestEntry(
            clip_id="clip_001",
            absolute_path="C:/media/clip_001.mp4",
            file_name="clip_001.mp4",
            duration_seconds=10.0,
            frame_rate=30.0,
            resolution="1920x1080",
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
            has_audio=True,
        )
    ]


@pytest.fixture
def sample_edl_entry() -> EDLEntry:
    return EDLEntry(
        file_reference="clip_001",
        start_trim=1.0,
        end_trim=6.0,
        scene_rationale="Highlight action scene",
        transition_intent=TransitionIntent.CUT,
        engagement_score=9.0,
    )


class TestRootWorkflowAgentHierarchy:
    """Validate hierarchy, subagent composition, and ordering for RootWorkflowAgent."""

    def test_inheritance_and_type(self):
        agent = create_root_workflow_agent(offline=True)
        assert isinstance(agent, SequentialAgent)
        assert isinstance(agent, RootWorkflowAgent)
        assert isinstance(agent, BaseAgent)
        assert agent.name == "videoguru_root_pipeline"

    def test_singleton_and_accessor(self):
        assert root_workflow_agent is not None
        assert isinstance(root_workflow_agent, RootWorkflowAgent)
        assert get_root_workflow_agent() is root_workflow_agent

    def test_five_subagents_in_exact_order(self):
        agent = create_root_workflow_agent(offline=True)
        assert len(agent.sub_agents) == 5

        # 1. RootGreeterAgent
        assert isinstance(agent.sub_agents[0], RootGreeterAgent)
        assert agent.sub_agents[0].name == "root_greeter"

        # 2. IngestionAgent
        assert isinstance(agent.sub_agents[1], IngestionAgent)
        assert agent.sub_agents[1].name == "ingestion_agent"

        # 3. LoopAgent
        assert isinstance(agent.sub_agents[2], LoopAgent)
        assert agent.sub_agents[2].name == "video_editing_loop"
        loop_sub = agent.sub_agents[2].sub_agents
        assert len(loop_sub) == 3
        assert isinstance(loop_sub[0], CurationAgent)
        assert isinstance(loop_sub[1], ReviewerAgent)
        from agents.critic import CriticAgent
        assert isinstance(loop_sub[2], CriticAgent)

        # 4. ReviewOrchestratorAgent
        assert isinstance(agent.sub_agents[3], ReviewOrchestratorAgent)
        assert agent.sub_agents[3].name == "review_orchestrator_agent"

        # 5. EnhancementRenderingAgent
        assert isinstance(agent.sub_agents[4], EnhancementRenderingAgent)
        assert agent.sub_agents[4].name == "enhancement_rendering_agent"

    def test_custom_name_and_subagents(self):
        custom_sub = [RootGreeterAgent(name="custom_greeter", offline=True)]
        agent = create_root_workflow_agent(name="custom_pipeline", offline=True, sub_agents=custom_sub)
        assert agent.name == "custom_pipeline"
        assert len(agent.sub_agents) == 1
        assert agent.sub_agents[0].name == "custom_greeter"


class TestSecurityCallbacksRegistration:
    """Validate that all 4 Phase VI security callbacks are attached across the pipeline."""

    def test_callbacks_registered_on_all_llm_subagents(self):
        agent = create_root_workflow_agent(offline=True, enable_security_callbacks=True)

        # Gather all LLM Agent instances across the hierarchy
        llm_agents = [
            agent.sub_agents[0],  # greeter
            agent.sub_agents[1],  # ingestion
            agent.sub_agents[2].sub_agents[0],  # curation
            agent.sub_agents[2].sub_agents[1],  # reviewer
            agent.sub_agents[2].sub_agents[2],  # critic
            agent.sub_agents[3],  # review orchestrator
            agent.sub_agents[4],  # enhancement rendering
        ]

        for ag in llm_agents:
            assert hasattr(ag, "before_model_callback")
            assert input_guardrail_callback in ag.before_model_callback, f"Missing input_guardrail on {ag.name}"

            assert hasattr(ag, "before_tool_callback")
            assert before_tool_sandbox_callback in ag.before_tool_callback, f"Missing tool_sandbox on {ag.name}"

            assert hasattr(ag, "after_tool_callback")
            assert after_tool_error_recovery_callback in ag.after_tool_callback, f"Missing error_recovery on {ag.name}"

            assert hasattr(ag, "on_tool_error_callback")
            assert on_tool_error_recovery_callback in ag.on_tool_error_callback, f"Missing on_tool_error on {ag.name}"

            assert hasattr(ag, "after_model_callback")
            assert schema_validation_callback in ag.after_model_callback, f"Missing schema_validation on {ag.name}"

    def test_disable_security_callbacks_flag(self):
        agent = create_root_workflow_agent(offline=True, enable_security_callbacks=False)
        # Verify callbacks were not attached when explicitly disabled
        assert getattr(agent.sub_agents[0], "before_model_callback", None) is None


class TestSessionStatePassthrough:
    """Validate complete session state passthrough across all 5 agents in offline mode."""

    def test_end_to_end_state_passthrough(self, sample_manifest, sample_edl_entry, tmp_path):
        async def _test():
            agent = create_root_workflow_agent(offline=True)
            session_service = InMemorySessionService()
            session_id = f"test_e2e_{tmp_path.name}"

            output_file = tmp_path / "final.mp4"
            staging_otio = tmp_path / "timeline.otio"

            with patch("agents.ingestion.build_clip_manifest", return_value=sample_manifest), \
                 patch("tools.curation_tools.analyze_clip", return_value=sample_edl_entry), \
                 patch("agents.review_orchestrator.edl_to_otio", return_value=str(staging_otio)), \
                 patch.object(EnhancementRenderingAgent, "execute_rendering_pipeline", return_value={
                     "final_video_path": str(output_file),
                     "transition_video_path": str(tmp_path / "xfade.mp4"),
                     "ducked_video_path": None,
                     "srt_path": str(tmp_path / "subtitles.srt"),
                     "captioned_video_path": str(output_file),
                     "has_ducking": False,
                     "has_captions": True,
                 }):

                result = await agent.run_pipeline_async(
                    user_prompt="Epic Mountain Hiking Vlog",
                    session_id=session_id,
                    initial_state={"media_dir": str(tmp_path), "auto_approve": True},
                    session_service=session_service,
                )

            assert result["success"] is True
            assert result["session_id"] == session_id
            assert result["theme"] == "Epic Mountain Hiking Vlog"
            assert result["clip_manifest"] is not None
            assert len(result["clip_manifest"]) == 1
            assert result["edl"] is not None
            assert len(result["edl"]) == 1
            assert result["otio_file_path"] == str(staging_otio)
            assert result["review_status"] == "approved"
            assert result["final_video_path"] == str(output_file)
            assert result["rendering_complete"] is True
            assert result["error"] is None

            # Verify state passthrough in session
            state = result["session_state"]
            assert state["theme"] == "Epic Mountain Hiking Vlog"
            assert "clip_manifest" in state
            assert "edl" in state
            assert "otio_file_path" in state
            assert state["final_video_path"] == str(output_file)
            assert state["rendering_complete"] is True

        asyncio.run(_test())

    def test_sync_execute_pipeline_helper(self, sample_manifest, sample_edl_entry, tmp_path):
        agent = create_root_workflow_agent(offline=True)
        session_service = InMemorySessionService()
        session_id = f"test_sync_{tmp_path.name}"
        output_file = tmp_path / "final_sync.mp4"

        with patch("agents.ingestion.build_clip_manifest", return_value=sample_manifest), \
             patch("tools.curation_tools.analyze_clip", return_value=sample_edl_entry), \
             patch("tools.otio_converter.edl_to_otio", return_value=str(tmp_path / "sync.otio")), \
             patch.object(EnhancementRenderingAgent, "execute_rendering_pipeline", return_value={
                 "final_video_path": str(output_file),
                 "transition_video_path": str(tmp_path / "xfade.mp4"),
                 "ducked_video_path": None,
                 "srt_path": None,
                 "captioned_video_path": str(output_file),
                 "has_ducking": False,
                 "has_captions": False,
             }):

            result = execute_pipeline(
                agent=agent,
                user_prompt="Cooking Italian Pasta in Rome",
                session_id=session_id,
                initial_state={"media_dir": str(tmp_path), "auto_approve": True},
                session_service=session_service,
            )

        assert result["success"] is True
        assert result["theme"] == "Cooking Italian Pasta in Rome"
        assert result["final_video_path"] == str(output_file)
        assert result["rendering_complete"] is True

    def test_top_level_run_pipeline_async_helper(self, sample_manifest, sample_edl_entry, tmp_path):
        async def _test():
            agent = create_root_workflow_agent(offline=True)
            session_service = InMemorySessionService()
            session_id = f"test_top_{tmp_path.name}"
            output_file = tmp_path / "final_top.mp4"

            with patch("agents.ingestion.build_clip_manifest", return_value=sample_manifest), \
                 patch("tools.curation_tools.analyze_clip", return_value=sample_edl_entry), \
                 patch("agents.review_orchestrator.edl_to_otio", return_value=str(tmp_path / "top.otio")), \
                 patch.object(EnhancementRenderingAgent, "execute_rendering_pipeline", return_value={
                     "final_video_path": str(output_file),
                     "transition_video_path": str(tmp_path / "xfade.mp4"),
                     "ducked_video_path": None,
                     "srt_path": None,
                     "captioned_video_path": str(output_file),
                     "has_ducking": False,
                     "has_captions": False,
                 }):

                result = await run_pipeline_async(
                    agent=agent,
                    user_prompt="Surfing in Hawaii",
                    session_id=session_id,
                    initial_state={"media_dir": str(tmp_path), "auto_approve": True},
                    session_service=session_service,
                )

            assert result["success"] is True
            assert result["theme"] == "Surfing in Hawaii"
            assert result["final_video_path"] == str(output_file)

        asyncio.run(_test())

    def test_auto_approve_disabled(self, sample_manifest, sample_edl_entry, tmp_path):
        async def _test():
            agent = create_root_workflow_agent(offline=True)
            session_service = InMemorySessionService()
            session_id = f"test_pending_{tmp_path.name}"

            with patch("agents.ingestion.build_clip_manifest", return_value=sample_manifest), \
                 patch("tools.curation_tools.analyze_clip", return_value=sample_edl_entry), \
                 patch("agents.review_orchestrator.edl_to_otio", return_value=str(tmp_path / "p.otio")), \
                 patch.object(EnhancementRenderingAgent, "execute_rendering_pipeline", return_value={
                     "final_video_path": str(tmp_path / "final.mp4"),
                     "transition_video_path": str(tmp_path / "xfade.mp4"),
                     "ducked_video_path": None,
                     "srt_path": None,
                     "captioned_video_path": str(tmp_path / "final.mp4"),
                     "has_ducking": False,
                     "has_captions": False,
                 }):

                result = await agent.run_pipeline_async(
                    user_prompt="Surfing in Hawaii",
                    session_id=session_id,
                    initial_state={"media_dir": str(tmp_path)},
                    session_service=session_service,
                    auto_approve=False,
                )

            assert result["success"] is True
            assert result["review_status"] == "pending"

        asyncio.run(_test())


class TestErrorHandlingAndDegradation:
    """Validate graceful error handling and degradation during pipeline execution."""

    def test_pipeline_catches_error_without_raising_by_default(self, tmp_path):
        async def _test():
            agent = create_root_workflow_agent(offline=True)
            session_service = InMemorySessionService()

            with patch("agents.ingestion.build_clip_manifest", side_effect=RuntimeError("Disk failure during scan")):
                result = await agent.run_pipeline_async(
                    user_prompt="Road trip vlog",
                    session_id=f"test_err_{tmp_path.name}",
                    initial_state={"media_dir": str(tmp_path)},
                    session_service=session_service,
                    raise_on_error=False,
                )

            assert result["success"] is False
            assert "Disk failure during scan" in result["error"]
            assert result["rendering_complete"] is False

        asyncio.run(_test())

    def test_pipeline_raises_when_raise_on_error_true(self, tmp_path):
        async def _test():
            agent = create_root_workflow_agent(offline=True)
            session_service = InMemorySessionService()

            with patch("agents.ingestion.build_clip_manifest", side_effect=RuntimeError("Fatal error")):
                with pytest.raises(RuntimeError, match="Fatal error"):
                    await agent.run_pipeline_async(
                        user_prompt="Road trip vlog",
                        session_id=f"test_err_raise_{tmp_path.name}",
                        initial_state={"media_dir": str(tmp_path)},
                        session_service=session_service,
                        raise_on_error=True,
                    )

        asyncio.run(_test())


class TestCliIntegration:
    """Validate main.py CLI integration for --pipeline and --run-pipeline."""

    def test_cli_pipeline_flag_help(self):
        cmd = [sys.executable, "main.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "--pipeline" in result.stdout
        assert "--run-pipeline" in result.stdout
        assert "--input-dir" in result.stdout
        assert "--auto-approve" in result.stdout

    def test_cli_pipeline_offline_execution(self, tmp_path):
        # Run CLI in offline mode targeting a temp dir
        cmd = [
            sys.executable,
            "main.py",
            "--pipeline",
            "--theme",
            "Weekend Camping Highlights",
            "--input-dir",
            str(tmp_path),
            "--offline",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Even without clips, pipeline should report theme and graceful completion or ingestion status
        assert "VideoGuru: Root Workflow Pipeline Assembly" in result.stdout
        assert "Weekend Camping Highlights" in result.stdout
