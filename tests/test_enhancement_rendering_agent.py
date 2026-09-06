"""Unit tests for SPEC-020: Enhancement & Rendering Agent (agents/enhancement_rendering.py)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from agents.enhancement_rendering import (
    ENHANCEMENT_RENDERING_INSTRUCTION,
    EnhancementRenderingAgent,
    create_enhancement_rendering_agent,
    enhancement_rendering_agent,
)
from schemas.edl import EDLEntry, EditDecisionList, TransitionIntent
from schemas.media import ClipManifestEntry
from tools.audio_ducking import apply_audio_ducking
from tools.transition_renderer import render_with_transitions
from tools.whisper_captioning import generate_captions


@pytest.fixture
def sample_manifest() -> list[ClipManifestEntry]:
    return [
        ClipManifestEntry(
            clip_id="clip_01",
            absolute_path="C:/media/clip_01.mp4",
            file_name="clip_01.mp4",
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
def sample_edl() -> EditDecisionList:
    return EditDecisionList(
        entries=[
            EDLEntry(
                file_reference="clip_01",
                start_trim=0.0,
                end_trim=5.0,
                scene_rationale="Highlight segment",
                transition_intent=TransitionIntent.CUT,
            )
        ]
    )


class TestEnhancementRenderingAgentInit:
    """Tests for agent initialization and metadata."""

    def test_agent_initialization(self):
        agent = EnhancementRenderingAgent(offline=True)
        assert agent.name == "enhancement_rendering_agent"
        assert agent.instruction == ENHANCEMENT_RENDERING_INSTRUCTION
        assert agent.offline is True
        tool_names = [t.__name__ for t in agent.tools]
        assert "render_with_transitions" in tool_names
        assert "apply_audio_ducking" in tool_names
        assert "generate_captions" in tool_names

    def test_factory_and_singleton(self):
        agent = create_enhancement_rendering_agent(offline=True)
        assert isinstance(agent, EnhancementRenderingAgent)
        assert agent.name == "enhancement_rendering_agent"
        assert enhancement_rendering_agent is not None
        assert isinstance(enhancement_rendering_agent, EnhancementRenderingAgent)


class TestExecuteRenderingPipeline:
    """Tests for execute_rendering_pipeline."""

    def test_missing_edl_raises_error(self, sample_manifest):
        agent = EnhancementRenderingAgent(offline=True)
        state = {"clip_manifest": sample_manifest}
        with pytest.raises(ValueError, match="Missing Edit Decision List"):
            agent.execute_rendering_pipeline(state)

    def test_missing_manifest_raises_error(self, sample_edl):
        agent = EnhancementRenderingAgent(offline=True)
        state = {"edl": sample_edl}
        with pytest.raises(ValueError, match="Missing Clip Manifest"):
            agent.execute_rendering_pipeline(state)

    @patch("agents.enhancement_rendering.render_with_transitions")
    @patch("agents.enhancement_rendering.apply_audio_ducking")
    @patch("agents.enhancement_rendering.generate_captions")
    def test_pipeline_with_music_and_captions(
        self,
        mock_captions,
        mock_ducking,
        mock_transitions,
        sample_edl,
        sample_manifest,
        tmp_path,
    ):
        music_file = tmp_path / "music.mp3"
        music_file.write_bytes(b"dummy audio")
        mock_trans_video = tmp_path / "step1.mp4"
        mock_trans_video.write_bytes(b"video step 1")
        mock_duck_video = tmp_path / "step2.mp4"
        mock_duck_video.write_bytes(b"video step 2")
        mock_cap_video = tmp_path / "step3.mp4"
        mock_cap_video.write_bytes(b"video step 3")
        mock_srt = tmp_path / "subs.srt"
        mock_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")

        mock_transitions.return_value = str(mock_trans_video)
        mock_ducking.return_value = str(mock_duck_video)
        mock_captions.return_value = {
            "captioned_video_path": str(mock_cap_video),
            "srt_path": str(mock_srt),
        }

        agent = EnhancementRenderingAgent(offline=True)
        state = {
            "edl": sample_edl,
            "clip_manifest": sample_manifest,
            "music_path": str(music_file),
        }

        out_target = tmp_path / "custom_final.mp4"
        result = agent.execute_rendering_pipeline(state, output_path=out_target)

        assert result["final_video_path"] == str(out_target)
        assert result["has_ducking"] is True
        assert result["has_captions"] is True
        assert state["final_video_path"] == str(out_target)
        assert state["rendering_complete"] is True
        assert out_target.is_file()

        mock_transitions.assert_called_once()
        mock_ducking.assert_called_once()
        mock_captions.assert_called_once()

    @patch("agents.enhancement_rendering.render_with_transitions")
    @patch("agents.enhancement_rendering.apply_audio_ducking")
    @patch("agents.enhancement_rendering.generate_captions")
    def test_pipeline_without_music(
        self,
        mock_captions,
        mock_ducking,
        mock_transitions,
        sample_edl,
        sample_manifest,
        tmp_path,
    ):
        mock_trans_video = tmp_path / "step1.mp4"
        mock_trans_video.write_bytes(b"video step 1")
        mock_cap_video = tmp_path / "step3.mp4"
        mock_cap_video.write_bytes(b"video step 3")

        mock_transitions.return_value = str(mock_trans_video)
        mock_captions.return_value = {
            "captioned_video_path": str(mock_cap_video),
            "srt_path": None,
        }

        agent = EnhancementRenderingAgent(offline=True)
        state = {
            "edl": sample_edl,
            "clip_manifest": sample_manifest,
        }

        result = agent.execute_rendering_pipeline(state)

        assert result["has_ducking"] is False
        assert result["has_captions"] is True
        mock_ducking.assert_not_called()
        assert Path(result["final_video_path"]).is_file()

    @patch("agents.enhancement_rendering.render_with_transitions")
    @patch("agents.enhancement_rendering.apply_audio_ducking", side_effect=RuntimeError("Ducking error"))
    @patch("agents.enhancement_rendering.generate_captions")
    def test_pipeline_ducking_failure_graceful_degradation(
        self,
        mock_captions,
        mock_ducking,
        mock_transitions,
        sample_edl,
        sample_manifest,
        tmp_path,
    ):
        music_file = tmp_path / "music.mp3"
        music_file.write_bytes(b"audio")
        mock_trans_video = tmp_path / "step1.mp4"
        mock_trans_video.write_bytes(b"video step 1")
        mock_cap_video = tmp_path / "step3.mp4"
        mock_cap_video.write_bytes(b"video step 3")

        mock_transitions.return_value = str(mock_trans_video)
        mock_captions.return_value = {
            "captioned_video_path": str(mock_cap_video),
            "srt_path": None,
        }

        agent = EnhancementRenderingAgent(offline=True)
        state = {
            "edl": sample_edl,
            "clip_manifest": sample_manifest,
            "music_path": str(music_file),
        }

        result = agent.execute_rendering_pipeline(state)
        # Ducking should degrade gracefully and pipeline should succeed
        assert result["has_ducking"] is False
        assert result["has_captions"] is True
        assert Path(result["final_video_path"]).is_file()

    @patch("agents.enhancement_rendering.render_with_transitions")
    @patch("agents.enhancement_rendering.generate_captions", side_effect=RuntimeError("Whisper error"))
    def test_pipeline_captioning_failure_graceful_degradation(
        self,
        mock_captions,
        mock_transitions,
        sample_edl,
        sample_manifest,
        tmp_path,
    ):
        mock_trans_video = tmp_path / "step1.mp4"
        mock_trans_video.write_bytes(b"video step 1")

        mock_transitions.return_value = str(mock_trans_video)

        agent = EnhancementRenderingAgent(offline=True)
        state = {
            "edl": sample_edl,
            "clip_manifest": sample_manifest,
        }

        result = agent.execute_rendering_pipeline(state)
        # Captioning degradation: video still finalizes
        assert result["has_captions"] is False
        assert Path(result["final_video_path"]).is_file()


@pytest.mark.anyio
class TestAgentRunAsyncOffline:
    """Tests for offline agent invocation turn."""

    @patch("agents.enhancement_rendering.render_with_transitions")
    @patch("agents.enhancement_rendering.generate_captions")
    async def test_run_async_offline_success(
        self,
        mock_captions,
        mock_transitions,
        sample_edl,
        sample_manifest,
        tmp_path,
    ):
        mock_trans_video = tmp_path / "step1.mp4"
        mock_trans_video.write_bytes(b"v1")
        mock_cap_video = tmp_path / "step3.mp4"
        mock_cap_video.write_bytes(b"v3")

        mock_transitions.return_value = str(mock_trans_video)
        mock_captions.return_value = {
            "captioned_video_path": str(mock_cap_video),
            "srt_path": None,
        }

        class MockSession:
            def __init__(self):
                self.state = {
                    "edl": sample_edl,
                    "clip_manifest": sample_manifest,
                }

        class MockContext:
            def __init__(self):
                self.session = MockSession()

        agent = EnhancementRenderingAgent(offline=True)
        ctx = MockContext()

        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)

        assert len(events) == 1
        ev = events[0]
        assert ev.author == "enhancement_rendering_agent"
        assert "Rendering Complete" in ev.content.parts[0].text
        assert ev.actions.state_delta["rendering_complete"] is True
        assert "final_video_path" in ev.actions.state_delta

    async def test_run_async_offline_error(self):
        class MockSession:
            def __init__(self):
                self.state = {}

        class MockContext:
            def __init__(self):
                self.session = MockSession()

        agent = EnhancementRenderingAgent(offline=True)
        ctx = MockContext()

        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)

        assert len(events) == 1
        ev = events[0]
        assert "Rendering failed" in ev.content.parts[0].text
        assert ev.actions.state_delta["rendering_complete"] is False
